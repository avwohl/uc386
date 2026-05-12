"""i386 emulator harness for uc386's NASM-bin output.

Loads a flat binary produced by `nasm -f bin` at virtual address 0
(matching NASM's default `org 0`), runs it under unicorn-engine in
32-bit protected-mode-ish, and intercepts INT 21h so the program's
DOS-style I/O calls reach a Python-side handler.

The implemented INT 21h functions cover what uc386's mini-libc and
the c-testsuite / Fujitsu / GCC-torture programs actually use:

    AH=02   putchar (AL)                       → emu.stdout
    AH=09   print '$'-terminated string (DS:EDX → emu.stdout
    AH=40   write handle (BX=fd, CX=count, DS:EDX=buf) → stdout/stderr
    AH=4C   exit (AL)
    AH=00   terminate (= exit 1)

Returns a `Result` with stdout text, stderr text, exit_code, timed_out
flag, and any error string from unicorn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import struct

# unicorn-engine is optional: required to actually run() / assemble_and_run()
# (the emulator), but NOT required for bundle_text / bundle_user_asm /
# _is_already_bundled (the compile-time libc embedding path that main.py
# pulls in). CI installs minimal pytest deps and skips unicorn — so we
# tolerate ImportError here and let run() raise a clean error instead.
try:
    import unicorn
    from unicorn import Uc, UC_ARCH_X86, UC_MODE_32, UC_HOOK_INTR, UcError
    from unicorn.x86_const import (
        UC_X86_REG_EAX, UC_X86_REG_EBX, UC_X86_REG_ECX, UC_X86_REG_EDX,
        UC_X86_REG_ESI, UC_X86_REG_EDI, UC_X86_REG_EBP, UC_X86_REG_ESP,
        UC_X86_REG_EIP, UC_X86_REG_EFLAGS,
    )
    _UNICORN_AVAILABLE = True
except ImportError:
    _UNICORN_AVAILABLE = False
    unicorn = None
    Uc = UcError = None
    UC_ARCH_X86 = UC_MODE_32 = UC_HOOK_INTR = None
    UC_X86_REG_EAX = UC_X86_REG_EBX = UC_X86_REG_ECX = UC_X86_REG_EDX = None
    UC_X86_REG_ESI = UC_X86_REG_EDI = UC_X86_REG_EBP = UC_X86_REG_ESP = None
    UC_X86_REG_EIP = UC_X86_REG_EFLAGS = None


# Memory layout
#   0x00000000 .. 0x00800000   code/data (8 MB) — the loaded binary lives here
#   0x00800000 .. 0x01000000   heap (8 MB, growable in principle)
#   0x000F0000 .. 0x000F1000   fake DOS PSP + env block (within CODE region)
#   0x00F00000 .. 0x00F01000   argv area (4 KB)
#   0x01000000 .. 0x01100000   stack (1 MB, top at 0x010FFFF0)
#
# PSP_BASE sits within the CODE region so we don't need a separate
# unicorn mapping. PMODE/W exposes DOS conventional memory directly
# via flat 32-bit selectors, so a real DOS extender hands the program
# linear addresses computed from segment*16 — same shape as our fake.
# 0x000F0000 = 960 KB, well clear of the loaded binary (mp.bin is
# ~280 KB) and below the 1 MB ceiling so segment*16 (where seg fits
# in 16 bits) can address it.
CODE_BASE = 0x00000000
CODE_SIZE = 0x00800000
HEAP_BASE = 0x00800000
HEAP_SIZE = 0x00800000
PSP_BASE = 0x000F0000
PSP_SIZE = 0x00001000
PSP_SEG = PSP_BASE >> 4          # 0xF000
ENV_OFFSET = 0x100               # PSP is 256 bytes; env block starts after
ENV_BASE = PSP_BASE + ENV_OFFSET
ENV_SEG = ENV_BASE >> 4          # 0xF010
ARGV_BASE = 0x00F00000
ARGV_SIZE = 0x00001000
STACK_BASE = 0x01000000
STACK_SIZE = 0x00100000
STACK_TOP = STACK_BASE + STACK_SIZE - 16


@dataclass
class Result:
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    timed_out: bool = False
    error: str | None = None
    # bytes consumed by INT 21h string ops, useful for diagnostics
    instructions_executed: int | None = None
    # Synthetic BIOS-tick counter for INT 1Ah AH=00h. Increments on
    # every read, simulating a monotonic clock; user code that uses
    # `time.ticks_ms()` / `sleep_ms` exercises this counter.
    _bios_tick: int = 0


def _read_cstr(uc: Uc, addr: int, max_len: int = 4096, term: bytes = b"\x00") -> bytes:
    """Read a `term`-terminated string starting at `addr`."""
    out = b""
    for _ in range(max_len):
        b = uc.mem_read(addr, 1)
        if b == term:
            break
        out += b
        addr += 1
    return out


def run(
    binary: bytes | Path,
    *,
    timeout_seconds: float = 10.0,
    instruction_limit: int = 50_000_000,
    stdin_bytes: bytes = b"",
    argv: list[str] | None = None,
    env: dict[str, str] | None = None,
    program_path: str | None = None,
    vfiles_init: dict[bytes, bytes] | None = None,
    net=None,
) -> Result:
    """Emulate a flat-binary i386 program; return its stdout + exit code.

    `vfiles_init` seeds the virtual file system with named files
    visible to fopen("name", "r"). Pass `{b"data.txt": b"hello\n"}`
    so a program can open and read `data.txt`.

    `env` populates a fake DOS environment block reachable through
    PSP[0x2C] (INT 21h AH=0x62 returns BX=PSP_seg). `program_path`
    sets the trailing argv[0] string DOS 3.0+ writes after the env
    block — used by libc's `_dos_argv0()` helper. Defaults to the
    first element of `argv` (or "PROGRAM.EXE" if argv is empty).

    `net` is an optional `dos_emu_netsim.NetworkSimulator` (or any
    object with `init_mac() -> bytes`, `send_frame(bytes) -> int`,
    and `recv_frame(int) -> bytes|None`). When provided, the program's
    INT 0x83 packet-driver calls (lib/i386_dos_libc.asm:`ethdrv_*`)
    are routed to it. Without one, INT 0x83 returns AL=1 (error) so
    `ethdrv_init` cleanly fails and the program can detect "no NIC".
    """
    if not _UNICORN_AVAILABLE:
        raise RuntimeError(
            "dos_emu.run() requires the `unicorn` package. "
            "Install with: pip install unicorn"
        )
    if isinstance(binary, Path):
        binary = binary.read_bytes()

    mu = Uc(UC_ARCH_X86, UC_MODE_32)
    mu.mem_map(CODE_BASE, CODE_SIZE)
    mu.mem_map(ARGV_BASE, ARGV_SIZE)
    mu.mem_map(STACK_BASE, STACK_SIZE)

    # Load the program at address 0 (matches NASM `-f bin` default org 0).
    mu.mem_write(CODE_BASE, binary)

    # Locate `pktdrv_int_invoke` in the loaded binary, if present, so
    # we can intercept its calls — see the long comment further down
    # where the UC_HOOK_CODE that does the intercept is installed.
    # The function is uc386 libc's wrapper that takes a cdecl
    # (int_num, regs[8]) and dispatches the INT. Its prologue is
    # distinctive: push ebp / mov ebp,esp / push esi / push edi /
    # push ebx / **push es** — that final `push es` (opcode 0x06) at
    # function entry only happens in this helper, so a 7-byte byte
    # pattern search finds the function reliably without symbol
    # tables (the binary is flat). Verified empirically: the prefix
    # appears at exactly one address in the MicroPython port build.
    _PKTDRV_INT_INVOKE_SIG = b"\x55\x89\xe5\x56\x57\x53\x06"
    pktdrv_int_invoke_addr = binary.find(_PKTDRV_INT_INVOKE_SIG)
    pktdrv_invoke_seg_arena = [0x4000]   # bump allocator state

    # When a NetworkSimulator is wired with Crynwr enabled, plant a
    # "PKT DRVR" signature in low memory so the binary's IVT scan
    # via DPMI INT 31h fn 0x0200 lands a real-looking handler. The
    # signature is just data — the actual INT is handled by the
    # Python-side hooks below, so the seg:offset returned from
    # 0x0200 doesn't need to point to executable code.
    pktdrv_int = (
        getattr(net, "crynwr_int_num", None)
        if net is not None else None
    )
    if pktdrv_int is not None:
        from .dos_emu_netsim import PKTDRV_HANDLER_LINEAR
        # `EB 09 90` = JMP SHORT +9 + NOP padding so the 8-byte
        # signature lands at offset +3 (per the Crynwr spec).
        mu.mem_write(PKTDRV_HANDLER_LINEAR,
                     b"\xeb\x09\x90PKT DRVR")

    # Build argv: a contiguous region with [argc+1 dwords of pointers]
    # followed by the null-terminated argv strings. argc lands in EAX
    # and the address of the pointer array lands in EBX before _start
    # runs; codegen's _start_stub pushes both onto the stack before
    # calling _main so cdecl `int main(int argc, char **argv)` sees
    # them at [ebp+8] / [ebp+12].
    if argv is None:
        argv = ["program"]
    argc = len(argv)
    ptr_array_bytes = 4 * (argc + 1)
    abuf = bytearray(ARGV_SIZE)
    cursor = ptr_array_bytes
    for i, arg in enumerate(argv):
        s = arg.encode("utf-8") + b"\x00"
        abuf[i * 4:(i + 1) * 4] = struct.pack("<I", ARGV_BASE + cursor)
        abuf[cursor:cursor + len(s)] = s
        cursor += len(s)
    # Final null pointer terminator already 0 from bytearray init.
    if cursor > ARGV_SIZE:
        return Result(error=f"argv too large: {cursor} > {ARGV_SIZE}")
    mu.mem_write(ARGV_BASE, bytes(abuf))

    # Fake DOS PSP + environment block. PSP[0x2C] holds the env
    # segment; the env block is `KEY=VAL\0` strings terminated by
    # an extra `\0`, then (DOS 3.0+) a 16-bit count word and the
    # program's full path as a `\0`-terminated string. INT 21h
    # AH=0x62 returns BX = PSP_seg so the program can find its
    # PSP without needing extender-specific calls.
    psp = bytearray(PSP_SIZE)
    # Standard PSP markers we don't actually use but might as well
    # populate for parity with real DOS:
    psp[0x00] = 0xCD  # INT 20h opcode (legacy CP/M-style exit)
    psp[0x01] = 0x20
    # PSP[0x2C..0x2D]: env segment (little-endian word).
    psp[0x2C] = ENV_SEG & 0xFF
    psp[0x2D] = (ENV_SEG >> 8) & 0xFF

    # Build the env block at ENV_BASE (offset ENV_OFFSET into the PSP
    # page). `env` defaults to empty so the block is just the
    # double-NUL terminator + program-path tail.
    env_buf = bytearray()
    if env:
        for key, val in env.items():
            entry = f"{key}={val}".encode("utf-8")
            if b"\x00" in entry:
                return Result(error=f"env entry contains NUL: {key!r}")
            env_buf.extend(entry)
            env_buf.append(0)
    env_buf.append(0)  # terminating empty string ("\0\0" overall)
    # DOS 3.0+ tail: count word (1) + program path (NUL-terminated).
    if program_path is None:
        program_path = argv[0] if argv else "PROGRAM.EXE"
    env_buf.extend(struct.pack("<H", 1))
    env_buf.extend(program_path.encode("utf-8"))
    env_buf.append(0)
    if ENV_OFFSET + len(env_buf) > PSP_SIZE:
        return Result(error=f"env block too large: {len(env_buf)} bytes")
    psp[ENV_OFFSET:ENV_OFFSET + len(env_buf)] = env_buf
    mu.mem_write(PSP_BASE, bytes(psp))

    # Initialize stack near the top of the stack region. Push a fake return
    # address (0xFFFFFFFF) so `ret` from the entry function ends up at an
    # unmapped location — we treat that as a clean exit. Entry doesn't
    # actually return for our test programs (they call INT 21h AH=4C), but
    # this protects against malformed code.
    esp = STACK_TOP
    mu.reg_write(UC_X86_REG_ESP, esp)
    mu.reg_write(UC_X86_REG_EBP, esp)
    mu.reg_write(UC_X86_REG_EAX, argc)
    mu.reg_write(UC_X86_REG_EBX, ARGV_BASE)

    res = Result()
    stdin_pos = [0]
    # Signal handler table: signum → handler addr. Populated by
    # libc's signal() via INT 21h AH=0x99. On hardware exceptions
    # (currently INT 0 / divide-by-zero → SIGFPE) we look up the
    # handler and dispatch by setting EIP to it.
    signal_handlers: dict[int, int] = {}
    # Virtual file system: name → bytearray. Persists for the lifetime
    # of the run, so a `fopen` for read after a `fopen`+`fclose` for
    # write returns the previously-written bytes. Caller can seed the
    # vfs via `vfiles_init` so the program can fopen pre-existing files.
    vfiles: dict[bytes, bytearray] = {}
    if vfiles_init:
        for name, content in vfiles_init.items():
            vfiles[name] = bytearray(content)
    # Virtual directory set. Stores directory names that mkdir
    # has created (or that vfiles_init populated for tests). Used
    # by INT 21h AH=0x39 (mkdir), 0x3A (rmdir), and 0x3B (chdir
    # — chdir validates the dir exists). Always contains "." and
    # the synthetic CWD's parents implicitly.
    vdirs: set[bytes] = set()
    # Synthetic current working directory. Reset to "C:\" at boot.
    # chdir updates this; getcwd reads it. We don't enforce real
    # path resolution here — tests interact with the vfs by full
    # filename only.
    vcwd = [b"C:\\"]
    # DTA address (set by INT 21h AH=0x1A); used by find-first/
    # find-next to write filename matches into the caller's buffer.
    vdta_addr = [0]
    # INT 21h AH=0x4B (EXEC) → AH=0x4D (Get Return Code) handoff: the
    # child process's exit code gets stashed by the EXEC handler and
    # returned by the next AH=0x4D call.
    child_exit_code = [0]
    # find-first/find-next iteration state. List of (filename_bytes,
    # is_dir_bool) entries left to enumerate after the initial
    # find-first call.
    vfind_state: list[tuple[bytes, bool]] = []
    # Open-file table: fd → {"name": bytes, "pos": int, "mode": str}.
    # Modes: "r" (read), "w" (write/truncate), "a" (append).
    vfd_table: dict[int, dict] = {}
    next_vfd = [3]
    tmpnam_seq = [0]

    def _vfile_open(name: bytes, mode: str) -> int:
        """Open or create a virtual file. Returns fd or -1 on error."""
        if mode == "w":
            vfiles[name] = bytearray()
        elif mode == "a":
            if name not in vfiles:
                vfiles[name] = bytearray()
        elif mode == "r":
            if name not in vfiles:
                return -1
        else:
            return -1
        fd = next_vfd[0]
        next_vfd[0] += 1
        pos = len(vfiles[name]) if mode == "a" else 0
        vfd_table[fd] = {"name": name, "pos": pos, "mode": mode}
        return fd

    def _vfile_close(fd: int) -> int:
        if fd in vfd_table:
            del vfd_table[fd]
            return 0
        return -1

    def _vfile_seek(fd: int, off: int, whence: int) -> int:
        """lseek(2) — backs INT 21h AH=0x42. Returns new pos or -1."""
        e = vfd_table.get(fd)
        if e is None:
            return -1
        buf = vfiles.get(e["name"], bytearray())
        if whence == 0:
            new = off
        elif whence == 1:
            new = e["pos"] + off
        elif whence == 2:
            new = len(buf) + off
        else:
            return -1
        if new < 0:
            return -1
        e["pos"] = new
        return new

    def _vfile_read(fd: int, addr: int, count: int) -> int:
        e = vfd_table.get(fd)
        if e is None or e["mode"] not in ("r",):
            return -1
        buf = vfiles.get(e["name"])
        if buf is None:
            return 0
        start = e["pos"]
        end = min(start + count, len(buf))
        chunk = bytes(buf[start:end])
        if chunk:
            mu.mem_write(addr, chunk)
        e["pos"] = end
        return len(chunk)

    def _vfile_write(fd: int, addr: int, count: int) -> int:
        e = vfd_table.get(fd)
        if e is None or e["mode"] not in ("w", "a"):
            return -1
        buf = vfiles.setdefault(e["name"], bytearray())
        data = bytes(mu.mem_read(addr, count))
        # Extend buf if pos > len
        while len(buf) < e["pos"]:
            buf.append(0)
        # Replace bytes at pos, extending as needed
        end = e["pos"] + len(data)
        if len(buf) < end:
            buf.extend(b"\x00" * (end - len(buf)))
        buf[e["pos"]:end] = data
        e["pos"] = end
        return count

    def _vfile_delete(name: bytes) -> int:
        if name in vfiles:
            del vfiles[name]
            return 0
        return -1

    def _write_stdout(s: bytes) -> None:
        # Translate '\r\n' to '\n' (DOS line endings → POSIX) so test diffs
        # against `.expected` files written on Unix line up.
        text = s.decode("latin1", errors="replace").replace("\r\n", "\n")
        res.stdout += text

    def _write_stderr(s: bytes) -> None:
        text = s.decode("latin1", errors="replace").replace("\r\n", "\n")
        res.stderr += text

    def _read_cstr_local(addr: int, max_len: int = 4096) -> bytes:
        out = b""
        for _ in range(max_len):
            b = bytes(mu.mem_read(addr, 1))
            if b == b"\x00":
                break
            out += b
            addr += 1
        return out

    def _printf_format(fmt: bytes, va_ptr: int) -> bytes:
        """Python-side printf formatter. Reads varargs from emulator
        memory at va_ptr (advancing as we consume each spec). Supports
        %d/%i/%u/%x/%X/%o/%c/%s/%p/%f/%g/%e/%%, the common width/precision
        flags ('0', integer width, '.N' precision), and length modifiers
        which are ignored (treated as 32-bit / double).
        """
        out = bytearray()
        i = 0
        ap = va_ptr
        n = len(fmt)

        def read32_le(addr_ref):
            bs = bytes(mu.mem_read(addr_ref[0], 4))
            addr_ref[0] += 4
            return int.from_bytes(bs, "little")

        def read64_le(addr_ref):
            bs = bytes(mu.mem_read(addr_ref[0], 8))
            addr_ref[0] += 8
            return int.from_bytes(bs, "little")

        ap_box = [ap]
        while i < n:
            c = fmt[i:i+1]
            if c != b"%":
                out += c
                i += 1
                continue
            i += 1
            if i >= n:
                break
            # flags
            zero_pad = False
            left_align = False
            hash_flag = False
            plus_flag = False
            space_flag = False
            while i < n and fmt[i:i+1] in (b"0", b"-", b"+", b" ", b"#"):
                if fmt[i:i+1] == b"0":
                    zero_pad = True
                elif fmt[i:i+1] == b"-":
                    left_align = True
                elif fmt[i:i+1] == b"#":
                    hash_flag = True
                elif fmt[i:i+1] == b"+":
                    plus_flag = True
                elif fmt[i:i+1] == b" ":
                    space_flag = True
                i += 1
            # width — either an integer or `*` to read from ap.
            width = 0
            if i < n and fmt[i:i+1] == b"*":
                width = read32_le(ap_box)
                # read32_le returns unsigned; convert to signed.
                if width >= 0x80000000:
                    width -= 0x100000000
                if width < 0:
                    # Negative width = '-' flag + abs value (C99 7.21.6.1#5).
                    left_align = True
                    width = -width
                i += 1
            else:
                while i < n and 0x30 <= fmt[i] <= 0x39:
                    width = width * 10 + (fmt[i] - 0x30)
                    i += 1
            # precision — `.N` (literal int) or `.*` (read from ap).
            precision = -1
            if i < n and fmt[i:i+1] == b".":
                i += 1
                if i < n and fmt[i:i+1] == b"*":
                    precision = read32_le(ap_box)
                    # read32_le returns unsigned; convert to signed.
                    if precision >= 0x80000000:
                        precision -= 0x100000000
                    if precision < 0:
                        precision = -1   # negative precision = no precision
                    i += 1
                else:
                    precision = 0
                    while i < n and 0x30 <= fmt[i] <= 0x39:
                        precision = precision * 10 + (fmt[i] - 0x30)
                        i += 1
            # length modifiers
            length_long_long = False
            length_short = False        # h
            length_char = False         # hh
            while i < n and fmt[i:i+1] in (b"l", b"h", b"L", b"z", b"j", b"t"):
                if fmt[i:i+1] == b"l" and i + 1 < n and fmt[i+1:i+2] == b"l":
                    length_long_long = True
                    i += 2
                    continue
                if fmt[i:i+1] == b"h" and i + 1 < n and fmt[i+1:i+2] == b"h":
                    length_char = True
                    i += 2
                    continue
                if fmt[i:i+1] == b"h":
                    length_short = True
                i += 1
            if i >= n:
                break
            conv = fmt[i:i+1]
            i += 1

            # `read32_le`/`read64_le`/`ap_box` are defined at the top
            # of `_printf_format` so the `*` width/precision branches
            # can read from ap during format-spec parsing. Each `%` spec
            # advances `ap_box[0]` as it consumes args. We capture the
            # current ap value at this conversion's start so the
            # original `ap = va_ptr` line at the top still holds (the
            # outer `ap` variable is no longer used after init).
            if conv == b"%":
                out += b"%"
                continue
            if conv == b"d" or conv == b"i":
                if length_long_long:
                    val = read64_le(ap_box)
                    if val >= 0x8000000000000000:
                        val -= 0x10000000000000000
                else:
                    val = read32_le(ap_box)
                    if length_char:
                        val &= 0xFF
                        if val >= 0x80:
                            val -= 0x100
                    elif length_short:
                        val &= 0xFFFF
                        if val >= 0x8000:
                            val -= 0x10000
                    elif val >= 0x80000000:
                        val -= 0x100000000
                if val >= 0:
                    digits = str(val).encode()
                    sign = b"+" if plus_flag else (b" " if space_flag else b"")
                else:
                    digits = str(-val).encode()
                    sign = b"-"
                # Per C99 7.21.6.1#5: "The result of converting a zero
                # value with a precision of zero is no characters."
                if val == 0 and precision == 0:
                    digits = b""
                # Precision = minimum digit count (zero-pad digits).
                # Per C99 7.21.6.1#5: when precision is given, the
                # `0` flag is ignored.
                if precision >= 0 and len(digits) < precision:
                    digits = b"0" * (precision - len(digits)) + digits
                s = sign + digits
                if precision >= 0:
                    pad = b" "
                else:
                    pad = b"0" if zero_pad else b" "
                if width > len(s):
                    if left_align:
                        s = s + b" " * (width - len(s))
                    elif pad == b"0" and sign:
                        # Zero pad goes between the sign and the digits.
                        s = sign + b"0" * (width - len(s)) + digits
                    else:
                        s = pad * (width - len(s)) + s
                out += s
            elif conv in (b"u", b"x", b"X", b"o"):
                if length_long_long:
                    val = read64_le(ap_box) & 0xFFFFFFFFFFFFFFFF
                else:
                    val = read32_le(ap_box) & 0xFFFFFFFF
                    if length_char:
                        val &= 0xFF
                    elif length_short:
                        val &= 0xFFFF
                if conv == b"u":
                    digits = str(val).encode()
                    prefix = b""
                elif conv == b"x":
                    digits = f"{val:x}".encode()
                    prefix = b"0x" if (hash_flag and val != 0) else b""
                elif conv == b"X":
                    digits = f"{val:X}".encode()
                    prefix = b"0X" if (hash_flag and val != 0) else b""
                else:  # o
                    digits = f"{val:o}".encode()
                    prefix = b"" if digits.startswith(b"0") else (b"0" if hash_flag else b"")
                # Per C99: zero value with precision 0 produces no
                # characters (except for `o` with `#` flag, which
                # still gets the leading `0`).
                if val == 0 and precision == 0:
                    digits = b""
                # Precision = minimum digit count.
                if precision >= 0 and len(digits) < precision:
                    digits = b"0" * (precision - len(digits)) + digits
                s = prefix + digits
                if precision >= 0:
                    pad = b" "
                else:
                    pad = b"0" if zero_pad else b" "
                if width > len(s):
                    if left_align:
                        s = s + b" " * (width - len(s))
                    elif pad == b"0" and prefix:
                        # Zero pad goes between prefix and digits.
                        s = prefix + b"0" * (width - len(s)) + digits
                    else:
                        s = pad * (width - len(s)) + s
                out += s
            elif conv == b"c":
                val = read32_le(ap_box)
                s = bytes([val & 0xFF])
                if width > len(s):
                    if left_align:
                        s = s + b" " * (width - len(s))
                    else:
                        s = b" " * (width - len(s)) + s
                out += s
            elif conv == b"s":
                addr = read32_le(ap_box)
                s = _read_cstr_local(addr)
                if precision >= 0 and len(s) > precision:
                    s = s[:precision]
                if width > len(s):
                    if left_align:
                        s = s + b" " * (width - len(s))
                    else:
                        s = b" " * (width - len(s)) + s
                out += s
            elif conv == b"p":
                val = read32_le(ap_box) & 0xFFFFFFFF
                out += f"0x{val:x}".encode()
            elif conv == b"n":
                # Write the count of characters output so far to the
                # int* argument. Length modifiers (`%hn`, `%hhn`,
                # `%lln`) select narrower stores.
                ptr = read32_le(ap_box)
                count = len(out)
                if length_long_long:
                    mu.mem_write(ptr, count.to_bytes(8, "little"))
                elif length_short:
                    mu.mem_write(ptr, (count & 0xFFFF).to_bytes(2, "little"))
                elif length_char:
                    mu.mem_write(ptr, bytes([count & 0xFF]))
                else:
                    mu.mem_write(ptr, (count & 0xFFFFFFFF).to_bytes(4, "little"))
            elif conv in (b"f", b"g", b"e", b"E", b"G"):
                bs = bytes(mu.mem_read(ap_box[0], 8))
                ap_box[0] += 8
                import struct as _st
                val = _st.unpack("<d", bs)[0]
                if precision < 0:
                    precision = 6
                # Apply +/space sign flags via Python format ourselves
                # since `%` doesn't have a leading-space flag.
                fmt_py = f"%.{precision}{conv.decode()}"
                s = (fmt_py % val).encode()
                # Determine sign prefix (already in `s` for negative).
                # For positive values with `+` or space flag, prepend.
                if val >= 0 and not (s and s[:1] in b"+-"):
                    if plus_flag:
                        s = b"+" + s
                    elif space_flag:
                        s = b" " + s
                if width > len(s):
                    if left_align:
                        s = s + b" " * (width - len(s))
                    elif zero_pad:
                        # Zero pad goes between sign and digits.
                        if s and s[:1] in b"+- ":
                            s = s[:1] + b"0" * (width - len(s)) + s[1:]
                        else:
                            s = b"0" * (width - len(s)) + s
                    else:
                        s = b" " * (width - len(s)) + s
                out += s
            else:
                # Unknown — output as-is.
                out += b"%" + conv
        return bytes(out)

    def on_int(uc, intno, user_data):
        eax = uc.reg_read(UC_X86_REG_EAX)
        ah = (eax >> 8) & 0xFF
        al = eax & 0xFF
        if intno == 0x80:
            # Private uc386 trap: 64-bit divide / modulo.
            #   EDX:EAX = numer (high:low)
            #   EBX:ECX = denom (high:low)
            #   ESI low byte = op (0=udiv, 1=sdiv, 2=umod, 3=smod)
            # Result in EDX:EAX.
            ecx = uc.reg_read(UC_X86_REG_ECX)
            ebx = uc.reg_read(UC_X86_REG_EBX)
            edx = uc.reg_read(UC_X86_REG_EDX)
            esi = uc.reg_read(UC_X86_REG_ESI)
            op = esi & 0xFF
            num = ((edx & 0xFFFFFFFF) << 32) | (eax & 0xFFFFFFFF)
            den = ((ebx & 0xFFFFFFFF) << 32) | (ecx & 0xFFFFFFFF)
            if op in (1, 3):
                if num >= 0x8000000000000000:
                    num -= 0x10000000000000000
                if den >= 0x8000000000000000:
                    den -= 0x10000000000000000
            if den == 0:
                res.error = "long-long divide by zero"
                uc.emu_stop()
                return
            if op in (0, 1):
                # Truncated division (matching C99 behavior).
                if (num < 0) != (den < 0) and num % den != 0:
                    quot = num // den + 1
                else:
                    quot = num // den
                result = quot
            else:
                if (num < 0) != (den < 0) and num % den != 0:
                    rem = num - (num // den + 1) * den
                else:
                    rem = num - (num // den) * den
                result = rem
            result_64 = result & 0xFFFFFFFFFFFFFFFF
            uc.reg_write(UC_X86_REG_EAX, result_64 & 0xFFFFFFFF)
            uc.reg_write(UC_X86_REG_EDX, (result_64 >> 32) & 0xFFFFFFFF)
            return
        if intno == 0x83:
            # uc386 virtual ethernet. AH=0/1/2 = init/send/recv. With
            # no NetworkSimulator wired, return AL=1 so ethdrv_init
            # fails cleanly and the program treats the NIC as absent.
            if net is None:
                uc.reg_write(UC_X86_REG_EAX, (eax & ~0xFF) | 1)
                return
            if ah == 0x00:
                # AH=0 init: write 6 MAC bytes to [EDI].
                edi = uc.reg_read(UC_X86_REG_EDI)
                uc.mem_write(edi, net.init_mac())
                uc.reg_write(UC_X86_REG_EAX, eax & ~0xFF)
                return
            if ah == 0x01:
                # AH=1 send: ESI=buf, ECX=len. Read from guest memory,
                # hand to the simulator, return its rc in AL.
                esi = uc.reg_read(UC_X86_REG_ESI)
                ecx = uc.reg_read(UC_X86_REG_ECX)
                length = ecx & 0xFFFF
                frame = bytes(uc.mem_read(esi, length)) if length else b""
                rc = net.send_frame(frame)
                uc.reg_write(UC_X86_REG_EAX,
                             (eax & ~0xFF) | (rc & 0xFF))
                return
            if ah == 0x02:
                # AH=2 recv: EDI=buf, ECX=maxlen. Pop next sim frame
                # (if any), copy to buf, return actual length in ECX.
                edi = uc.reg_read(UC_X86_REG_EDI)
                ecx = uc.reg_read(UC_X86_REG_ECX)
                maxlen = ecx & 0xFFFF
                frame = net.recv_frame(maxlen)
                if frame is None:
                    uc.reg_write(UC_X86_REG_ECX, ecx & ~0xFFFF)
                    return
                uc.mem_write(edi, frame)
                uc.reg_write(UC_X86_REG_ECX,
                             (ecx & ~0xFFFF) | (len(frame) & 0xFFFF))
                return
            # Unknown AH — flag error in AL, leave ECX/etc unchanged.
            uc.reg_write(UC_X86_REG_EAX, (eax & ~0xFF) | 1)
            return
        if intno == 0x31:
            # DPMI services. Two fns implemented:
            #   0x0200 — Get Real Mode Interrupt Vector. Used by
            #            pktdrv_detect() to find Crynwr in the IVT.
            #   0x0303 — Allocate Real Mode Callback. Used by
            #            pktdrv_init() to register a DPMI trampoline
            #            so real-mode Crynwr drivers can call back
            #            into our 32-bit receiver. Under emulation
            #            we record the registration but never invoke
            #            it (AH=0x99 polling drives RX instead);
            #            real DOS hardware does the real call.
            # Always handled (even without a NetworkSimulator) so
            # the binary's probes don't trip the unhandled-INT trap.
            ax = eax & 0xFFFF
            flags = uc.reg_read(UC_X86_REG_EFLAGS)
            if ax == 0x0200 and pktdrv_int is not None:
                from .dos_emu_netsim import PKTDRV_HANDLER_LINEAR
                ebx = uc.reg_read(UC_X86_REG_EBX)
                bl = ebx & 0xFF
                if bl == pktdrv_int:
                    seg = (PKTDRV_HANDLER_LINEAR >> 4) & 0xFFFF
                    off = PKTDRV_HANDLER_LINEAR & 0xF
                    ecx_new = (uc.reg_read(UC_X86_REG_ECX) & 0xFFFF0000) | seg
                    edx_new = (uc.reg_read(UC_X86_REG_EDX) & 0xFFFF0000) | off
                    uc.reg_write(UC_X86_REG_ECX, ecx_new)
                    uc.reg_write(UC_X86_REG_EDX, edx_new)
                    uc.reg_write(UC_X86_REG_EFLAGS, flags & ~0x1)
                    return
            if ax == 0x0303 and net is not None:
                handler_linear = uc.reg_read(UC_X86_REG_ESI)
                rmcs_linear    = uc.reg_read(UC_X86_REG_EDI)
                fake_seg = net._next_dpmi_real_seg
                net._next_dpmi_real_seg = (fake_seg + 0x10) & 0xFFFF
                fake_off = 0
                net.dpmi_callbacks.append(
                    (handler_linear, rmcs_linear, fake_seg, fake_off))
                ecx_new = (uc.reg_read(UC_X86_REG_ECX) & 0xFFFF0000) | fake_seg
                edx_new = (uc.reg_read(UC_X86_REG_EDX) & 0xFFFF0000) | fake_off
                uc.reg_write(UC_X86_REG_ECX, ecx_new)
                uc.reg_write(UC_X86_REG_EDX, edx_new)
                uc.reg_write(UC_X86_REG_EFLAGS, flags & ~0x1)
                return
            # Unimplemented fn or unmatched int_num → CF=1, AX=0.
            # That's how DPMI signals "not handled" to the caller.
            uc.reg_write(UC_X86_REG_EFLAGS, flags | 0x1)
            return
        if intno == pktdrv_int and net is not None and pktdrv_int is not None:
            # Crynwr packet driver — see the Phase-3 plan in
            # addons/gnu/micropython/uc386-dos/pktdrv_uc386dos.c.
            # Only the calls used by `pktdrv_init` + `pktdrv_send`
            # are implemented. Receive callback support (AH=0x02
            # invocation by sim of the registered handler) is not
            # yet wired — we keep INT 0x83 as the RX path under
            # dos_emu and document a DPMI INT 31h fn 0x0303 thunk
            # as the real-DOS follow-up.
            flags = uc.reg_read(UC_X86_REG_EFLAGS)
            if ah == 0x01:
                # driver_info — return any valid handle's metadata.
                # AL=basic class, BX=version, CH=class, DS:SI=name,
                # CX=type, DL=number. Caller (pktdrv probe) only
                # checks CF.
                uc.reg_write(UC_X86_REG_EFLAGS, flags & ~0x1)
                return
            if ah == 0x02:
                # access_type — register receiver. EAX low byte set
                # to 0, AX (16-bit) set to handle. Save EDI as the
                # receiver address for the future RX bridge.
                edi = uc.reg_read(UC_X86_REG_EDI)
                net.pktdrv_receiver_addr = edi
                handle = net.pktdrv_handle
                uc.reg_write(UC_X86_REG_EAX,
                             (eax & ~0xFFFF) | (handle & 0xFFFF))
                uc.reg_write(UC_X86_REG_EFLAGS, flags & ~0x1)
                return
            if ah == 0x04:
                # send_pkt — DS:SI = buf, CX = len.
                esi = uc.reg_read(UC_X86_REG_ESI)
                ecx = uc.reg_read(UC_X86_REG_ECX)
                length = ecx & 0xFFFF
                frame = bytes(uc.mem_read(esi, length)) if length else b""
                rc = net.send_frame(frame)
                if rc != 0:
                    uc.reg_write(UC_X86_REG_EFLAGS, flags | 0x1)
                else:
                    # Drain inbound frames into the binary's RX slot
                    # if it registered the polling triple via
                    # AH=0x99. Single-slot — the binary's pump_rx
                    # clears `pending` between deliveries, so we
                    # only post when the slot is free.
                    if net.pktdrv_rx_buf_addr:
                        while net.rx_queue:
                            pending = struct.unpack(
                                "<I",
                                uc.mem_read(net.pktdrv_rx_pending_addr, 4),
                            )[0]
                            if pending:
                                break
                            f = net.rx_queue.pop(0)
                            uc.mem_write(net.pktdrv_rx_buf_addr, f)
                            uc.mem_write(net.pktdrv_rx_len_addr,
                                         struct.pack("<I", len(f)))
                            uc.mem_write(net.pktdrv_rx_pending_addr,
                                         struct.pack("<I", 1))
                    uc.reg_write(UC_X86_REG_EFLAGS, flags & ~0x1)
                return
            if ah == 0x06:
                # get_address — ES:DI = buf, CX = max len. CX
                # returned = actual.
                edi = uc.reg_read(UC_X86_REG_EDI)
                mac = net.init_mac()
                uc.mem_write(edi, mac)
                ecx = uc.reg_read(UC_X86_REG_ECX)
                uc.reg_write(UC_X86_REG_ECX,
                             (ecx & ~0xFFFF) | len(mac))
                uc.reg_write(UC_X86_REG_EFLAGS, flags & ~0x1)
                return
            if ah == 0x99:
                # uc386dos polling-mode RX registration. The binary
                # hands us pointers to its RX buffer + pending flag
                # + length slot; we post frames there directly,
                # bypassing the real-mode receiver-callback dance
                # that real DOS Crynwr drivers require. Real DOS
                # would never see this AH — pktdrv_recv on hardware
                # waits for the DPMI-thunk receiver to land.
                net.pktdrv_rx_buf_addr     = uc.reg_read(UC_X86_REG_EDI)
                net.pktdrv_rx_pending_addr = uc.reg_read(UC_X86_REG_ESI)
                net.pktdrv_rx_len_addr     = uc.reg_read(UC_X86_REG_ECX)
                uc.reg_write(UC_X86_REG_EFLAGS, flags & ~0x1)
                return
            # Unsupported subfunction — CF=1, AL=undefined.
            uc.reg_write(UC_X86_REG_EFLAGS, flags | 0x1)
            return
        if intno == 0:
            # x86 #DE (divide error). Map to SIGFPE (signum=2 per our
            # libc's signal.h) and dispatch the registered handler if
            # any. Without a handler, abort.
            handler = signal_handlers.get(2)
            if handler:
                # Call handler(signum=2). Push fake retaddr and arg.
                esp = uc.reg_read(UC_X86_REG_ESP)
                esp -= 4
                uc.mem_write(esp, struct.pack("<I", 2))  # signum
                esp -= 4
                uc.mem_write(esp, struct.pack("<I", 0xFFFFFFFF))  # ret
                uc.reg_write(UC_X86_REG_ESP, esp)
                uc.reg_write(UC_X86_REG_EIP, handler)
                return
            res.error = "SIGFPE: no handler"
            uc.emu_stop()
            return
        if intno == 0x1A and ah == 0x00:
            # BIOS time-of-day read. Real BIOS returns CX:DX = system
            # tick count (~18.2 Hz, ~55 ms/tick). Under emulation
            # there's no clock tied to wall time, so we synthesize a
            # monotonically-increasing counter that advances every
            # time the program reads it. That gives user code a
            # `ticks_ms` source whose `ticks_diff` is always positive
            # and whose absolute progression matches the number of
            # times user code polled the BIOS counter — which is
            # exactly what `time.sleep_ms()`'s busy-wait depends on.
            tick = res._bios_tick
            res._bios_tick = (tick + 1) & 0xFFFFFFFF
            ecx = uc.reg_read(UC_X86_REG_ECX)
            edx = uc.reg_read(UC_X86_REG_EDX)
            uc.reg_write(UC_X86_REG_ECX,
                         (ecx & 0xFFFF0000) | ((tick >> 16) & 0xFFFF))
            uc.reg_write(UC_X86_REG_EDX,
                         (edx & 0xFFFF0000) | (tick & 0xFFFF))
            # AL = 0 (no midnight rollover this call).
            uc.reg_write(UC_X86_REG_EAX, eax & ~0xFF)
            return
        if intno != 0x21:
            res.error = f"unexpected interrupt {intno:#x}"
            uc.emu_stop()
            return
        if ah == 0x5C:
            # sprintf via harness: EBX=buf, ECX=fmt, EDX=va_ptr
            ebx = uc.reg_read(UC_X86_REG_EBX)
            ecx = uc.reg_read(UC_X86_REG_ECX)
            edx = uc.reg_read(UC_X86_REG_EDX)
            fmt = _read_cstr_local(ecx)
            formatted = _printf_format(fmt, edx)
            uc.mem_write(ebx, formatted + b"\x00")
            new_eax = (eax & ~0xFFFFFFFF) | len(formatted)
            uc.reg_write(UC_X86_REG_EAX, new_eax)
            return
        if ah == 0x5E:
            # printf via harness: ECX=fmt, EDX=va_ptr → format, write to stdout.
            ecx = uc.reg_read(UC_X86_REG_ECX)
            edx = uc.reg_read(UC_X86_REG_EDX)
            fmt = _read_cstr_local(ecx)
            formatted = _printf_format(fmt, edx)
            _write_stdout(formatted)
            new_eax = (eax & ~0xFFFFFFFF) | (len(formatted) & 0xFFFFFFFF)
            uc.reg_write(UC_X86_REG_EAX, new_eax)
            return
        if ah == 0x5F:
            # fprintf via harness: EBX=stream(fd 1=stdout, 2=stderr,
            # 3+ = virtual file), ECX=fmt, EDX=va_ptr.
            ebx = uc.reg_read(UC_X86_REG_EBX)
            ecx = uc.reg_read(UC_X86_REG_ECX)
            edx = uc.reg_read(UC_X86_REG_EDX)
            fmt = _read_cstr_local(ecx)
            formatted = _printf_format(fmt, edx)
            fd = ebx & 0xFFFFFFFF
            # Translate stdin/stdout/stderr magic FILE* sentinels.
            if fd == 0xF0: fd = 0
            elif fd == 0xF1: fd = 1
            elif fd == 0xF2: fd = 2
            if fd == 2:
                _write_stderr(formatted)
            elif fd == 1:
                _write_stdout(formatted)
            elif fd >= 3 and fd in vfd_table:
                # Append formatted bytes to the virtual file. We can't
                # use _vfile_write directly because the bytes are a
                # Python bytes, not in emulator memory.
                e = vfd_table[fd]
                if e["mode"] in ("w", "a"):
                    buf = vfiles.setdefault(e["name"], bytearray())
                    while len(buf) < e["pos"]:
                        buf.append(0)
                    end = e["pos"] + len(formatted)
                    if len(buf) < end:
                        buf.extend(b"\x00" * (end - len(buf)))
                    buf[e["pos"]:end] = formatted
                    e["pos"] = end
            new_eax = (eax & ~0xFFFFFFFF) | (len(formatted) & 0xFFFFFFFF)
            uc.reg_write(UC_X86_REG_EAX, new_eax)
            return
        if ah == 0x5D:
            # snprintf: EBX=buf, ESI=size, ECX=fmt, EDX=va_ptr
            ebx = uc.reg_read(UC_X86_REG_EBX)
            esi = uc.reg_read(UC_X86_REG_ESI)
            ecx = uc.reg_read(UC_X86_REG_ECX)
            edx = uc.reg_read(UC_X86_REG_EDX)
            fmt = _read_cstr_local(ecx)
            formatted = _printf_format(fmt, edx)
            if esi > 0:
                truncated = formatted[: max(esi - 1, 0)]
                uc.mem_write(ebx, truncated + b"\x00")
            new_eax = (eax & ~0xFFFFFFFF) | len(formatted)
            uc.reg_write(UC_X86_REG_EAX, new_eax)
            return
        if ah == 0x4C or ah == 0x00:
            res.exit_code = al
            uc.emu_stop()
            return
        if ah == 0x02:
            # putchar in DL on real DOS, but we accept AL too for codegen
            # convenience. Look at DL primarily.
            edx = uc.reg_read(UC_X86_REG_EDX)
            ch = bytes([edx & 0xFF])
            _write_stdout(ch)
            return
        if ah == 0x06:
            # Direct console I/O: DL=char (or 0xFF for input). Output only.
            edx = uc.reg_read(UC_X86_REG_EDX)
            dl = edx & 0xFF
            if dl != 0xFF:
                _write_stdout(bytes([dl]))
            return
        if ah == 0x2A:
            # AH=0x2A — get system date.
            #   Returns: CX = year (1980-2099), DH = month, DL = day,
            #            AL = day of week (0=Sun, 6=Sat).
            # Synthesize a fixed deterministic date for tests
            # (2026-05-03, the day this slice was written). Real DOS
            # under PMODE/W in the .exe pipeline gets the real RTC.
            ecx = uc.reg_read(UC_X86_REG_ECX)
            edx = uc.reg_read(UC_X86_REG_EDX)
            uc.reg_write(UC_X86_REG_ECX,
                         (ecx & 0xFFFF0000) | 2026)
            uc.reg_write(UC_X86_REG_EDX,
                         (edx & 0xFFFF0000) | (5 << 8) | 3)
            uc.reg_write(UC_X86_REG_EAX, (eax & ~0xFF) | 0)
            return
        if ah == 0x2C:
            # AH=0x2C — get system time-of-day.
            #   Returns: CH = hour, CL = minute,
            #            DH = second, DL = hundredths.
            # Synthesize 12:34:00. Adequate for proving the date+time
            # → epoch-seconds path lights up; the smoke test for
            # `time.time()` checks that the result lands in the
            # right ballpark, not an exact value.
            ecx = uc.reg_read(UC_X86_REG_ECX)
            edx = uc.reg_read(UC_X86_REG_EDX)
            uc.reg_write(UC_X86_REG_ECX,
                         (ecx & 0xFFFF0000) | (12 << 8) | 34)
            uc.reg_write(UC_X86_REG_EDX,
                         (edx & 0xFFFF0000) | (0 << 8) | 0)
            return
        if ah == 0x09:
            edx = uc.reg_read(UC_X86_REG_EDX)
            s = _read_cstr(uc, edx, term=b"$")
            _write_stdout(s)
            return
        if ah == 0x40:
            # write(fd, buf, count): BX=fd, CX=count, DS:EDX=buf
            ebx = uc.reg_read(UC_X86_REG_EBX)
            ecx = uc.reg_read(UC_X86_REG_ECX)
            edx = uc.reg_read(UC_X86_REG_EDX)
            count = ecx & 0xFFFF  # spec is 16-bit count, but tolerate larger
            fd = ebx & 0xFFFF
            # Translate stdin/stdout/stderr magic FILE* sentinels (libc
            # uses 0xF0 / 0xF1 / 0xF2 so they don't compare-equal to NULL).
            if fd == 0xF0: fd = 0
            elif fd == 0xF1: fd = 1
            elif fd == 0xF2: fd = 2
            if fd == 1:
                data = bytes(uc.mem_read(edx, count))
                _write_stdout(data)
                actual = count
            elif fd == 2:
                data = bytes(uc.mem_read(edx, count))
                _write_stderr(data)
                actual = count
            elif fd >= 3:
                actual = _vfile_write(fd, edx, count)
                if actual < 0:
                    actual = 0
            else:
                actual = 0
            # Return bytes-written in AX; CF clear on success.
            new_eax = (eax & ~0xFFFF) | (actual & 0xFFFF)
            uc.reg_write(UC_X86_REG_EAX, new_eax)
            return
        if ah == 0x3F:
            # read(fd, buf, count): BX=fd, CX=count, DS:EDX=buf.
            ebx = uc.reg_read(UC_X86_REG_EBX)
            ecx = uc.reg_read(UC_X86_REG_ECX)
            edx = uc.reg_read(UC_X86_REG_EDX)
            count = ecx & 0xFFFF
            fd = ebx & 0xFFFF
            # Translate stdin/stdout/stderr magic FILE* sentinels.
            if fd == 0xF0: fd = 0
            elif fd == 0xF1: fd = 1
            elif fd == 0xF2: fd = 2
            if fd == 0:
                start = stdin_pos[0]
                end = min(start + count, len(stdin_bytes))
                chunk = stdin_bytes[start:end]
                if chunk:
                    uc.mem_write(edx, chunk)
                stdin_pos[0] = end
                actual = len(chunk)
            elif fd >= 3:
                actual = _vfile_read(fd, edx, count)
                if actual < 0:
                    actual = 0
            else:
                actual = 0
            new_eax = (eax & ~0xFFFF) | (actual & 0xFFFF)
            uc.reg_write(UC_X86_REG_EAX, new_eax)
            return
        if ah == 0x1A:
            # set DTA address: DS:EDX = new DTA pointer.
            edx = uc.reg_read(UC_X86_REG_EDX)
            vdta_addr[0] = edx & 0xFFFFFFFF
            return
        if ah == 0x4E:
            # find-first(mask, attr): DS:EDX=mask, CX=attribute.
            # On success: DTA filled with first match (offset +21
            # attr, +30 8.3 filename + NUL); CF clear.
            # On no match: CF set.
            edx = uc.reg_read(UC_X86_REG_EDX)
            mask = bytes(_read_cstr_local(edx)).decode("ascii",
                                                       errors="replace")
            # Build candidate list from vfiles + vdirs. Match
            # the mask trivially — support "*", "*.*", "*.ext",
            # and exact names.
            def _match(name: str) -> bool:
                if mask in ("*", "*.*"):
                    return True
                if mask.startswith("*.") and "." in mask:
                    ext = mask[2:].lower()
                    return name.lower().endswith("." + ext)
                # Strip trailing "\*.*" — caller wants dir contents.
                if mask.endswith("\\*.*") or mask.endswith("/*.*"):
                    return True
                return name.lower() == mask.lower()
            entries = []
            for fname, _content in vfiles.items():
                n = fname.decode("ascii", errors="replace")
                if _match(n):
                    entries.append((fname, False))
            for dname in vdirs:
                n = dname.decode("ascii", errors="replace")
                if _match(n):
                    entries.append((dname, True))
            eflags = uc.reg_read(UC_X86_REG_EFLAGS)
            if not entries:
                uc.reg_write(UC_X86_REG_EFLAGS, eflags | 1)
                return
            # Pop the first; queue the rest for find-next.
            first, is_dir = entries[0]
            vfind_state[:] = entries[1:]
            # Write into DTA: attr at +21, name at +30 (max 13 bytes).
            if vdta_addr[0]:
                uc.mem_write(vdta_addr[0] + 21,
                             bytes([0x10 if is_dir else 0x00]))
                truncated = first[:12] + b"\x00"
                uc.mem_write(vdta_addr[0] + 30, truncated)
            uc.reg_write(UC_X86_REG_EFLAGS, eflags & ~1)
            return
        if ah == 0x4F:
            # find-next: continue iteration. CF set on no more.
            eflags = uc.reg_read(UC_X86_REG_EFLAGS)
            if not vfind_state:
                uc.reg_write(UC_X86_REG_EFLAGS, eflags | 1)
                return
            next_name, is_dir = vfind_state.pop(0)
            if vdta_addr[0]:
                uc.mem_write(vdta_addr[0] + 21,
                             bytes([0x10 if is_dir else 0x00]))
                truncated = next_name[:12] + b"\x00"
                uc.mem_write(vdta_addr[0] + 30, truncated)
            uc.reg_write(UC_X86_REG_EFLAGS, eflags & ~1)
            return
        if ah == 0x39:
            # mkdir(path): DS:EDX=path. Set CF on error.
            edx = uc.reg_read(UC_X86_REG_EDX)
            name = bytes(_read_cstr_local(edx))
            eflags = uc.reg_read(UC_X86_REG_EFLAGS)
            if name in vdirs or name in vfiles:
                # Already exists — error.
                uc.reg_write(UC_X86_REG_EFLAGS, eflags | 1)
            else:
                vdirs.add(name)
                uc.reg_write(UC_X86_REG_EFLAGS, eflags & ~1)
            return
        if ah == 0x3A:
            # rmdir(path): DS:EDX=path. Set CF on error.
            edx = uc.reg_read(UC_X86_REG_EDX)
            name = bytes(_read_cstr_local(edx))
            eflags = uc.reg_read(UC_X86_REG_EFLAGS)
            if name in vdirs:
                vdirs.discard(name)
                uc.reg_write(UC_X86_REG_EFLAGS, eflags & ~1)
            else:
                uc.reg_write(UC_X86_REG_EFLAGS, eflags | 1)
            return
        if ah == 0x3B:
            # chdir(path): DS:EDX=path. Set CF on error.
            edx = uc.reg_read(UC_X86_REG_EDX)
            name = bytes(_read_cstr_local(edx))
            eflags = uc.reg_read(UC_X86_REG_EFLAGS)
            if name in vdirs or name == b"." or name == b"..":
                # Trivial cwd update — store the requested path
                # verbatim with a "C:\" prefix if it isn't drive-
                # qualified yet.
                if not (len(name) >= 2 and name[1:2] == b":"):
                    vcwd[0] = b"C:\\" + name
                else:
                    vcwd[0] = name
                uc.reg_write(UC_X86_REG_EFLAGS, eflags & ~1)
            else:
                uc.reg_write(UC_X86_REG_EFLAGS, eflags | 1)
            return
        if ah == 0x3C:
            # creat(name, mode): create or truncate. DS:EDX=name. Returns fd.
            edx = uc.reg_read(UC_X86_REG_EDX)
            name = bytes(_read_cstr_local(edx))
            fd = _vfile_open(name, "w")
            new_eax = (eax & ~0xFFFFFFFF) | (fd & 0xFFFFFFFF)
            uc.reg_write(UC_X86_REG_EAX, new_eax)
            return
        if ah == 0x3D:
            # open(name, mode): DS:EDX=name, AL=mode (0=r, 1=w, 2=rw).
            # We support 0 (read) and 1/2 (write — truncate).
            edx = uc.reg_read(UC_X86_REG_EDX)
            name = bytes(_read_cstr_local(edx))
            mode_byte = al
            if mode_byte == 0:
                fd = _vfile_open(name, "r")
            elif mode_byte in (1, 2):
                fd = _vfile_open(name, "w")
            else:
                fd = -1
            new_eax = (eax & ~0xFFFFFFFF) | (fd & 0xFFFFFFFF)
            uc.reg_write(UC_X86_REG_EAX, new_eax)
            return
        if ah == 0x3E:
            # close(fd): BX=fd. Returns 0 on success.
            ebx = uc.reg_read(UC_X86_REG_EBX)
            fd = ebx & 0xFFFFFFFF
            # Standard streams: closing them is a no-op (success).
            if fd in (0xF0, 0xF1, 0xF2, 0, 1, 2):
                new_eax = (eax & ~0xFFFFFFFF) | 0
                uc.reg_write(UC_X86_REG_EAX, new_eax)
                return
            rc = _vfile_close(fd)
            new_eax = (eax & ~0xFFFFFFFF) | (rc & 0xFFFFFFFF)
            uc.reg_write(UC_X86_REG_EAX, new_eax)
            return
        if ah == 0x41:
            # unlink(name): DS:EDX=name. Returns 0 on success, -1 on error.
            edx = uc.reg_read(UC_X86_REG_EDX)
            name = bytes(_read_cstr_local(edx))
            rc = _vfile_delete(name)
            new_eax = (eax & ~0xFFFFFFFF) | (rc & 0xFFFFFFFF)
            uc.reg_write(UC_X86_REG_EAX, new_eax)
            return
        if ah == 0x42:
            # lseek(fd, off, whence): BX=fd, CX:DX=offset, AL=whence.
            # Real DOS returns DX:AX=new pos / CF on error. The libc
            # asm packs DX:AX -> EAX so we only need to put the new
            # position in EAX (split into AX-low + DX-high). On error
            # we leave CF set and AX=-1.
            ebx = uc.reg_read(UC_X86_REG_EBX)
            ecx = uc.reg_read(UC_X86_REG_ECX)
            edx = uc.reg_read(UC_X86_REG_EDX)
            fd = ebx & 0xFFFF
            off = ((ecx & 0xFFFF) << 16) | (edx & 0xFFFF)
            # Sign-extend 32-bit offset.
            if off & 0x80000000:
                off -= 0x100000000
            whence = al
            new_pos = _vfile_seek(fd, off, whence)
            if new_pos < 0:
                # Error: set CF in EFLAGS, AX=-1.
                eflags = uc.reg_read(UC_X86_REG_EFLAGS)
                uc.reg_write(UC_X86_REG_EFLAGS, eflags | 1)
                new_eax = (eax & ~0xFFFF) | 0xFFFF
                uc.reg_write(UC_X86_REG_EAX, new_eax)
                return
            # Clear CF, return DX:AX = new_pos.
            eflags = uc.reg_read(UC_X86_REG_EFLAGS)
            uc.reg_write(UC_X86_REG_EFLAGS, eflags & ~1)
            new_eax = (eax & ~0xFFFF) | (new_pos & 0xFFFF)
            uc.reg_write(UC_X86_REG_EAX, new_eax)
            new_edx = (edx & ~0xFFFF) | ((new_pos >> 16) & 0xFFFF)
            uc.reg_write(UC_X86_REG_EDX, new_edx)
            return
        if ah == 0x47:
            # getcwd(drive, buf): DL=drive (0=current, 1=A, 2=B, ...),
            # DS:ESI=buf. DOS writes the path WITHOUT the leading
            # "X:\". Returns 0 on success, CF on error. Our libc's
            # `getcwd()` wraps this with the drive prefix.
            esi = uc.reg_read(UC_X86_REG_ESI)
            eflags = uc.reg_read(UC_X86_REG_EFLAGS)
            # vcwd[0] is "C:\<rest>" — strip the "C:\" prefix to
            # mirror DOS's no-prefix output.
            cwd = vcwd[0]
            if len(cwd) >= 3 and cwd[1:2] == b":" and cwd[2:3] == b"\\":
                rest = cwd[3:]
            else:
                rest = cwd
            uc.mem_write(esi, rest + b"\x00")
            uc.reg_write(UC_X86_REG_EFLAGS, eflags & ~1)
            return
        if ah == 0x19:
            # Get current drive: AL = 0 (A:), 1 (B:), 2 (C:), ...
            new_eax = (eax & ~0xFF) | 2  # always C:
            uc.reg_write(UC_X86_REG_EAX, new_eax)
            return
        if ah == 0x56:
            # rename(old, new): DS:EDX=old, ES:EDI=new. Set CF on error.
            edx = uc.reg_read(UC_X86_REG_EDX)
            edi = uc.reg_read(UC_X86_REG_EDI)
            old_name = bytes(_read_cstr_local(edx))
            new_name = bytes(_read_cstr_local(edi))
            eflags = uc.reg_read(UC_X86_REG_EFLAGS)
            if old_name in vfiles:
                vfiles[new_name] = vfiles.pop(old_name)
                uc.reg_write(UC_X86_REG_EFLAGS, eflags & ~1)
            elif old_name in vdirs:
                vdirs.discard(old_name)
                vdirs.add(new_name)
                uc.reg_write(UC_X86_REG_EFLAGS, eflags & ~1)
            else:
                uc.reg_write(UC_X86_REG_EFLAGS, eflags | 1)
            return
        if ah == 0x5A:
            # tmpnam: DS:EDX=buf (output). Writes a unique name. Returns
            # buf in AX.
            edx = uc.reg_read(UC_X86_REG_EDX)
            seq = tmpnam_seq[0]
            tmpnam_seq[0] += 1
            name = f"/tmp/uctmp_{seq:04d}".encode("ascii")
            uc.mem_write(edx, name + b"\x00")
            new_eax = (eax & ~0xFFFFFFFF) | (edx & 0xFFFFFFFF)
            uc.reg_write(UC_X86_REG_EAX, new_eax)
            return
        if ah == 0x62:
            # Get current PSP segment. Returns BX = PSP_seg.
            ebx = uc.reg_read(UC_X86_REG_EBX)
            new_ebx = (ebx & ~0xFFFF) | (PSP_SEG & 0xFFFF)
            uc.reg_write(UC_X86_REG_EBX, new_ebx)
            return
        if ah == 0x4B and al == 0x00:
            # Load and Execute: DS:DX = path, ES:BX = parameter block.
            # We can't actually fork/exec under unicorn — instead we
            # parse the command tail (param_block + 2 → far ptr to
            # length-prefixed string) and recognize a small set of
            # built-in DOS commands so libc system() smoke tests can
            # pin behavior.
            edx = uc.reg_read(UC_X86_REG_EDX)
            ebx = uc.reg_read(UC_X86_REG_EBX)
            path = bytes(_read_cstr_local(edx))
            # Read far pointer at param_block + 2: low 16 = offset,
            # high 16 = segment. Linear = (seg << 4) | offset.
            tail_far = struct.unpack("<HH", uc.mem_read(ebx + 2, 4))
            tail_off, tail_seg = tail_far
            tail_lin = (tail_seg << 4) | tail_off
            # Pascal-string: 1 length byte then chars.
            tail_len = uc.mem_read(tail_lin, 1)[0]
            tail = bytes(uc.mem_read(tail_lin + 1, tail_len))
            # Strip trailing CR (DOS terminator) if present.
            if tail.endswith(b"\r"):
                tail = tail[:-1]
            # Recognize "/C <cmd>" or "/c <cmd>" (libc's system()
            # prepends " /C " so we lstrip whitespace before matching).
            cmd = tail.lstrip()
            if cmd[:3].upper() == b"/C ":
                cmd = cmd[3:]
            cmd = cmd.strip()
            # Tiny built-in interpreter: enough to make smoke tests
            # observable. ECHO writes its argv to stdout; anything
            # else exits 0 silently. Real DOS would actually shell
            # out; our exit code stash matches that contract.
            child_exit_code[0] = 0
            if cmd[:5].upper() == b"ECHO ":
                _write_stdout(cmd[5:] + b"\r\n")
            elif cmd.upper() == b"ECHO":
                _write_stdout(b"\r\n")
            elif cmd[:5].upper() == b"EXIT ":
                try:
                    child_exit_code[0] = int(cmd[5:].strip()) & 0xFF
                except ValueError:
                    child_exit_code[0] = 0
            # No carry → success. AX is otherwise unspecified; clear
            # to 0 to avoid leaking stale state.
            uc.reg_write(UC_X86_REG_EAX, 0)
            uc.reg_write(UC_X86_REG_EFLAGS,
                         uc.reg_read(UC_X86_REG_EFLAGS) & ~1)
            return
        if ah == 0x4D:
            # Get child return code: AL = stored exit code, AH = exit
            # type (00 = normal). We track the last exec's exit code
            # in `child_exit_code[0]`; AH=0x4B sets it, AH=0x4D reads
            # and returns it.
            code = child_exit_code[0]
            new_eax = (eax & ~0xFFFF) | (code & 0xFF)
            uc.reg_write(UC_X86_REG_EAX, new_eax)
            return
        if ah == 0x99:
            # signal(signum, handler): AL=signum, EBX=handler.
            # Returns prev handler in EAX.
            ebx = uc.reg_read(UC_X86_REG_EBX)
            signum = al
            prev = signal_handlers.get(signum, 0)
            signal_handlers[signum] = ebx
            new_eax = (eax & ~0xFFFFFFFF) | (prev & 0xFFFFFFFF)
            uc.reg_write(UC_X86_REG_EAX, new_eax)
            return
        if ah == 0xA0:
            # uc386 extension: POSIX open(path, flags). DS:EDX=name,
            # ECX=POSIX flags. Returns fd in AX, -1 on error.
            #
            # The DOS-flavored AH=0x3D handler only takes a 0/1/2 mode
            # byte (read/write/rdwr) — too restrictive for sbase-style
            # `open(name, O_WRONLY|O_CREAT|O_APPEND, 0666)`. This handler
            # consumes the full POSIX flag word and dispatches to the
            # right vfile mode.
            edx = uc.reg_read(UC_X86_REG_EDX)
            ecx = uc.reg_read(UC_X86_REG_ECX)
            name = bytes(_read_cstr_local(edx))
            flags = ecx & 0xFFFFFFFF
            O_WRONLY = 1
            O_RDWR = 2
            O_CREAT = 0o100
            O_TRUNC = 0o1000
            O_APPEND = 0o2000
            if flags & O_TRUNC:
                mode = "w"
            elif flags & O_APPEND:
                mode = "a"
            elif flags & (O_WRONLY | O_RDWR):
                # Open-for-write without trunc/append: treat as truncate
                # (real DOS would fail; the simpler semantics is fine).
                mode = "w"
            else:
                mode = "r"
            fd = _vfile_open(name, mode)
            new_eax = (eax & ~0xFFFFFFFF) | (fd & 0xFFFFFFFF)
            uc.reg_write(UC_X86_REG_EAX, new_eax)
            return
        # Unimplemented — record and exit.
        res.error = f"unimplemented INT 21h AH={ah:#04x}"
        uc.emu_stop()

    mu.hook_add(UC_HOOK_INTR, on_int)

    # Set up an instruction-count limit hook to bound runaway programs.
    insn_count = [0]

    def on_code(uc, address, size, user_data):
        insn_count[0] += 1
        if insn_count[0] >= instruction_limit:
            res.timed_out = True
            uc.emu_stop()
        if pktdrv_int_invoke_addr != -1 and address == pktdrv_int_invoke_addr:
            _pktdrv_int_invoke_intercept(uc)

    def _pktdrv_int_invoke_intercept(uc):
        # Stand-in for the libc helper `pktdrv_int_invoke(int_num, regs)`
        # defined in uc386's `lib/i386_dos_libc.asm`. The asm wraps an
        # INT instruction with a `push es / push ds / pop es / ... INT
        # n / ... / pop es` dance so DPMI's rmcs pointer (which lives in
        # ES:EDI) hits a flat data selector instead of the PSP selector
        # DOS hands the program at startup.
        #
        # That body trips two distinct unicorn 2.x bugs the moment any
        # AH we don't otherwise implement reaches it:
        #
        #  - `pop es` faults with spurious #GP even when the value popped
        #    is the null selector (0) — Intel explicitly permits null in
        #    DS/ES; isolated unicorn programs of `pop es` from 0 work
        #    fine, but only in the long-running emulator flow does the
        #    fault appear.
        #
        #  - After we rewrote the asm to use `mov es, ax` (semantically
        #    identical, and confirmed-good in isolation), the subsequent
        #    `pop edx` started returning a phantom value (0xe8000679)
        #    instead of the actual byte at [ESP] (verified 0x00000000 by
        #    direct memory dump). Smells like a JIT cache invalidation
        #    issue around ESP-relative addressing after a hook-handled
        #    INT.
        #
        # Rather than try to patch unicorn or hand-balance the asm to
        # avoid the buggy paths, we just bypass the function entirely:
        # read its args (cdecl `int_num`, `regs`), service the AH=0x48
        # / DPMI calls inline, write the results back into `regs[]`,
        # and short-circuit straight to the caller's return address.
        #
        # The MicroPython port's bounce-buffer prealloc
        # (upstream/ports/minimal/main.c:_preallocate_bounce_buffer)
        # is the only user of this function that runs during dos_emu
        # smoke tests. Real-DOS deployments (DOSBox-X, QEMU FreeDOS)
        # use the real function and a real DPMI host — no interception.
        esp = uc.reg_read(UC_X86_REG_ESP)
        # cdecl: [esp+0] = return addr, [esp+4] = int_num, [esp+8] = regs
        ret_addr, int_num, regs_ptr = struct.unpack(
            "<III", uc.mem_read(esp, 12))
        regs = list(struct.unpack("<8I", uc.mem_read(regs_ptr, 32)))
        eax_in = regs[0]
        ax_in  = eax_in & 0xFFFF
        ah     = (eax_in >> 8) & 0xFF
        carry  = 1

        if int_num == 0x21 and ah == 0x48:
            # INT 21h AH=0x48 — DOS Allocate Memory Block. BX = # of
            # paragraphs requested; on success, AX = segment.
            paragraphs = (regs[1] & 0xFFFF) or 1
            seg = pktdrv_invoke_seg_arena[0]
            pktdrv_invoke_seg_arena[0] += paragraphs
            regs[0] = (eax_in & ~0xFFFF) | (seg & 0xFFFF)
            carry = 0
        elif int_num == 0x31 and ax_in == 0x0002:
            # DPMI fn 0x0002 — Allocate LDT Descriptor for Real Mode
            # Segment. BX = real-mode seg; on success, AX = selector.
            # We don't simulate a real LDT; just hand back the segment
            # value itself as the "selector" — the only thing the
            # caller does with it is hand it to fn 0x0006 below.
            regs[0] = (eax_in & ~0xFFFF) | (regs[1] & 0xFFFF)
            carry = 0
        elif int_num == 0x31 and ax_in == 0x0006:
            # DPMI fn 0x0006 — Get Selector Base Address. BX = selector;
            # on success, CX:DX = 32-bit linear base. We claim the
            # base is seg << 4 (the conventional-memory mapping). This
            # also matches what real DOS extenders do when the selector
            # maps a real-mode segment 1:1.
            sel = regs[1] & 0xFFFF
            linear = (sel << 4) & 0xFFFFFFFF
            regs[2] = (regs[2] & ~0xFFFF) | ((linear >> 16) & 0xFFFF)
            regs[3] = (regs[3] & ~0xFFFF) | (linear & 0xFFFF)
            carry = 0
        elif int_num == 0x31 and ax_in == 0x0100:
            # DPMI fn 0x0100 — Allocate DOS Memory. BX = paragraphs; on
            # success, AX = real-mode segment, DX = selector (we reuse
            # seg as selector, same shape as 0x0002).
            paragraphs = (regs[1] & 0xFFFF) or 1
            seg = pktdrv_invoke_seg_arena[0]
            pktdrv_invoke_seg_arena[0] += paragraphs
            regs[0] = (eax_in & ~0xFFFF) | (seg & 0xFFFF)
            regs[3] = (regs[3] & ~0xFFFF) | (seg & 0xFFFF)
            carry = 0
        elif int_num == 0x31 and ax_in == 0x0301:
            # DPMI fn 0x0301 — Simulate Real Mode Procedure With Far
            # Return. ES:EDI is a flat pointer to a real-mode call
            # structure (rmcs). Dispatch the INT it represents through
            # the same handlers as direct INT 21h, then write the
            # results back into the rmcs.
            #
            # The port (dosint21_uc386dos.c:dos_int21_call) builds an
            # rmcs with cs:ip set to a 3-byte real-mode thunk (`CD 21
            # CB` — int 21h + retf). DPMI 0x0301 would normally jump
            # to that thunk and run it in real mode; we short-circuit
            # by inspecting the rmcs.eax to recover the AH, and
            # dispatching it directly.
            _pktdrv_int_invoke_dispatch_rmcs(uc, regs[5])
            carry = 0
        # Everything else: return CF=1, regs untouched (caller sees a
        # "host doesn't support this call" — valid on every real-DOS
        # host too).

        uc.mem_write(regs_ptr, struct.pack("<8I", *regs))
        # Function returns CF (0 or 1) in EAX (low byte). Caller stores
        # it as `unsigned char carry`.
        uc.reg_write(UC_X86_REG_EAX, carry)
        # Return to caller (cdecl: caller cleans args, we just pop the
        # return-addr dword and set EIP to it).
        uc.reg_write(UC_X86_REG_EIP, ret_addr)
        uc.reg_write(UC_X86_REG_ESP, esp + 4)

    def _pktdrv_int_invoke_dispatch_rmcs(uc, rmcs_linear):
        # rmcs layout from port/dosint21_uc386dos.c:
        #   uint32 edi, esi, ebp, reserved, ebx, edx, ecx, eax;
        #   uint16 flags, es, ds, fs, gs, ip, cs, sp, ss;
        # Total 32 + 18 = 50 bytes (the struct is packed).
        edi, esi, ebp, _, ebx, edx, ecx, eax = struct.unpack(
            "<IIIIIIII", uc.mem_read(rmcs_linear, 32))
        flags, es, ds = struct.unpack("<HHH",
                                       uc.mem_read(rmcs_linear + 32, 6))

        # Reconstruct the linear address the program expected real-mode
        # DS:DX to point to. Under real-DOS DS would be a segment register
        # and DX a 16-bit offset; here both are usable directly because
        # our 0x0002/0x0006 helpers above promised `linear = seg << 4`.
        ds_linear = (ds << 4) & 0xFFFFFFFF
        es_linear = (es << 4) & 0xFFFFFFFF

        ah = (eax >> 8) & 0xFF
        new_eax = eax
        new_flags = flags & ~1  # default: CF=0

        if ah == 0x3D:
            # open(name, mode). Path is DS:DX in real-mode; the bounce
            # buffer that holds the path lives at linear ds_linear.
            name_linear = ds_linear + (edx & 0xFFFF)
            name = bytes(_read_cstr_local(name_linear))
            mode_byte = eax & 0xFF
            if mode_byte == 0:
                fd = _vfile_open(name, "r")
            elif mode_byte in (1, 2):
                fd = _vfile_open(name, "w")
            else:
                fd = -1
            if fd < 0:
                new_eax = (eax & ~0xFFFF) | 0x02   # ENOENT
                new_flags |= 1                      # CF=1
            else:
                new_eax = (eax & ~0xFFFF) | (fd & 0xFFFF)
        elif ah == 0x3F:
            # read(fd, buf, count). DS:DX = buf, ECX = count.
            buf_linear = ds_linear + (edx & 0xFFFF)
            count = ecx & 0xFFFF
            fd = ebx & 0xFFFF
            n = _vfile_read(fd, buf_linear, count)
            if n < 0:
                new_eax = (eax & ~0xFFFF) | 0x06   # EBADF
                new_flags |= 1
            else:
                new_eax = (eax & ~0xFFFF) | (n & 0xFFFF)
        elif ah == 0x40:
            # write(fd, buf, count).
            buf_linear = ds_linear + (edx & 0xFFFF)
            count = ecx & 0xFFFF
            fd = ebx & 0xFFFF
            n = _vfile_write(fd, buf_linear, count)
            if n < 0:
                new_eax = (eax & ~0xFFFF) | 0x06   # EBADF
                new_flags |= 1
            else:
                new_eax = (eax & ~0xFFFF) | (n & 0xFFFF)
        elif ah == 0x3E:
            # close(fd).
            fd = ebx & 0xFFFF
            rc = _vfile_close(fd)
            if rc != 0:
                new_eax = (eax & ~0xFFFF) | 0x06
                new_flags |= 1
        elif ah == 0x42:
            # lseek(fd, offset, whence). CX:DX = 32-bit offset, AL =
            # whence (0=SET, 1=CUR, 2=END). Result in DX:AX.
            fd = ebx & 0xFFFF
            offset = (((ecx & 0xFFFF) << 16) | (edx & 0xFFFF)) & 0xFFFFFFFF
            if offset >= 0x80000000:
                offset -= 0x100000000
            whence = eax & 0xFF
            new_pos = _vfile_seek(fd, offset, whence)
            if new_pos < 0:
                new_eax = (eax & ~0xFFFF) | 0x06
                new_flags |= 1
            else:
                new_eax = (eax & 0xFFFF0000) | (new_pos & 0xFFFF)
                edx = (edx & 0xFFFF0000) | ((new_pos >> 16) & 0xFFFF)
        else:
            # Unrecognized AH inside an rmcs dispatch — set CF=1 so the
            # caller (file_uc386dos.c) sees a generic DOS error and
            # falls back / surfaces an error to Python.
            new_flags |= 1
            new_eax = (eax & ~0xFFFF) | 0x01       # invalid function

        # Write the modified rmcs back. Only the fields the dispatch
        # touched can change in our simplified world; rewrite the
        # whole 32+6 byte head to keep things simple.
        uc.mem_write(rmcs_linear, struct.pack(
            "<IIIIIIII", edi, esi, ebp, 0, ebx, edx, ecx, new_eax))
        uc.mem_write(rmcs_linear + 32, struct.pack(
            "<HHH", new_flags & 0xFFFF, es, ds))

    mu.hook_add(unicorn.UC_HOOK_CODE, on_code)

    try:
        mu.emu_start(
            CODE_BASE,
            CODE_BASE + len(binary),
            timeout=int(timeout_seconds * 1_000_000),
        )
    except UcError as e:
        # Don't overwrite an already-set error / exit.
        if res.exit_code is None and not res.error:
            try:
                eip = mu.reg_read(UC_X86_REG_EIP)
                res.error = f"unicorn: {e} (EIP={eip:#010x})"
            except Exception:
                res.error = f"unicorn: {e}"

    res.instructions_executed = insn_count[0]
    if res.exit_code is None and res.error is None and not res.timed_out:
        # No explicit exit. Either we ran off the end of the binary
        # (treat as exit 0) or unicorn's wallclock timeout fired (treat
        # as timeout). Distinguish by checking EIP — if it's still
        # within the binary, the wallclock killed us.
        eip = mu.reg_read(UC_X86_REG_EIP)
        if CODE_BASE <= eip < CODE_BASE + len(binary):
            res.timed_out = True
        else:
            res.exit_code = 0
    return res


LIBC_ASM_PATH = Path(__file__).resolve().parent / "lib" / "i386_dos_libc.asm"


def _libc_provided_symbols() -> set[str]:
    """Names defined by the bundled libc.asm — these get their `extern`
    declarations stripped from user code before nasm sees them.
    """
    syms: set[str] = set()
    if not LIBC_ASM_PATH.exists():
        return syms
    for line in LIBC_ASM_PATH.read_text().splitlines():
        s = line.strip()
        if s.startswith(";") or not s:
            continue
        # Match `_name:` at the start of a line (label definition).
        if s.startswith("_") and ":" in s:
            label = s.split(":", 1)[0].strip()
            if label.startswith("_") and label[1:].replace("_", "").isalnum():
                syms.add(label[1:])
    return syms


def _user_defined_symbols(asm_text: str) -> set[str]:
    """Names defined in `asm_text` (top-level labels of the form
    `_name:`). Used to decide which libc routines to drop when the
    user code provides its own version (the test program's own
    `sin`/`cos` etc. shadow ours).
    """
    out: set[str] = set()
    for line in asm_text.splitlines():
        s = line.strip()
        if s.startswith(";") or not s:
            continue
        if s.startswith("_") and ":" in s:
            label = s.split(":", 1)[0].strip()
            if label.startswith("_") and "." not in label[1:]:
                # Top-level globals — keep `_name`. Skip `.local` labels.
                if label[1:].replace("_", "").isalnum():
                    out.add(label[1:])
    return out


def _strip_libc_function(libc_text: str, name: str) -> str:
    """Remove the function body labeled `_name:` from libc.asm so the
    user's definition wins. The body extends from `_name:` up to the
    next top-level label.
    """
    lines = libc_text.splitlines()
    out: list[str] = []
    skip = False
    target = f"_{name}:"
    target_alias_pre = f"_{name} "  # tolerate weird formatting
    for line in lines:
        s = line.strip()
        if not skip and (s == target or s.startswith(target_alias_pre)):
            skip = True
            continue
        if skip:
            # Stop skipping when we hit the next top-level label.
            if s.startswith("_") and ":" in s and not s.startswith("."):
                skip = False
        if not skip:
            out.append(line)
    return "\n".join(out)


def _user_referenced_symbols(asm_text: str) -> set[str]:
    """All identifiers starting with `_` that appear in non-comment
    user asm lines, excluding labels defined by the user.

    Catches calls (`call _foo`), function-pointer loads (`mov eax,
    _foo`, `push _foo`), data references (`mov eax, [_foo]`), and
    initialized data (`dd _foo`). Conservative — over-includes is
    fine; under-includes would break linkage.

    `extern _foo` declarations alone do NOT count: the AST optimizer
    can leave behind unused externs (e.g., a `<math.h>` include that
    declared all 162 math functions but the user only calls `sin`).
    Embedding those would defeat the point.
    """
    pattern = re.compile(r"\b(_[_A-Za-z][A-Za-z0-9_]*)\b")
    out: set[str] = set()
    for line in asm_text.splitlines():
        s = line.strip()
        if s.startswith(";") or not s:
            continue
        # Skip extern declarations — they're declarations, not refs.
        if s.startswith("extern "):
            continue
        # Skip global declarations.
        if s.startswith("global "):
            continue
        # Skip section directives.
        if s.startswith("section "):
            continue
        # Strip inline comments.
        idx = line.find(";")
        if idx >= 0:
            line = line[:idx]
        for tok in pattern.findall(line):
            out.add(tok)
    return out


_LIBC_BUNDLE_MARKER = "; ==== bundled libc ===="


def _is_already_bundled(text: str) -> bool:
    """Detect whether asm text already has the libc embedded.

    Reliable signals (from least to most stringent):
    1. The bundle marker comment (`; ==== bundled libc ====`).
    2. Libc-internal data labels — `__heap`, `__heap_ptr`,
       `__sret_buf`, `__stdout`, `__stderr`, etc. — that user code
       never defines. Presence of any of these means libc is
       embedded.

    NOT reliable: presence of any libc-named function body, because
    user code can legitimately shadow a libc symbol (e.g.,
    20020720-1 defines its own `link_error`). We avoid false
    positives that would skip bundling when needed.
    """
    if _LIBC_BUNDLE_MARKER in text:
        return True
    # libc-internal data/bss labels that no user code should define.
    # If any of these appear at column 0 with a `:` suffix, libc is
    # embedded.
    LIBC_INTERNAL_LABELS = (
        "\n__heap:",
        "\n__heap_ptr:",
        "\n__heap_end:",
        "\n__sret_buf:",
        "\n__tmp_qword:",
        "\n__signal_handlers:",
        "\n__stdin:",
        "\n__stdout:",
        "\n__stderr:",
    )
    for marker in LIBC_INTERNAL_LABELS:
        if marker in text:
            return True
    return False


def bundle_text(user_text: str, *, selective_libc: bool = True) -> str:
    """Pure-text version of bundle_user_asm: take user asm and return
    user+libc combined text. Used by both the runtime path
    (`bundle_user_asm` writes a temp file) and the compile-time path
    (main.py embeds before peephole/asm DCE so they see libc too).

    Idempotent: if the asm is already bundled (marker comment OR
    any canonical libc function body is present), returns unchanged.
    """
    if _is_already_bundled(user_text):
        return user_text
    libc_syms = _libc_provided_symbols()
    user_syms = _user_defined_symbols(user_text)

    if selective_libc:
        from .libc_split import parse_libc
        libc_text = LIBC_ASM_PATH.read_text()
        parsed = parse_libc(libc_text)
        for name in list(parsed.functions):
            if name.lstrip("_") in user_syms:
                del parsed.functions[name]
        seeds = _user_referenced_symbols(user_text)
        needed = parsed.transitive_closure(seeds)
        libc_text = parsed.emit(needed)
    else:
        libc_text = LIBC_ASM_PATH.read_text()
        for name in libc_syms & user_syms:
            libc_text = _strip_libc_function(libc_text, name)

    user_lines = user_text.splitlines()
    out_lines: list[str] = []
    marker_inserted = False
    for line in user_lines:
        s = line.strip()
        if s.startswith("extern "):
            name = s[7:].strip().rstrip(",")
            if name.startswith("_") and name[1:] in libc_syms:
                continue
        out_lines.append(line)
        # Inject the marker into the file header so it survives
        # asm DCE (which collects header lines until the first
        # top-level label inside `.text`). We insert right after the
        # `bits 32` directive — early enough to be in the header,
        # late enough to be readable as a structured marker.
        if not marker_inserted and s == "bits 32":
            out_lines.append(_LIBC_BUNDLE_MARKER)
            marker_inserted = True
    # If we didn't find `bits 32` (unusual codegen output), fall
    # back to inserting at the very start so the marker is still
    # in the header.
    if not marker_inserted:
        out_lines.insert(0, _LIBC_BUNDLE_MARKER)
    return (
        "\n".join(out_lines)
        + "\n\n"
        + libc_text
    )


def bundle_user_asm(asm_path: Path, *, selective_libc: bool = True) -> Path:
    """Strip `extern _name` lines for libc-provided symbols and append
    `lib/i386_dos_libc.asm`. Writes the merged asm next to `asm_path`
    with a `.bundled.asm` suffix and returns its path.

    User code may define its own version of a libc symbol (e.g., a
    test that ships its own `sin`). When that happens, we strip the
    matching definition from libc.asm so nasm doesn't see a
    duplicate.

    With `selective_libc=True` (default), only the libc functions
    transitively reachable from the user's externs and call targets
    are embedded.

    If the input asm is already bundled (compile-time embedding via
    `bundle_text` puts a marker line in the file), this function
    short-circuits and returns the path unchanged — no second bundle.
    """
    user_text = asm_path.read_text()
    if _is_already_bundled(user_text):
        # Already bundled at compile time — skip the runtime bundle.
        return asm_path
    bundled_text = bundle_text(user_text, selective_libc=selective_libc)
    bundled = asm_path.with_suffix(".bundled.asm")
    bundled.write_text(bundled_text)
    return bundled


def assemble_and_run(
    asm_path: Path,
    *,
    timeout_seconds: float = 10.0,
    instruction_limit: int = 50_000_000,
    bundle_libc: bool = True,
    keep_intermediate: bool = False,
    stdin_bytes: bytes = b"",
    argv: list[str] | None = None,
    env: dict[str, str] | None = None,
    program_path: str | None = None,
    vfiles_init: dict[bytes, bytes] | None = None,
) -> Result:
    """Convenience: optionally bundle libc, nasm-assemble (-f bin), and run.

    The output binary lives next to `asm_path` with `.bin` suffix.
    """
    import subprocess
    if bundle_libc:
        asm_to_assemble = bundle_user_asm(asm_path)
    else:
        asm_to_assemble = asm_path
    bin_path = asm_to_assemble.with_suffix(".bin")
    proc = subprocess.run(
        ["nasm", "-f", "bin", str(asm_to_assemble), "-o", str(bin_path)],
        capture_output=True, text=True, timeout=15,
    )
    if proc.returncode != 0:
        return Result(error=f"nasm: {proc.stderr.strip()[:400]}")
    res = run(
        bin_path,
        timeout_seconds=timeout_seconds,
        instruction_limit=instruction_limit,
        stdin_bytes=stdin_bytes,
        argv=argv,
        env=env,
        program_path=program_path,
        vfiles_init=vfiles_init,
    )
    if not keep_intermediate:
        try:
            bin_path.unlink()
            # Only delete the bundled asm if we created it (i.e.,
            # it's distinct from the caller's input). When the input
            # is already-bundled, `bundle_user_asm` returns the input
            # path unchanged — deleting that would clobber the caller's
            # source-of-truth asm file.
            if bundle_libc and asm_to_assemble != asm_path:
                asm_to_assemble.unlink()
        except FileNotFoundError:
            pass
    return res
