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
#   0x00F00000 .. 0x00F01000   argv area (4 KB)
#   0x01000000 .. 0x01100000   stack (1 MB, top at 0x010FFFF0)
CODE_BASE = 0x00000000
CODE_SIZE = 0x00800000
HEAP_BASE = 0x00800000
HEAP_SIZE = 0x00800000
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
    vfiles_init: dict[bytes, bytes] | None = None,
) -> Result:
    """Emulate a flat-binary i386 program; return its stdout + exit code.

    `vfiles_init` seeds the virtual file system with named files
    visible to fopen("name", "r"). Pass `{b"data.txt": b"hello\n"}`
    so a program can open and read `data.txt`.
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


LIBC_ASM_PATH = Path(__file__).resolve().parents[2] / "lib" / "i386_dos_libc.asm"


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
