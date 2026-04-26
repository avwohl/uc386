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
import struct

import unicorn
from unicorn import Uc, UC_ARCH_X86, UC_MODE_32, UC_HOOK_INTR, UcError
from unicorn.x86_const import (
    UC_X86_REG_EAX, UC_X86_REG_EBX, UC_X86_REG_ECX, UC_X86_REG_EDX,
    UC_X86_REG_ESI, UC_X86_REG_EDI, UC_X86_REG_EBP, UC_X86_REG_ESP,
    UC_X86_REG_EIP, UC_X86_REG_EFLAGS,
)


# Memory layout
#   0x00000000 .. 0x00800000   code/data (8 MB) — the loaded binary lives here
#   0x00800000 .. 0x01000000   heap (8 MB, growable in principle)
#   0x01000000 .. 0x01100000   stack (1 MB, top at 0x010FFFF0)
CODE_BASE = 0x00000000
CODE_SIZE = 0x00800000
HEAP_BASE = 0x00800000
HEAP_SIZE = 0x00800000
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
) -> Result:
    """Emulate a flat-binary i386 program; return its stdout + exit code."""
    if isinstance(binary, Path):
        binary = binary.read_bytes()

    mu = Uc(UC_ARCH_X86, UC_MODE_32)
    mu.mem_map(CODE_BASE, CODE_SIZE)
    mu.mem_map(STACK_BASE, STACK_SIZE)

    # Load the program at address 0 (matches NASM `-f bin` default org 0).
    mu.mem_write(CODE_BASE, binary)

    # Initialize stack near the top of the stack region. Push a fake return
    # address (0xFFFFFFFF) so `ret` from the entry function ends up at an
    # unmapped location — we treat that as a clean exit. Entry doesn't
    # actually return for our test programs (they call INT 21h AH=4C), but
    # this protects against malformed code.
    esp = STACK_TOP
    mu.reg_write(UC_X86_REG_ESP, esp)
    mu.reg_write(UC_X86_REG_EBP, esp)

    res = Result()
    stdin_pos = [0]

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
            # width
            width = 0
            while i < n and 0x30 <= fmt[i] <= 0x39:
                width = width * 10 + (fmt[i] - 0x30)
                i += 1
            # precision
            precision = -1
            if i < n and fmt[i:i+1] == b".":
                i += 1
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

            def read32_le(addr_ref):
                bs = bytes(mu.mem_read(addr_ref[0], 4))
                addr_ref[0] += 4
                return int.from_bytes(bs, "little")

            def read64_le(addr_ref):
                bs = bytes(mu.mem_read(addr_ref[0], 8))
                addr_ref[0] += 8
                return int.from_bytes(bs, "little")

            ap_box = [ap]
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
                    if plus_flag:
                        s = b"+" + str(val).encode()
                    elif space_flag:
                        s = b" " + str(val).encode()
                    else:
                        s = str(val).encode()
                else:
                    s = str(val).encode()
                pad = b"0" if zero_pad else b" "
                if width > len(s):
                    if left_align:
                        s = s + b" " * (width - len(s))
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
                    s = str(val).encode()
                elif conv == b"x":
                    s = f"{val:x}".encode()
                    if hash_flag and val != 0:
                        s = b"0x" + s
                elif conv == b"X":
                    s = f"{val:X}".encode()
                    if hash_flag and val != 0:
                        s = b"0X" + s
                else:  # o
                    s = f"{val:o}".encode()
                    if hash_flag and not s.startswith(b"0"):
                        s = b"0" + s
                pad = b"0" if zero_pad else b" "
                if width > len(s):
                    if left_align:
                        s = s + b" " * (width - len(s))
                    else:
                        s = pad * (width - len(s)) + s
                out += s
            elif conv == b"c":
                val = read32_le(ap_box)
                out += bytes([val & 0xFF])
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
            elif conv in (b"f", b"g", b"e"):
                bs = bytes(mu.mem_read(ap_box[0], 8))
                ap_box[0] += 8
                import struct as _st
                val = _st.unpack("<d", bs)[0]
                if precision < 0:
                    precision = 6
                fmt_py = f"%.{precision}{conv.decode()}"
                s = (fmt_py % val).encode()
                if width > len(s):
                    if left_align:
                        s = s + b" " * (width - len(s))
                    else:
                        s = b" " * (width - len(s)) + s
                out += s
            else:
                # Unknown — output as-is.
                out += b"%" + conv
            ap = ap_box[0]
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
            # fprintf via harness: EBX=stream(fd 1=stdout, 2=stderr),
            # ECX=fmt, EDX=va_ptr.
            ebx = uc.reg_read(UC_X86_REG_EBX)
            ecx = uc.reg_read(UC_X86_REG_ECX)
            edx = uc.reg_read(UC_X86_REG_EDX)
            fmt = _read_cstr_local(ecx)
            formatted = _printf_format(fmt, edx)
            if (ebx & 0xFFFF) == 2:
                _write_stderr(formatted)
            else:
                _write_stdout(formatted)
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
            data = bytes(uc.mem_read(edx, count))
            fd = ebx & 0xFFFF
            if fd == 1:
                _write_stdout(data)
            elif fd == 2:
                _write_stderr(data)
            # Return bytes-written in AX; CF clear on success.
            new_eax = (eax & ~0xFFFF) | (count & 0xFFFF)
            uc.reg_write(UC_X86_REG_EAX, new_eax)
            return
        if ah == 0x3F:
            # read(fd, buf, count): BX=fd, CX=count, DS:EDX=buf. Only fd=0
            # (stdin) handled.
            ebx = uc.reg_read(UC_X86_REG_EBX)
            ecx = uc.reg_read(UC_X86_REG_ECX)
            edx = uc.reg_read(UC_X86_REG_EDX)
            count = ecx & 0xFFFF
            fd = ebx & 0xFFFF
            if fd == 0:
                start = stdin_pos[0]
                end = min(start + count, len(stdin_bytes))
                chunk = stdin_bytes[start:end]
                if chunk:
                    uc.mem_write(edx, chunk)
                stdin_pos[0] = end
                actual = len(chunk)
            else:
                actual = 0
            new_eax = (eax & ~0xFFFF) | (actual & 0xFFFF)
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


def bundle_user_asm(asm_path: Path) -> Path:
    """Strip `extern _name` lines for libc-provided symbols and append
    `lib/i386_dos_libc.asm`. Writes the merged asm next to `asm_path`
    with a `.bundled.asm` suffix and returns its path.

    User code may define its own version of a libc symbol (e.g., a
    test that ships its own `sin`). When that happens, we strip the
    matching definition from libc.asm so nasm doesn't see a
    duplicate.
    """
    libc_syms = _libc_provided_symbols()
    user_text = asm_path.read_text()
    user_syms = _user_defined_symbols(user_text)
    libc_text = LIBC_ASM_PATH.read_text()
    # Drop libc definitions that the user provides.
    for name in libc_syms & user_syms:
        libc_text = _strip_libc_function(libc_text, name)
    user_lines = user_text.splitlines()
    out_lines: list[str] = []
    for line in user_lines:
        s = line.strip()
        # Strip lines like `extern _printf` for any libc-provided name.
        if s.startswith("extern "):
            name = s[7:].strip().rstrip(",")
            if name.startswith("_") and name[1:] in libc_syms:
                continue
        out_lines.append(line)
    bundled = asm_path.with_suffix(".bundled.asm")
    bundled.write_text(
        "\n".join(out_lines) + "\n\n; ==== bundled libc ====\n"
        + libc_text
    )
    return bundled


def assemble_and_run(
    asm_path: Path,
    *,
    timeout_seconds: float = 10.0,
    instruction_limit: int = 50_000_000,
    bundle_libc: bool = True,
    keep_intermediate: bool = False,
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
    )
    if not keep_intermediate:
        try:
            bin_path.unlink()
            if bundle_libc:
                asm_to_assemble.unlink()
        except FileNotFoundError:
            pass
    return res
