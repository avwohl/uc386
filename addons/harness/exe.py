#!/usr/bin/env python3
"""Pipeline: uc386 .asm → MZ+LE .exe that runs on FreeDOS / DOSBox / dosiz.

Today this orchestrates external tools rather than emitting the LE
format directly:

    1. NASM (`-f obj`) turns uc386's NASM-syntax .asm into a 32-bit
       OMF (Object Module Format) .obj file. NASM's OMF backend
       produces Watcom-compatible objects with USE32 segments.

    2. Open Watcom's `wlink` consumes the .obj and produces an MZ+LE
       executable. The `system causeway` directive bundles the
       CauseWay DOS extender (~10 KB free stub) into the .exe so the
       result runs unmodified on FreeDOS / DOSBox / dosiz / real DOS,
       no separate `dos4gw.exe` redistribution required.

The pipeline isn't free of caveats — uc386's libc was written
assuming flat-bin layout under dos_emu (INT 21h calls reach our
Python harness directly). Under DOS/4GW or CauseWay those same
INT 21h calls get reflected back to real-mode DOS by the extender,
which means the *extender* loads our binary — so its protected-mode
stack, segment selectors, and PSP are owned by the extender.

Watcom availability: Linux + Windows have native builds. macOS does
not (per the comment in `compare.py`). On macOS the function returns
None and the harness must skip — `compare.py` does this for the same
reason.

Usage:
    python -m addons.harness.exe addons/gnu/echo/main.c -o echo.exe

After build, the .exe runs under DOSBox:
    dosbox echo.exe

Or under dosiz (`../dosiz/dosiz echo.exe` once the LE-loader is
wired up — see `docs/dosiz-integration.md`).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LIB_INCLUDE = REPO_ROOT / "lib" / "include"


# The PMODE/W bridge stub. Linked into every .exe build; provides:
#   - Real DOS handles (0/1/2) for stdin/stdout/stderr (libc's
#     0xF0/F1/F2 dos_emu sentinels are stripped by the asm rewriter).
#   - Captured-at-entry register snapshot for diagnostic programs.
#   - DPMI INT 31h-driven argv parsing: get PSP selector (AX=0x51),
#     translate to linear address (AX=0x06), copy command tail from
#     PSP+0x80 into a flat buffer, tokenize on whitespace into an
#     argv[] of pointers, set EAX=argc and EBX=&argv[0] so the
#     codegen-emitted _start (which `push ebx; push eax`) hands main
#     the right cdecl args.
#
# Empirical PMODE/W contract (from exe_regs_probe.c output):
#   EAX = 0x300 (768)   — opaque, possibly stack/PSP related
#   EBX = 0x21 (33)     — opaque
#   ECX = 0
#   EDX = 0xF159        — looks like a real-mode segment value
#   ESI = 0xD2 / EDI = 0x2CC6 / EBP = 0x2CC6 / ESP = 0x2D00
# None of these are argc/argv. PMODE/W expects the program to parse
# the cmdline itself, same as the Watcom CRT does.
BRIDGE_ASM = """
        section _DATA use32 class=DATA
        global _stdin
        global _stdout
        global _stderr
_stdin:  dd 0
_stdout: dd 1
_stderr: dd 2

        global _pmodew_eax_at_entry
        global _pmodew_ebx_at_entry
        global _pmodew_ecx_at_entry
        global _pmodew_edx_at_entry
        global _pmodew_esi_at_entry
        global _pmodew_edi_at_entry
        global _pmodew_ebp_at_entry
        global _pmodew_esp_at_entry
_pmodew_eax_at_entry: dd 0
_pmodew_ebx_at_entry: dd 0
_pmodew_ecx_at_entry: dd 0
_pmodew_edx_at_entry: dd 0
_pmodew_esi_at_entry: dd 0
_pmodew_edi_at_entry: dd 0
_pmodew_ebp_at_entry: dd 0
_pmodew_esp_at_entry: dd 0

        ; Diagnostic globals for the DPMI argv parser. Exported so a
        ; probe can verify each step worked (selector valid, linear
        ; base computed, cmdline length read, etc.).
        global _pmodew_psp_selector
        global _pmodew_psp_linear
        global _pmodew_cmdline_len
        global _pmodew_argc
        global _pmodew_argv
_pmodew_psp_selector: dd 0
_pmodew_psp_linear:   dd 0
_pmodew_cmdline_len:  dd 0
_pmodew_argc:         dd 0
_pmodew_argv:         dd _pmodew_argv_array

        ; argv array: up to 32 args, NULL-terminated. argv[0] is the
        ; program-name placeholder (DOS PSP doesn't include argv[0];
        ; the real program path requires DPMI 0x60h env-block walk).
_pmodew_argv_array: times 33 dd 0

        ; Flat buffer holding the parsed/tokenized command tail.
        ; PSP cmdline tail max is 127 bytes; double for safety.
_pmodew_argv_buffer: times 256 db 0

        ; Placeholder for argv[0] — real DOS doesn't put the program
        ; name in PSP+0x80, so we fake it with a neutral string. Real
        ; argv[0] needs DPMI 0x60h to walk the environment block,
        ; which lands as a follow-up if needed.
_pmodew_argv0_placeholder: db "program", 0

        section _TEXT use32 class=CODE
        global _pmodew_start
        extern _start

_pmodew_start:
        ; Capture entry state for diagnostics
        mov     [_pmodew_eax_at_entry], eax
        mov     [_pmodew_ebx_at_entry], ebx
        mov     [_pmodew_ecx_at_entry], ecx
        mov     [_pmodew_edx_at_entry], edx
        mov     [_pmodew_esi_at_entry], esi
        mov     [_pmodew_edi_at_entry], edi
        mov     [_pmodew_ebp_at_entry], ebp
        mov     [_pmodew_esp_at_entry], esp

        ; --- argv parsing ---
        ; PMODE/W passes the PSP selector in EBX (=0x21 empirically)
        ; and the PSP real-mode segment in EDX (=0xF1B3 empirically).
        ; Try reading PSP via FS override using PMODE/W's selector.
        ;
        ; History of attempts:
        ; - DPMI INT 31h AX=0x06 (Get Segment Base) returned linear=0
        ;   for selector 0x21. Likely DOSBox/PMODE/W gap.
        ; - Flat DS read at EDX*16 (= 0xF1B30) returned 0 — flat DS
        ;   doesn't map real-mode physical memory in PMODE/W.
        ; - `mov es, 0x21` HUNG the program. ES is critical to PMODE/W
        ;   (probably used for INT 21h reflection); overwriting it
        ;   broke the runtime.
        ; - Trying FS instead: PMODE/W shouldn't touch FS. If FS
        ;   override works, we get the cmdline tail. If it ALSO hangs,
        ;   we fall back to argc=1+placeholder.
        mov     eax, [_pmodew_ebx_at_entry]
        mov     [_pmodew_psp_selector], eax

        mov     eax, [_pmodew_edx_at_entry]
        and     eax, 0xFFFF
        shl     eax, 4
        mov     [_pmodew_psp_linear], eax

        push    fs
        mov     bx, [_pmodew_ebx_at_entry]
        mov     fs, bx

        ; Read command tail length at FS:0x80. Use 32-bit form
        ; explicitly via dword displacement to avoid USE16 ambiguity.
        movzx   eax, byte [fs:0x80]
        mov     [_pmodew_cmdline_len], eax

        ; Copy command tail from FS:0x81 into our flat (DS) buffer.
        push    eax               ; save length
        mov     ecx, eax
        test    ecx, ecx
        jz      .copy_done
        cmp     ecx, 255
        jbe     .copy_ok
        mov     ecx, 255
.copy_ok:
        mov     edi, _pmodew_argv_buffer
        mov     edx, 0x81
.copy_loop:
        mov     al, [fs:edx]
        mov     [edi], al
        inc     edx
        inc     edi
        dec     ecx
        jnz     .copy_loop
.copy_done:
        mov     byte [edi], 0
        pop     eax
        pop     fs

        ; 5. Tokenize. argv[0] = placeholder. Then walk buffer
        ;    splitting on space/tab into argv[1..argc-1].
        mov     edi, _pmodew_argv_array
        mov     dword [edi], _pmodew_argv0_placeholder
        add     edi, 4
        mov     ecx, 1            ; argc starts at 1 (placeholder argv[0])
        mov     esi, _pmodew_argv_buffer

.skip_ws:
        cmp     ecx, 32           ; argv array has 33 slots; leave 1 for NULL
        jge     .tokenize_done
        mov     al, [esi]
        test    al, al
        jz      .tokenize_done
        cmp     al, ' '
        je      .skip_one
        cmp     al, 9
        je      .skip_one
        cmp     al, 13            ; CR — PSP cmdline terminator
        je      .tokenize_done
        ; start of token: record pointer
        mov     [edi], esi
        add     edi, 4
        inc     ecx

.in_token:
        inc     esi
        mov     al, [esi]
        test    al, al
        jz      .tokenize_done
        cmp     al, ' '
        je      .end_token
        cmp     al, 9
        je      .end_token
        cmp     al, 13
        je      .end_token_cr
        jmp     .in_token

.end_token:
        mov     byte [esi], 0     ; null-terminate this token
        inc     esi
        jmp     .skip_ws

.end_token_cr:
        mov     byte [esi], 0
        jmp     .tokenize_done

.skip_one:
        inc     esi
        jmp     .skip_ws

.tokenize_done:
        mov     dword [edi], 0    ; argv[argc] = NULL
        mov     [_pmodew_argc], ecx

        ; 6. Hand off to the codegen-emitted _start. Set up the
        ;    dos_emu register-passing convention: EAX=argc,
        ;    EBX=&argv[0]. _start does `push ebx; push eax`, then
        ;    `call _main` — main reads [ebp+8]=argc, [ebp+12]=argv.
        mov     eax, [_pmodew_argc]
        mov     ebx, _pmodew_argv_array
        jmp     _start
"""


# Same Watcom-discovery pattern as `compare.py` (CI sets WATCOM env;
# dev hosts on Linux typically install via `~/.local/opt/watcom`).
WATCOM_CANDIDATES = [
    "wlink",
    str(Path.home() / ".local/opt/watcom/binl64/wlink"),
    str(Path.home() / ".local/opt/watcom/binl/wlink"),
]
if env := os.environ.get("WATCOM"):
    WATCOM_CANDIDATES.insert(0, str(Path(env) / "binl64/wlink"))
    WATCOM_CANDIDATES.insert(1, str(Path(env) / "binl/wlink"))


def _which_first(candidates: list[str]) -> str | None:
    for c in candidates:
        if "/" in c:
            if Path(c).is_file() and os.access(c, os.X_OK):
                return c
        else:
            found = shutil.which(c)
            if found:
                return found
    return None


def build_exe(
    asm_path: Path,
    out_path: Path,
    *,
    extender: str = "pmodew",
    extra_obj_files: list[Path] | None = None,
) -> tuple[bool, str]:
    """Run nasm + wlink to turn `asm_path` into `out_path` (.exe).

    Returns (ok, message). The message is human-readable on failure
    (preserved stderr from whichever tool died) or empty on success.

    `extender` controls the wlink `system <X>` directive:
        - "pmodew"   : bundles PMODE/W (BSD-ish) — self-contained
                       .exe, ~9 KB stub overhead. Default.
        - "causeway" : LE binary that needs cwstub.exe alongside.
                       (verified empirically: `system causeway`
                       does not bind the extender — it produces a
                       371-byte stub-only .exe whose MZ stub prints
                       "This is a CauseWay executable" and exits.)
        - "dos4g"    : LE binary that needs dos4gw.exe alongside.

    `extra_obj_files` are additional .obj files to link in (e.g. a
    libc shim that bridges between uc386's calling convention and
    DOS/4GW's startup expectations — not yet written, see
    `docs/path-a-mz-le.md` for the plan)."""
    if shutil.which("nasm") is None:
        return False, "nasm not found — install with apt/brew"
    wlink = _which_first(WATCOM_CANDIDATES)
    if wlink is None:
        return False, (
            "wlink not found — install Open Watcom V2 "
            "(https://github.com/open-watcom/open-watcom-v2/releases/"
            "download/Current-build/open-watcom-2_0-c-linux-x64). "
            "Set WATCOM=<install-dir> if it's somewhere unusual. "
            "macOS hasn't a native Watcom build today."
        )

    # uc386 emits `section .text` / `section .data` / `section .bss`
    # without the OMF-specific `use32 class=...` modifiers. NASM's
    # `-f obj` defaults to USE16 segments, which makes the resulting
    # OMF declare 32-bit code as 16-bit. wlink links it cleanly but
    # the LE-loader runs it with the D-bit clear → CPU treats every
    # instruction as 16-bit and execution wanders off into garbage
    # (DOSBox: "Illegal read from 4cb4f3*").
    # Rewrite each section line to include `use32` + an OMF class
    # before NASM sees it.
    #
    # Note on argv: uc386's `_start` does `push ebx; push eax` to
    # convert dos_emu's register-passed argc/argv into cdecl on the
    # stack. Under PMODE/W those registers contain extender-internal
    # state, so the pushes pass garbage to _main. Empirically:
    # `echo hello dos > out.txt` produces `exe hello dos` (argv has
    # 4 elements with argv[1]="exe" — looks like PMODE/W's command-
    # line parser is contributing something through a side channel).
    # Stripping the pushes (tested in Phase 7) didn't change the
    # output, so PMODE/W isn't placing argc/argv on the stack at
    # entry either — argv reaches _main via some channel uc386
    # doesn't read. Real fix needs a bridge stub that:
    #   1. parses PSP+0x80 (real-mode cmdline tail) via DPMI INT 31h
    #   2. allocates argv[] in the LE data segment
    #   3. sets EAX=argc, EBX=&argv[0]
    #   4. jumps to _start
    # That's a separate addons/harness/exe_argv_bridge.asm. For now
    # `.exe` programs that don't read argv work correctly (true,
    # false, yes, factor with default input).
    asm_text = asm_path.read_text()
    rewritten = []
    for line in asm_text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("section .text"):
            rewritten.append("        section _TEXT use32 class=CODE")
            continue
        if stripped.startswith("section .data"):
            rewritten.append("        section _DATA use32 class=DATA")
            continue
        if stripped.startswith("section .bss"):
            rewritten.append("        section _BSS use32 class=BSS")
            continue
        # Strip libc's _stdin/_stdout/_stderr definitions — the
        # bridge stub redefines them with real DOS handles 0/1/2
        # instead of the dos_emu sentinels 0xF0/F1/F2. Match the
        # exact one-line `_stdNN: dd 0xFN` form the codegen emits;
        # don't touch lines that merely *reference* the symbols
        # (e.g. `mov ebx, [_stdout]`).
        if stripped.startswith(("_stdin:", "_stdout:", "_stderr:")) \
                and "dd 0xF" in stripped:
            continue
        rewritten.append(line)
    # Declare the stripped stream globals as externs so user code
    # like `push dword [_stdout]` still assembles. The definitions
    # come from the bridge stub at link time.
    rewritten.insert(0, "        extern _stderr")
    rewritten.insert(0, "        extern _stdout")
    rewritten.insert(0, "        extern _stdin")
    asm_for_omf = out_path.with_suffix(".omf.asm")
    asm_for_omf.write_text("\n".join(rewritten) + "\n")

    obj_path = out_path.with_suffix(".obj")
    proc = subprocess.run(
        ["nasm", "-f", "obj", "-o", str(obj_path), str(asm_for_omf)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return False, f"nasm rc={proc.returncode}: {proc.stderr[:400]}"

    # Phase 7 bridge: dos_emu uses 0xF0/F1/F2 as magic stdin/out/err
    # values (so `fp == NULL` doesn't accidentally match stdin); real
    # DOS via PMODE/W needs raw fd 0/1/2 for INT 21h AH=0x40 (write).
    # A 7-byte mismatch silently breaks every fputs / fwrite / fprintf
    # call — `myecho.exe hello dos > out.txt` produces 767 spaces and
    # no actual content. Patch the globals at PMODE/W entry, then jump
    # to the codegen-emitted _start. argv parsing (PSP+0x80 via DPMI
    # INT 31h) lands in the same stub once stdout is verified working.
    bridge_asm = out_path.with_suffix(".bridge.asm")
    bridge_asm.write_text(BRIDGE_ASM)
    bridge_obj = out_path.with_suffix(".bridge.obj")
    proc = subprocess.run(
        ["nasm", "-f", "obj", "-o", str(bridge_obj), str(bridge_asm)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return False, f"nasm bridge rc={proc.returncode}: {proc.stderr[:400]}"

    # wlink wants WATCOM in env so it can find its stub library.
    env = os.environ.copy()
    if "WATCOM" not in env:
        env["WATCOM"] = str(Path(wlink).parent.parent)

    # Locate the extender stub binary so wlink can BIND it as the
    # MZ portion of the .exe (the file becomes self-contained: real
    # DOS / FreeDOS / DOSBox load the MZ stub, which is the extender
    # itself, which then loads the LE payload that follows).
    # Without `option stub=...`, `system <X>` produces a 371-byte LE
    # whose MZ portion just prints "This is a X executable" and
    # exits — verified empirically in CI.
    stub_name = {
        "pmodew": "pmodew.exe",
        "causeway": "cwstub.exe",
        "dos4g": "dos4gw.exe",
    }.get(extender)
    stub_path: Path | None = None
    if stub_name:
        # Watcom ships these under $WATCOM/binw/ (the 16-bit DOS
        # binaries — the stubs themselves are real-mode .exe).
        candidates = [
            Path(env["WATCOM"]) / "binw" / stub_name,
            Path(env["WATCOM"]) / "binnt" / stub_name,
        ]
        for p in candidates:
            if p.is_file():
                stub_path = p
                break

    cmd = [
        wlink, "system", extender,
        "name", str(out_path),
        "file", str(obj_path),
        # `option stack=64k` allocates a 64-KB protected-mode stack
        # at link time. Without it wlink prints `W1014: stack segment
        # not found` and the .exe runs with a stack at whatever
        # garbage address the LE-loader picks — DOSBox reports
        # "Illegal read from <addr>" when the program tries to push.
        "option", "stack=64k",
        # `option start=_pmodew_start` enters via the bridge stub
        # (fixes stdin/out/err sentinels, future home of argv setup),
        # which falls through to the codegen-emitted `_start` (FPU
        # init, BSS init, call _main, INT 21h AH=4Ch exit).
        "option", "start=_pmodew_start",
        "file", str(bridge_obj),
    ]
    if stub_path is not None:
        # wlink's `option stub=...` directive writes <stub-file>
        # bytes verbatim as the .exe's MZ portion, then writes the
        # LE payload after it.
        cmd.extend(["option", f"stub={stub_path}"])
    for extra in extra_obj_files or []:
        cmd.extend(["file", str(extra)])
    proc = subprocess.run(
        cmd, capture_output=True, text=True, env=env,
    )
    if proc.returncode != 0 or not out_path.exists():
        return False, (
            f"wlink rc={proc.returncode}: "
            f"stdout={proc.stdout[:400]} stderr={proc.stderr[:400]}"
        )

    return True, ""


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="addons.harness.exe",
        description=__doc__.splitlines()[0],
    )
    ap.add_argument("source", help=".c source to compile, OR .asm to skip uc386")
    ap.add_argument("-o", "--output", required=True, help="output .exe path")
    ap.add_argument(
        "--extender", default="pmodew",
        choices=["pmodew", "causeway", "dos4g"],
        help="DOS extender to bundle (default: pmodew)",
    )
    args = ap.parse_args()

    src = Path(args.source).resolve()
    out = Path(args.output).resolve()

    # If a .c is provided, run uc386 first to produce the .asm.
    if src.suffix == ".c":
        asm_path = out.with_suffix(".asm")
        proc = subprocess.run(
            [sys.executable, "-m", "uc386.main", str(src),
             "-o", str(asm_path), "-I", str(LIB_INCLUDE)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            sys.stderr.write(
                f"uc386 rc={proc.returncode}: {proc.stderr[:400]}\n"
            )
            return 1
    elif src.suffix == ".asm":
        asm_path = src
    else:
        sys.stderr.write(f"unrecognised extension: {src.suffix}\n")
        return 2

    ok, msg = build_exe(asm_path, out, extender=args.extender)
    if not ok:
        sys.stderr.write(f"exe build failed: {msg}\n")
        return 1
    print(f"wrote {out} ({out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
