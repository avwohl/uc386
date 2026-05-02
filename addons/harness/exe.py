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

    obj_path = out_path.with_suffix(".obj")
    proc = subprocess.run(
        ["nasm", "-f", "obj", "-o", str(obj_path), str(asm_path)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return False, f"nasm rc={proc.returncode}: {proc.stderr[:400]}"

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
        # `option start=_start` overrides wlink's default of looking
        # for `_cstart_` (Watcom clib startup). uc386's prologue is
        # `_start:` which sets up FPU/BSS then jumps to `_main`.
        "option", "start=_start",
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
