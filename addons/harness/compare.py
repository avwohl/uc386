#!/usr/bin/env python3
"""Build each addon with host gcc + uc386 + (optional) watcom/djgpp,
report sizes side-by-side. Output is a markdown table written to
`addons/results.md`.

Today only gcc + uc386 are wired. Watcom (`wcc386`) and djgpp need
cross-compilers we don't currently install; the table reserves
columns for them so the format is stable.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ADDONS_ROOT = REPO_ROOT / "addons"
LIB_INCLUDE = REPO_ROOT / "lib" / "include"

# DJGPP (i586-pc-msdosdjgpp-gcc) ships as an out-of-tree tarball — we
# look in $PATH, $DJGPP_PREFIX/bin, and the conventional ~/.local/opt/djgpp
# install we use on the dev host.
DJGPP_CANDIDATES = [
    "i586-pc-msdosdjgpp-gcc",
    str(Path.home() / ".local/opt/djgpp/bin/i586-pc-msdosdjgpp-gcc"),
    "/usr/local/djgpp/bin/i586-pc-msdosdjgpp-gcc",
]
if env := os.environ.get("DJGPP_PREFIX"):
    DJGPP_CANDIDATES.insert(0, str(Path(env) / "bin/i586-pc-msdosdjgpp-gcc"))

# Watcom: we never run wcc386 on macOS arm64 (no native build), but the
# CI runner is Linux-x64 where the upstream tarball works fine. Detection
# walks $WATCOM/binl64/wcc386, then plain $PATH.
WATCOM_CANDIDATES = ["wcc386"]
if env := os.environ.get("WATCOM"):
    WATCOM_CANDIDATES.insert(0, str(Path(env) / "binl64/wcc386"))
    WATCOM_CANDIDATES.insert(1, str(Path(env) / "binl/wcc386"))


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


def find_addons() -> list[Path]:
    """Each addon dir under addons/gnu/ that has a main.c counts."""
    out: list[Path] = []
    for sub in sorted((ADDONS_ROOT / "gnu").iterdir()):
        if not sub.is_dir() or sub.name.startswith("_"):
            continue
        if (sub / "main.c").exists():
            out.append(sub)
    return out


def build_gcc(addon: Path) -> int | None:
    """Build addon/main.c with host gcc; return ELF size or None on fail."""
    if not shutil.which("gcc"):
        return None
    out = REPO_ROOT / "build" / "compare" / addon.name
    out.mkdir(parents=True, exist_ok=True)
    bin_path = out / "gcc.bin"
    proc = subprocess.run(
        ["gcc", "-O2", "-w", str(addon / "main.c"), "-o", str(bin_path)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return None
    return bin_path.stat().st_size


def build_uc386(addon: Path) -> int | None:
    """Build via the harness path: uc386 → bundle libc → nasm → bin size."""
    out = REPO_ROOT / "build" / "compare" / addon.name
    out.mkdir(parents=True, exist_ok=True)
    asm_path = out / "uc386.asm"
    proc = subprocess.run(
        [sys.executable, "-m", "uc386.main", str(addon / "main.c"),
         "-o", str(asm_path), "-I", str(LIB_INCLUDE)],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    if proc.returncode != 0:
        return None
    bin_path = out / "uc386.bin"
    proc = subprocess.run(
        ["nasm", "-f", "bin", str(asm_path), "-o", str(bin_path)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return None
    return bin_path.stat().st_size


def build_watcom(addon: Path) -> int | None:
    """Build via Open Watcom wcc386. Linux/Windows hosts only — there is
    no native macOS build, so this returns None on the dev host and the
    real numbers come from the CI runner."""
    wcc = _which_first(WATCOM_CANDIDATES)
    if wcc is None:
        return None
    # wcc386 needs WATCOM env (for h/, lib386/) and INCLUDE for stdio.h
    # etc. If WATCOM isn't already set, default it from the wcc386 path.
    env = os.environ.copy()
    if "WATCOM" not in env:
        env["WATCOM"] = str(Path(wcc).parent.parent)
    if "INCLUDE" not in env:
        env["INCLUDE"] = str(Path(env["WATCOM"]) / "h")
    out = REPO_ROOT / "build" / "compare" / addon.name
    out.mkdir(parents=True, exist_ok=True)
    obj_path = out / "wcc386.o"
    exe_path = out / "wcc386.exe"
    # wcc386 wants /fo=obj, the linker is wlink. -ze enables Watcom
    # language extensions including "declarations anywhere in a block",
    # which most of our addons use (period code did this too once C99
    # was available). Without it the C89 default rejects mid-block
    # decls with E1063.
    proc = subprocess.run(
        [wcc, "-bt=dos", "-q", "-ze", "-fo=" + str(obj_path), str(addon / "main.c")],
        capture_output=True, text=True, cwd=out, env=env,
    )
    if proc.returncode != 0:
        # Surface the error to the workflow log so we can debug.
        sys.stderr.write(
            f"watcom {addon.name}: wcc386 rc={proc.returncode}\n"
            f"  stdout: {proc.stdout[:400]}\n"
            f"  stderr: {proc.stderr[:400]}\n"
        )
        return None
    wlink_candidates = ["wlink"]
    if "/" in wcc:
        wlink_candidates.insert(0, str(Path(wcc).parent / "wlink"))
    wlink = _which_first(wlink_candidates)
    if wlink is None:
        sys.stderr.write(f"watcom {addon.name}: wlink not found in PATH or {Path(wcc).parent}\n")
        return None
    proc = subprocess.run(
        [wlink, "system", "dos4g", "name", str(exe_path), "file", str(obj_path)],
        capture_output=True, text=True, cwd=out, env=env,
    )
    if proc.returncode != 0 or not exe_path.exists():
        sys.stderr.write(
            f"watcom {addon.name}: wlink rc={proc.returncode} exists={exe_path.exists()}\n"
            f"  stdout: {proc.stdout[:400]}\n"
            f"  stderr: {proc.stderr[:400]}\n"
        )
        return None
    return exe_path.stat().st_size


def build_djgpp(addon: Path) -> int | None:
    """Build with the DJGPP cross-compiler (i586-pc-msdosdjgpp-gcc). The
    output is a COFF DOS executable for the go32 DPMI extender."""
    djgcc = _which_first(DJGPP_CANDIDATES)
    if djgcc is None:
        return None
    out = REPO_ROOT / "build" / "compare" / addon.name
    out.mkdir(parents=True, exist_ok=True)
    exe_path = out / "djgpp.exe"
    proc = subprocess.run(
        [djgcc, "-O2", "-w", str(addon / "main.c"), "-o", str(exe_path)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return None
    return exe_path.stat().st_size


def main() -> int:
    ap = argparse.ArgumentParser(prog="addons.harness.compare")
    ap.add_argument("--write", action="store_true",
                    help="write results to addons/results.md (default: stdout)")
    args = ap.parse_args()

    addons = find_addons()
    rows: list[tuple[str, int | None, int | None, int | None, int | None]] = []
    for addon in addons:
        gcc_sz = build_gcc(addon)
        uc_sz = build_uc386(addon)
        wcc_sz = build_watcom(addon)
        djgpp_sz = build_djgpp(addon)
        rows.append((addon.name, uc_sz, gcc_sz, wcc_sz, djgpp_sz))

    def cell(n: int | None) -> str:
        return f"{n:,}" if n is not None else "—"

    md = ["# Build comparison: uc386 vs gcc / Watcom / DJGPP", "",
          "Sizes in bytes of the produced executable. The four columns ",
          "use four different output formats — the comparison is about ",
          "_how big does each compiler's smallest hello-world-class ",
          "program get_, not strict like-for-like:", "",
          "- **uc386**: flat i386 `.bin` that runs under dos_emu. ",
          "  Compile-time libc bundling + asm DCE strip everything ",
          "  the program doesn't reference.",
          "- **gcc**: native host ELF (Linux/macOS). Includes glibc ",
          "  startup machinery — biggest by a wide margin.",
          "- **Watcom (`wcc386`)**: 32-bit DOS/4GW `.exe`. Empty ",
          "  cells mean the toolchain wasn't available in this build ",
          "  environment (no native macOS build; CI populates it on ",
          "  the Linux-x64 runner).",
          "- **DJGPP**: COFF DOS executable for the go32 DPMI ",
          "  extender (gcc 12.2.0 cross). Includes the full djgpp ",
          "  C runtime — explains why even `true` is ~148 KB.", "",
          "| Tool | uc386 | gcc (host ELF) | Watcom wcc386 | DJGPP |",
          "|------|-------|----------------|---------------|-------|"]
    for name, uc_sz, gcc_sz, wcc_sz, djgpp_sz in rows:
        md.append(
            f"| {name} | {cell(uc_sz)} | {cell(gcc_sz)} | {cell(wcc_sz)} | {cell(djgpp_sz)} |"
        )
    md.append("")
    md.append("_Generated by `python -m addons.harness.compare --write`._")
    text = "\n".join(md)
    if args.write:
        out_path = ADDONS_ROOT / "results.md"
        out_path.write_text(text + "\n")
        print(f"Wrote {out_path}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
