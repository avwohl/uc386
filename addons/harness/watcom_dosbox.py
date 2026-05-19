#!/usr/bin/env python3
"""Run the DOS-hosted Open Watcom V2 toolchain under DOSBox-X.

Open Watcom V2 ships no native macOS build (and its Linux binaries
won't run under Rosetta — Rosetta bridges x86_64→arm64 user space,
not the Linux ABI). But the *DOS-hosted* `wcc386.exe` / `wlink.exe`
run fine under DOSBox-X, which IS available on macOS/Linux. That is
the only honest way to put a real Open Watcom size column next to
uc386 on a Mac, so the README's "smaller than Watcom" claim can be
checked instead of asserted.

The toolchain tree is the `binw/` + `h/` + `lib386/` subset unzipped
from the `open-watcom-2_0-c-dos.exe` release asset (the installer is
a plain self-extracting zip — `unzip` extracts it directly, no
interactive install). Default search:

    $WATCOM_DOS_DIR
    ~/.local/opt/watcom-dos
    /tmp/watcom

Install (macOS / Linux), ~99 MB extracted:

    curl -sL -o /tmp/ow.exe \\
      https://github.com/open-watcom/open-watcom-v2/releases/download/Current-build/open-watcom-2_0-c-dos.exe
    mkdir -p ~/.local/opt/watcom-dos
    unzip -q /tmp/ow.exe 'binw/*' 'h/*' 'lib386/dos/*' 'lib386/*.lib' \\
      -d ~/.local/opt/watcom-dos

Plus `brew install dosbox-x` (macOS) or the distro package (Linux).

This module is intentionally dependency-free (stdlib only) so the
freedos_git / freedos_micro_python sibling repos can vendor a thin
wrapper around it without pulling uc386 in.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

WATCOM_DOS_CANDIDATES = [
    os.environ.get("WATCOM_DOS_DIR", ""),
    str(Path.home() / ".local/opt/watcom-dos"),
    "/tmp/watcom",
]


def find_watcom_dos() -> Path | None:
    """Locate an extracted DOS Open Watcom tree (must have binw/wcc386.exe)."""
    for c in WATCOM_DOS_CANDIDATES:
        if not c:
            continue
        p = Path(c)
        if (p / "binw" / "wcc386.exe").is_file():
            return p
    return None


def have_toolchain() -> bool:
    """True iff both DOSBox-X and an extracted Watcom DOS tree are present."""
    return shutil.which("dosbox-x") is not None and find_watcom_dos() is not None


def _short_name(idx: int, src: Path) -> str:
    """An 8.3, collision-free DOS name. We don't trust upstream basenames
    to be unique-in-8.3 across a multi-TU project (git has dozens of
    same-prefix files), so key purely on position: S00.C, S01.C, ..."""
    return f"S{idx:02d}.C"


def build_watcom_dos(
    sources: list[Path],
    out_exe: Path,
    *,
    includes: list[Path] | None = None,
    defines: list[str] | None = None,
    extender: str = "dos4g",
    extra_cflags: list[str] | None = None,
    timeout: int = 600,
) -> tuple[bool, str]:
    """Compile `sources` with DOS-hosted wcc386 and link with wlink,
    all under DOSBox-X. Writes `out_exe` on the host on success.

    Returns (ok, message). `message` is a short diagnostic on failure.

    `extender` is the wlink `system <X>` directive: dos4g (default;
    LE binary, needs dos4gw/dos32a at runtime — but the .exe size is
    what we measure, matching `compare.py`'s historical column) or
    causeway. The compared size is the *bound .exe on disk*, the same
    artifact `compare.py` has always measured for the Watcom column.
    """
    wc = find_watcom_dos()
    if wc is None:
        return False, "no Watcom DOS tree (see module docstring to install)"
    if shutil.which("dosbox-x") is None:
        return False, "dosbox-x not found (brew install dosbox-x)"

    includes = includes or []
    defines = defines or []
    extra_cflags = extra_cflags or []

    work = Path(tempfile.mkdtemp(prefix="wcdos_"))
    try:
        # Stage each source under an 8.3 name in the DOS-visible scratch.
        staged: list[str] = []
        for i, s in enumerate(sources):
            name = _short_name(i, s)
            shutil.copyfile(s, work / name)
            staged.append(name)

        # Include dirs are mounted as I: and referenced 8.3-style. We
        # keep it simple: one extra include root is the common case
        # (Watcom's own h/ is always on INCLUDE). Multiple roots get
        # copied flat into an INC\ dir on the scratch — fine for the
        # header-light addons; the big ports pass their own -I tree.
        inc_flags = []
        if includes:
            incdir = work / "INC"
            incdir.mkdir()
            for root in includes:
                for h in Path(root).rglob("*.h"):
                    dst = incdir / h.name
                    if not dst.exists():
                        try:
                            shutil.copyfile(h, dst)
                        except OSError:
                            pass
            inc_flags = ["-i=C:\\INC"]

        # -otexan: full size/speed optimization (the fair analogue of
        # gcc -O2 / uc386's DCE). Matches compare.py's intent.
        cflags = ["-bt=dos", "-q", "-ze", "-otexan", *inc_flags,
                  *[f"-d{d}" for d in defines], *extra_cflags]

        objs = []
        autoexec = [
            "mount c " + str(work),
            "mount w " + str(wc),
            "set WATCOM=W:\\",
            "set INCLUDE=W:\\H",
            "PATH=W:\\BINW;Z:\\",
            "c:",
        ]
        for name in staged:
            obj = name[:-2] + ".O"
            objs.append(obj)
            autoexec.append(f"wcc386 {' '.join(cflags)} -fo={obj} {name}")
        # wlink defaults the `file` extension to .OBJ; wcc386 wrote .O,
        # so the extension MUST be explicit here. (The benign "Packed
        # file is corrupt" DOSBox-X warning on the LZ-packed Watcom
        # binaries does not actually stop the build.)
        files_dir = "+".join(objs)
        autoexec += [
            f"wlink system {extender} name OUT.EXE file {files_dir}",
            "exit",
        ]

        conf = work / "db.conf"
        conf.write_text(
            "[dosbox]\nmachine=svga_s3\n[cpu]\ncore=auto\n"
            "cputype=pentium_slow\n[dos]\nxms=true\nems=true\n"
            "[autoexec]\n" + "\n".join(autoexec) + "\n"
        )

        proc = subprocess.run(
            ["dosbox-x", "-silent", "-exit", "-conf", str(conf)],
            capture_output=True, text=True, timeout=timeout,
        )
        produced = work / "OUT.EXE"
        if not produced.is_file() or produced.stat().st_size == 0:
            tail = (proc.stdout or proc.stderr or "")[-300:]
            return False, f"wlink produced no OUT.EXE (dosbox tail: {tail!r})"
        out_exe.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(produced, out_exe)
        return True, ""
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(prog="addons.harness.watcom_dosbox")
    ap.add_argument("sources", nargs="+", type=Path)
    ap.add_argument("-o", "--out", required=True, type=Path)
    ap.add_argument("-I", "--include", action="append", default=[], type=Path)
    ap.add_argument("-D", "--define", action="append", default=[])
    ap.add_argument("--extender", default="dos4g")
    args = ap.parse_args()
    if not have_toolchain():
        print("Watcom DOS toolchain or dosbox-x missing — see "
              "addons/harness/watcom_dosbox.py docstring", file=sys.stderr)
        return 2
    ok, msg = build_watcom_dos(
        args.sources, args.out,
        includes=args.include, defines=args.define, extender=args.extender,
    )
    if not ok:
        print(f"FAILED: {msg}", file=sys.stderr)
        return 1
    print(f"{args.out} {args.out.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
