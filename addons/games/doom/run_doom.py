#!/usr/bin/env python3
"""Run the uc386-built DOOM under dos_emu with a user-supplied WAD.

Usage:
    .venv/bin/python addons/games/doom/run_doom.py /path/to/doom1.wad

Builds (if needed) and runs `addons/games/doom/build/doom.bin` under
`uc386.dos_emu.run`, mounting the WAD at `/doom1.wad` so DOOM's
shareware-mode detection finds it.

The WAD can be any of:
  - A user-owned shareware DOOM 1 WAD (legal: doom1.wad is freely
    redistributable per id Software's 1997 release).
  - Freedoom (https://freedoom.github.io/) — a GPL-licensed open
    WAD that's compatible with DOOM's WAD format.
  - Any other doom1-shaped WAD with the lump structure DOOM expects.

DOOM will reach the title-screen tic loop, but won't render —
addons/games/doom/stubs.c's I_FinishUpdate is a no-op (no real
video stub), and I_StartTic produces no input events. To see
actual rendering, those stubs would need to wire up a Pillow /
SDL window or a frame-hash dump.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from uc386.dos_emu import run

DOOM_BIN = REPO_ROOT / "addons" / "games" / "doom" / "build" / "doom.bin"


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <doom1.wad>", file=sys.stderr)
        return 2
    wad_path = Path(sys.argv[1])
    if not wad_path.exists():
        print(f"WAD not found: {wad_path}", file=sys.stderr)
        return 1
    if not DOOM_BIN.exists():
        print(f"DOOM bin not found at {DOOM_BIN}; run build.sh first.",
              file=sys.stderr)
        return 1

    wad_data = wad_path.read_bytes()
    blob = DOOM_BIN.read_bytes()
    print(f"Running DOOM ({len(blob)} bytes) with {wad_path.name} "
          f"({len(wad_data)} bytes) ...", file=sys.stderr)

    # DOOM's iwad search looks at $DOOMWADDIR (uc386 libc returns "/")
    # then probes doom1.wad, doom.wad, etc. Mount the WAD under all
    # the names DOOM might check so the detection chain always finds it.
    vfiles = {
        b"/doom1.wad": wad_data,
        b"//doom1.wad": wad_data,
        b"./doom1.wad": wad_data,
        b"doom1.wad": wad_data,
    }

    res = run(
        blob,
        timeout_seconds=30.0,
        instruction_limit=500_000_000,
        argv=["doom"],
        vfiles_init=vfiles,
    )

    sys.stdout.write(res.stdout)
    sys.stderr.write(res.stderr)
    if res.error:
        sys.stderr.write(f"\n[dos_emu error] {res.error}\n")
    return res.exit_code or 0


if __name__ == "__main__":
    sys.exit(main())
