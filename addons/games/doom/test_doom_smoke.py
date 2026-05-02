#!/usr/bin/env python3
"""Smoke test for the uc386-built DOOM binary.

Runs `addons/games/doom/build/doom.bin` under uc386.dos_emu without a
WAD and asserts:
  - exits cleanly (no emulator error / no timeout)
  - reaches `W_Init: Init WADfiles.` in stdout (the boot-progress line
    one INT past `Z_Init`)
  - exits with code 1 (DOOM's "no WAD found" exit; documented in
    addons/STATUS.md)

Skipped when `doom.bin` doesn't exist — the binary is a derived
artifact (built by `addons/games/doom/build.sh`); we don't check it
in. Run after a build to guard against regressions in either the
codegen pipeline or DOOM's boot sequence.

Usage (dev tree):
    .venv/bin/pytest addons/games/doom/test_doom_smoke.py
Usage (unpacked games tarball — DOOM lives under bin/doom/):
    pytest test_doom_smoke.py    # adjust path; tarball ships this file too
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Two locations: the dev-tree path and the games-tarball path.
#
# Dev tree:  addons/games/doom/test_doom_smoke.py → build/doom.bin
#   = parent / build / doom.bin
# Tarball:   uc386-games/doom/test_doom_smoke.py → bin/doom/doom.bin
#            (sibling to the doom/ dir under uc386-games/)
#   = parent.parent / bin / doom / doom.bin
_HERE = Path(__file__).resolve().parent
_BIN_CANDIDATES = [
    _HERE / "build" / "doom.bin",
    _HERE.parent / "bin" / "doom" / "doom.bin",
]


def _find_doom_bin() -> Path | None:
    for p in _BIN_CANDIDATES:
        if p.exists():
            return p
    return None


@pytest.fixture(scope="module")
def doom_bin() -> Path:
    p = _find_doom_bin()
    if p is None:
        pytest.skip(
            "doom.bin not built — run addons/games/doom/build.sh first"
        )
    return p


def test_doom_boots_to_wad_load(doom_bin: Path) -> None:
    """DOOM should boot through every visible init phase and reach
    W_InitFiles, exiting 1 because we don't supply a WAD.

    Asserts the full sequence of boot-progress lines that DOOM emits
    in `D_DoomMain` before it tries to load any WAD. Each line below
    corresponds to a different init function (the linuxdoom-1.10
    source has them in this order in `i_main.c` / `d_main.c`):

      Game mode indeterminate.   ← D_IdentifyVersion has no WAD
      Public DOOM - v1.10        ← printed banner
      V_Init: allocate screens.  ← V_Init  (memory for screen[0..3])
      M_LoadDefaults             ← config-file load
      Z_Init                     ← zone allocator + main heap
      W_Init: Init WADfiles      ← WAD loader (this is where it fails)

    Reaching ALL of these means uc386 codegen is correct for the
    full boot path: int locals, struct return, pointer arithmetic,
    static-init data tables, switch-case, libc calls (printf, malloc,
    open, lseek, read, exit). If any phase regresses, the test will
    say which one didn't print."""
    # Make uc386 importable when running from the dev tree (the
    # tarball variant assumes the user's already pip-installed it).
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from uc386.dos_emu import run

    res = run(doom_bin, timeout_seconds=10.0,
              instruction_limit=2_000_000_000)
    assert not res.timed_out, "DOOM hit the dos_emu timeout"
    assert res.error is None, f"dos_emu reported error: {res.error}"

    # Each init phase the runnable boot reaches today. The order
    # matters — we assert phases appear sequentially in stdout.
    expected_phases = [
        "Game mode indeterminate.",
        "Public DOOM - v1.10",
        "V_Init: allocate screens.",
        "M_LoadDefaults: Load system defaults.",
        "Z_Init: Init zone memory allocation daemon.",
        "W_Init: Init WADfiles.",
    ]
    cursor = 0
    for phase in expected_phases:
        idx = res.stdout.find(phase, cursor)
        assert idx != -1, (
            f"DOOM boot regressed: phase {phase!r} missing or "
            f"out-of-order. tail = {res.stdout[-400:]!r}"
        )
        cursor = idx + len(phase)

    # Without a WAD, DOOM's W_InitFiles aborts via I_Error — exit 1.
    # If this changes (e.g. exit code propagation gets wired up
    # differently), update this expectation alongside the boot trace.
    assert res.exit_code == 1, (
        f"unexpected exit code: {res.exit_code} "
        f"(expected 1 for no-WAD path)"
    )
