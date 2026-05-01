#!/usr/bin/env python3
"""Smoke test for the uc386-built MicroPython binary.

Runs `addons/gnu/micropython/build/micropython.bin` under
`uc386.dos_emu` and asserts it boots far enough to print the REPL
banner and the `>>> ` prompt.

Skipped when `micropython.bin` doesn't exist — the binary is a
derived artifact (built by `addons/gnu/micropython/build_port.sh`,
~14 min wall-clock); we don't check it in. Run after a port build
to guard against regressions.

The REPL banner format is fixed by `shared/runtime/pyexec.c`'s
`pyexec_friendly_repl_print_banner`:

    MicroPython <git-tag> on <build-date>; <hw-board> with <hw-mcu>
    Type "help()" for more information.
    >>>

We assert the prefix only (the date varies per build).

Usage:
    .venv/bin/pytest addons/gnu/micropython/test_micropython_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

_HERE = Path(__file__).resolve().parent
_BIN = _HERE / "build" / "micropython.bin"


@pytest.fixture(scope="module")
def micropython_bin() -> Path:
    if not _BIN.exists():
        pytest.skip(
            "micropython.bin not built — run "
            "addons/gnu/micropython/build_port.sh first (~14 min)"
        )
    return _BIN


def test_micropython_repl_banner(micropython_bin: Path) -> None:
    """MicroPython should boot far enough to print the REPL banner
    and the `>>> ` prompt, then wait on stdin."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from uc386.dos_emu import run

    res = run(micropython_bin,
              timeout_seconds=10.0,
              instruction_limit=2_000_000_000)
    # The REPL waits on stdin and dos_emu hits its timeout — that's
    # the expected steady state. error=None means no faults during
    # boot.
    assert res.error is None, f"dos_emu reported error: {res.error}"
    assert "MicroPython" in res.stdout, (
        f"REPL banner missing from stdout: {res.stdout!r}"
    )
    assert "uc386-dos with i386" in res.stdout, (
        f"port banner ('uc386-dos with i386') missing: "
        f"{res.stdout!r}"
    )
    assert ">>> " in res.stdout, (
        f"REPL prompt ('>>> ') missing: {res.stdout!r}"
    )


def test_micropython_clean_eof_exit(micropython_bin: Path) -> None:
    """Sending only Ctrl-D (EOF) at the prompt should exit the REPL
    cleanly with exit code 0 — exercises the readline → pyexec EOF
    path that runs after the boot banner."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from uc386.dos_emu import run

    res = run(micropython_bin,
              stdin_bytes=b"\x04",
              timeout_seconds=15.0,
              instruction_limit=2_000_000_000)
    assert not res.timed_out, "REPL didn't exit on Ctrl-D"
    assert res.error is None, f"dos_emu reported error: {res.error}"
    assert res.exit_code == 0, (
        f"unexpected exit code: {res.exit_code} "
        f"(expected 0 for clean Ctrl-D exit)"
    )
    assert ">>> " in res.stdout, (
        f"REPL prompt missing before Ctrl-D was processed: "
        f"{res.stdout!r}"
    )
