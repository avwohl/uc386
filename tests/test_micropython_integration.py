"""End-to-end MicroPython smoke test.

The MicroPython port lives in its own package now —
[`freedos_micro_python`](https://github.com/avwohl/freedos_micro_python).
This test pulls a built `micropython.bin` (built earlier via
`freedos-micropython port`) and runs it under `uc386.dos_emu` to
verify uc386's compiler + emulator still get the REPL banner out.

Why keep this in uc386's tests/: MicroPython is the most demanding
end-to-end exercise we have for the compiler. A regression that the
in-tree c-testsuite misses tends to surface here. The test is a
skip-if-not-built guard so it doesn't slow normal pytest runs.

To enable:

    pip install freedos_micro_python
    mkdir /tmp/mp-build && cd /tmp/mp-build
    freedos-micropython fetch
    freedos-micropython build
    freedos-micropython port      # ~14 min
    FREEDOS_MP_BIN=$(pwd)/build/micropython.bin pytest \\
        /path/to/uc386/tests/test_micropython_integration.py
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from uc386.dos_emu import run


def _find_bin() -> Path | None:
    env = os.environ.get("FREEDOS_MP_BIN")
    if env:
        p = Path(env)
        return p if p.exists() else None
    cwd_bin = Path.cwd() / "build" / "micropython.bin"
    return cwd_bin if cwd_bin.exists() else None


@pytest.fixture(scope="module")
def micropython_bin() -> Path:
    p = _find_bin()
    if p is None:
        pytest.skip(
            "micropython.bin not built. Install freedos_micro_python, "
            "run `freedos-micropython port` in a workdir, then point "
            "FREEDOS_MP_BIN at the resulting build/micropython.bin."
        )
    return p


def test_micropython_repl_banner(micropython_bin: Path) -> None:
    """The REPL banner ('MicroPython ...; uc386-dos with i386') must
    appear on stdout. The REPL waits on stdin and dos_emu hits its
    timeout — that's the steady state we expect."""
    res = run(micropython_bin,
              timeout_seconds=10.0,
              instruction_limit=2_000_000_000)
    assert res.error is None, f"dos_emu reported error: {res.error}"
    assert "MicroPython" in res.stdout, (
        f"REPL banner missing from stdout: {res.stdout!r}"
    )
    assert "uc386-dos with i386" in res.stdout, (
        f"port banner ('uc386-dos with i386') missing: {res.stdout!r}"
    )
