"""Harness selector — picks the right ``run`` implementation based
on the ``UC386_HARNESS`` env var:

- ``UC386_HARNESS=dosiz`` → dispatches to ``uc386.dosiz_run.run``
  (in-process dosbox-staging with full DPMI 0.9). Use when the
  unicorn-based dos_emu trips a quirk in the program under test.
- anything else (default) → ``uc386.dos_emu.run`` (unicorn-based).

Tests should ``from uc386.harness import run`` so the toggle works
without per-test edits.
"""
from __future__ import annotations

import os


def run(*args, **kwargs):
    if os.environ.get("UC386_HARNESS") == "dosiz":
        from uc386.dosiz_run import run as _dosiz_run
        return _dosiz_run(*args, **kwargs)
    from uc386.dos_emu import run as _dos_emu_run
    return _dos_emu_run(*args, **kwargs)
