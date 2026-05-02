#!/usr/bin/env python3
"""Smoke test for the uc386-built BWK awk binary.

Runs `addons/gnu/awk-bwk/build/awk.bin` under uc386.dos_emu against
a handful of representative scripts and asserts each produces the
expected output. Exercises the pieces of awk that exercise the
broadest uc386 codegen surface: BEGIN/END blocks, regex matching,
field access ($0, $1, NR), aggregation in associative arrays,
built-in string functions.

Skipped when `awk.bin` doesn't exist — the binary is a derived
artifact (built by `addons/gnu/awk-bwk/build.sh` after
`fetch.sh`); we don't check it in. Run after a build to guard
against regressions in either the codegen pipeline or BWK awk's
behavior.

Usage:
    .venv/bin/pytest addons/gnu/awk-bwk/test_awk_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

_HERE = Path(__file__).resolve().parent
_BIN = _HERE / "build" / "awk.bin"


@pytest.fixture(scope="module")
def awk_bin() -> Path:
    if not _BIN.exists():
        pytest.skip(
            "awk.bin not built — run "
            "addons/gnu/awk-bwk/{fetch,build}.sh first"
        )
    return _BIN


def _run_awk(awk_bin: Path, script: str, stdin: bytes = b""):
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from uc386.dos_emu import run

    return run(
        awk_bin,
        argv=["awk", script],
        stdin_bytes=stdin,
        timeout_seconds=15.0,
        instruction_limit=400_000_000,
    )


def test_awk_begin_block_arithmetic(awk_bin: Path) -> None:
    """`BEGIN { print 2*3 }` exercises the BEGIN-pattern path: no
    input read, just program-start action. Pins the lexer, parser,
    and integer-print path."""
    res = _run_awk(awk_bin, "BEGIN { print 2*3 }")
    assert res.error is None, f"dos_emu reported error: {res.error}"
    assert res.exit_code == 0
    assert res.stdout == "6\n", (
        f"expected `6\\n`, got: {res.stdout!r}"
    )


def test_awk_field_access_and_nr(awk_bin: Path) -> None:
    """`{ print NR, $0 }` exercises field access ($0 = full line),
    the auto-increment of NR, and the implicit per-record action
    loop. Pins the field-split + record-loop machinery."""
    res = _run_awk(
        awk_bin, "{ print NR, $0 }",
        stdin=b"one\ntwo\nthree\n",
    )
    assert res.error is None, f"dos_emu reported error: {res.error}"
    assert res.exit_code == 0
    assert res.stdout == "1 one\n2 two\n3 three\n", (
        f"expected numbered lines, got: {res.stdout!r}"
    )


def test_awk_regex_pattern(awk_bin: Path) -> None:
    """`/foo/` is a bare regex pattern with implicit `print $0`
    action. Lines matching the pattern are emitted; non-matching
    lines are dropped. Pins the regex engine + pattern dispatch."""
    res = _run_awk(
        awk_bin, "/foo/",
        stdin=b"one\nfoo\nbar\nfoobar\n",
    )
    assert res.error is None, f"dos_emu reported error: {res.error}"
    assert res.exit_code == 0
    assert res.stdout == "foo\nfoobar\n", (
        f"expected only foo-matching lines, got: {res.stdout!r}"
    )


def test_awk_aggregation(awk_bin: Path) -> None:
    """`{ s += $1 } END { print s }` — accumulator pattern. Tests
    persistent variable across records and the END block firing
    once after input is exhausted."""
    res = _run_awk(
        awk_bin, "{ s += $1 } END { print s }",
        stdin=b"10\n20\n30\n",
    )
    assert res.error is None, f"dos_emu reported error: {res.error}"
    assert res.exit_code == 0
    assert res.stdout == "60\n", (
        f"expected sum `60\\n`, got: {res.stdout!r}"
    )


def test_awk_string_function(awk_bin: Path) -> None:
    """`{ print toupper($0) }` exercises a built-in string function
    (toupper) on each record. Pins the string-builtin dispatch and
    the alloc/free of intermediate string values."""
    res = _run_awk(
        awk_bin, "{ print toupper($0) }",
        stdin=b"hello\nworld\n",
    )
    assert res.error is None, f"dos_emu reported error: {res.error}"
    assert res.exit_code == 0
    assert res.stdout == "HELLO\nWORLD\n", (
        f"expected uppercased output, got: {res.stdout!r}"
    )
