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

# Two locations: the dev-tree path and the FOSS-tarball path.
#
# Dev tree: addons/gnu/micropython/test_micropython_smoke.py
#           → addons/gnu/micropython/build/micropython.bin
#           = parent / build / micropython.bin
# Tarball:  uc386-foss/src/micropython/test_micropython_smoke.py
#           → uc386-foss/micropython.bin
#           = parent.parent / micropython.bin (src/<name>/.. = src/.. = uc386-foss/)
_HERE = Path(__file__).resolve().parent
_BIN_CANDIDATES = [
    _HERE / "build" / "micropython.bin",
    _HERE.parent.parent / "micropython.bin",
]


def _find_bin() -> Path | None:
    for p in _BIN_CANDIDATES:
        if p.exists():
            return p
    return None


@pytest.fixture(scope="module")
def micropython_bin() -> Path:
    p = _find_bin()
    if p is None:
        pytest.skip(
            "micropython.bin not built — run "
            "addons/gnu/micropython/build_port.sh first (~14 min)"
        )
    return p


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


def test_micropython_named_builtin(micropython_bin: Path) -> None:
    """`__name__\\n\\x04` exercises the static-qstr LOAD_GLOBAL path:
    mp_init's dict_main store of `__name__` (qstr id 67) had to
    survive the qstr_find_strn binary search through all 878
    main-pool entries. Required: LC_ALL=C ASCII collation in the
    qstrdefs sort + real strlen in the QDEF length field. Result
    should print the module name `'__main__'`."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from uc386.dos_emu import run

    res = run(micropython_bin,
              stdin_bytes=b"__name__\n\x04",
              timeout_seconds=15.0,
              instruction_limit=4_000_000_000)
    assert not res.timed_out, "REPL didn't exit"
    assert res.error is None, f"dos_emu reported error: {res.error}"
    assert res.exit_code == 0
    assert "'__main__'" in res.stdout, (
        f"expected `'__main__'` in stdout, got: {res.stdout!r}"
    )


def test_micropython_arithmetic(micropython_bin: Path) -> None:
    """`2+3\\n\\x04` exercises the value-print path: lex → parse →
    compile → VM dispatch (LOAD_CONST 2; LOAD_CONST 3; BINARY_OP +;
    __repl_print__ wrap) → mp_load_global → builtins dict lookup →
    print result. Until commit 19ae598 this trapped in
    `_pass_push_memory_to_push_reg`, which was incorrectly merging
    chained pointer dereferences (`mov eax, [eax+4]; push [eax+4]`
    → `push eax`)."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from uc386.dos_emu import run

    res = run(micropython_bin,
              stdin_bytes=b"2+3\n\x04",
              timeout_seconds=15.0,
              instruction_limit=4_000_000_000)
    assert not res.timed_out, "REPL didn't exit"
    assert res.error is None, f"dos_emu reported error: {res.error}"
    assert res.exit_code == 0, (
        f"unexpected exit code: {res.exit_code} "
        f"(expected 0 for `2+3\\n\\x04` clean exit)"
    )
    # Result `5` should appear on its own line between the two
    # `>>> ` prompts. Use a substring check so prompt-formatting
    # tweaks don't break the test.
    assert "\n5\n" in res.stdout, (
        f"expected arithmetic result `5` in stdout, got: {res.stdout!r}"
    )


def test_micropython_assign_statement(micropython_bin: Path) -> None:
    """`x = 5\\n\\x04` exercises the qstr-store path that the
    QDEF1 fix unblocked: lexes the identifier, allocates an
    interned qstr, dict-stores 5 against it, then EOF exits.

    Different from `pass` — that one didn't allocate any qstrs.
    Until commit 21dc0d9 this trapped in `qstr_find_strn`'s
    binary search because the empty-main-pool corner case
    underflowed the high-bound."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from uc386.dos_emu import run

    res = run(micropython_bin,
              stdin_bytes=b"x = 5\n\x04",
              timeout_seconds=15.0,
              instruction_limit=4_000_000_000)
    assert not res.timed_out, "REPL didn't exit"
    assert res.error is None, f"dos_emu reported error: {res.error}"
    assert res.exit_code == 0, (
        f"unexpected exit code: {res.exit_code} "
        f"(expected 0 for `x = 5\\n\\x04` clean exit)"
    )
    assert res.stdout.count(">>> ") >= 2, (
        f"expected two `>>> ` prompts in stdout, got: {res.stdout!r}"
    )


def test_micropython_pass_statement(micropython_bin: Path) -> None:
    """Sending `pass\\n\\x04` exercises the full lex → parse → compile
    → exec path: `pass` is a no-op statement, so the REPL shouldn't
    print anything but the next `>>> ` prompt before EOF exits.

    This pins the parser + compiler + VM + NLR (exception machinery)
    after the REPL banner. Expression statements that produce a value
    (`1`, `print(2+3)`) currently fail in the value-print path —
    that's a separate gap, tracked in NOTES.md."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from uc386.dos_emu import run

    res = run(micropython_bin,
              stdin_bytes=b"pass\n\x04",
              timeout_seconds=15.0,
              instruction_limit=4_000_000_000)
    assert not res.timed_out, "REPL didn't exit"
    assert res.error is None, f"dos_emu reported error: {res.error}"
    assert res.exit_code == 0, (
        f"unexpected exit code: {res.exit_code} "
        f"(expected 0 for `pass\\n\\x04` clean exit)"
    )
    # After `pass` executes (no output), REPL prints `>>> ` again,
    # then Ctrl-D exits. Two prompts is the diagnostic.
    assert res.stdout.count(">>> ") >= 2, (
        f"expected two `>>> ` prompts in stdout (one before `pass`, "
        f"one after), got: {res.stdout!r}"
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


def test_micropython_print_real_newline(micropython_bin: Path) -> None:
    """`print(2+3)\\n\\x04` exercises the qstr-payload path: the
    builtin `print()` ends each line with `\\n`, which is qstr id
    `MP_QSTR__0x0a_`. Until commit 2abb610 the qstrdefs generator
    captured the *sanitized* macro tail as the qstr's payload string,
    so `print()` emitted the literal text `_0x0a_` instead of a
    newline. Fix: gen_qstrdefs.py reverses upstream's qstr_escape
    (re-using upstream's codepoint2name map) so escaped qstrs ship
    their original byte string in the QDEF1 payload."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from uc386.dos_emu import run

    res = run(micropython_bin,
              stdin_bytes=b"print(2+3)\n\x04",
              timeout_seconds=15.0,
              instruction_limit=4_000_000_000)
    assert not res.timed_out, "REPL didn't exit"
    assert res.error is None, f"dos_emu reported error: {res.error}"
    assert res.exit_code == 0
    # Result `5` followed by a real `\n`, then the next `>>> ` prompt.
    # The bug rendered this as `5_0x0a_>>>` (concatenated, no newline).
    assert "5\n>>> " in res.stdout, (
        f"expected `5` + real newline + prompt in stdout, got: "
        f"{res.stdout!r}"
    )
    # Belt-and-suspenders: explicitly forbid the pre-fix mangled form.
    assert "_0x0a_" not in res.stdout, (
        f"found mangled `_0x0a_` (qstr_escape not reversed): "
        f"{res.stdout!r}"
    )


def test_micropython_def_and_call(micropython_bin: Path) -> None:
    """Exercise function definition + call: `def f(x): return x*2`,
    then `f(7)` → `14`. Pins compile of FunctionDef and the
    bytecode CALL_FUNCTION → MAKE_FUNCTION → arg binding → return
    path."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from uc386.dos_emu import run

    res = run(micropython_bin,
              stdin_bytes=b"def f(x): return x*2\n\nprint(f(7))\n\x04",
              timeout_seconds=15.0,
              instruction_limit=4_000_000_000)
    assert not res.timed_out, "REPL didn't exit"
    assert res.error is None, f"dos_emu reported error: {res.error}"
    assert res.exit_code == 0
    assert "\n14\n" in res.stdout, (
        f"expected `f(7) = 14` in stdout, got: {res.stdout!r}"
    )


def test_micropython_list_comprehension(micropython_bin: Path) -> None:
    """`[i*i for i in range(5)]` exercises the comprehension scope
    + generator + range iter path, then prints the list. Pins
    objlist + objgenerator + objrange + the BUILD_LIST opcode."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from uc386.dos_emu import run

    res = run(micropython_bin,
              stdin_bytes=b"print([i*i for i in range(5)])\n\x04",
              timeout_seconds=15.0,
              instruction_limit=4_000_000_000)
    assert not res.timed_out, "REPL didn't exit"
    assert res.error is None, f"dos_emu reported error: {res.error}"
    assert res.exit_code == 0
    assert "[0, 1, 4, 9, 16]" in res.stdout, (
        f"expected list-comprehension result in stdout, got: "
        f"{res.stdout!r}"
    )


def test_micropython_bin_hex_oct(micropython_bin: Path) -> None:
    """`bin(10)` → `0b1010`, `hex(255)` → `0xff`, `oct(8)` → `0o10`.
    These exercise qstrs whose escape names CONTAIN underscores
    (`brace_open`, `brace_close`, `colon`, `hash`) — the reverse-
    mangler must not split `_brace_open_` on the inner `_` and
    candidate-match `brace`. Pre-fix output was the macro-mangled
    text: `_brace_open_:#b_brace_close_` instead of `0b1010`."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from uc386.dos_emu import run

    res = run(micropython_bin,
              stdin_bytes=(
                  b"print(bin(10))\n"
                  b"print(hex(255))\n"
                  b"print(oct(8))\n"
                  b"\x04"
              ),
              timeout_seconds=15.0,
              instruction_limit=4_000_000_000)
    assert not res.timed_out, "REPL didn't exit"
    assert res.error is None, f"dos_emu reported error: {res.error}"
    assert res.exit_code == 0
    assert "0b1010" in res.stdout, (
        f"expected `bin(10)` → `0b1010`, got: {res.stdout!r}"
    )
    assert "0xff" in res.stdout, (
        f"expected `hex(255)` → `0xff`, got: {res.stdout!r}"
    )
    assert "0o10" in res.stdout, (
        f"expected `oct(8)` → `0o10`, got: {res.stdout!r}"
    )
    # Belt-and-suspenders against regression: the macro-mangled form
    # `_brace_open_` should never appear in output.
    assert "_brace_open_" not in res.stdout, (
        f"qstr macro-mangling leaked: {res.stdout!r}"
    )


def test_micropython_min_max_reversed(micropython_bin: Path) -> None:
    """`min([3,1,2])`, `max([3,1,2])`, `reversed([1,2,3])` exercise
    the CORE_FEATURES-gated builtins that the port now opts into
    selectively (MICROPY_PY_BUILTINS_MIN_MAX, MICROPY_PY_BUILTINS_REVERSED
    in mpconfigport.h, while staying at ROM_LEVEL_MINIMUM).

    Pinned because they're easy to lose: a future rebuild that
    forgets the opt-ins would silently lose `min`/`max`/`reversed`
    without breaking any other test."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from uc386.dos_emu import run

    res = run(micropython_bin,
              stdin_bytes=(
                  b"print(min([3,1,2]))\n"
                  b"print(max([3,1,2]))\n"
                  b"print(list(reversed([1,2,3])))\n"
                  b"\x04"
              ),
              timeout_seconds=15.0,
              instruction_limit=4_000_000_000)
    assert not res.timed_out, "REPL didn't exit"
    assert res.error is None, f"dos_emu reported error: {res.error}"
    assert res.exit_code == 0
    assert "\n1\n" in res.stdout, (
        f"expected `min` result `1` in stdout, got: {res.stdout!r}"
    )
    assert "\n3\n" in res.stdout, (
        f"expected `max` result `3` in stdout, got: {res.stdout!r}"
    )
    assert "[3, 2, 1]" in res.stdout, (
        f"expected `reversed` result `[3, 2, 1]` in stdout, got: "
        f"{res.stdout!r}"
    )


def test_micropython_try_except(micropython_bin: Path) -> None:
    """`try: 1/0 except ZeroDivisionError: print(\"caught\")`
    exercises the NLR (non-local return) path: the VM raises
    ZeroDivisionError, the SETUP_EXCEPT handler frame catches it,
    matches the type, runs the handler. setjmp-backed NLR
    (MICROPY_NLR_SETJMP=1 in mpconfigport.h) is what makes this
    work — uc386 can't compile nlrx86.c's inline asm."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from uc386.dos_emu import run

    res = run(micropython_bin,
              stdin_bytes=b"try:\n    1/0\nexcept:\n    print('caught')\n\n\x04",
              timeout_seconds=15.0,
              instruction_limit=4_000_000_000)
    assert not res.timed_out, "REPL didn't exit"
    assert res.error is None, f"dos_emu reported error: {res.error}"
    assert res.exit_code == 0
    assert "caught" in res.stdout, (
        f"expected `caught` in stdout (try/except didn't work): "
        f"{res.stdout!r}"
    )
