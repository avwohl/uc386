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


def test_micropython_enumerate_filter_property(micropython_bin: Path) -> None:
    """`enumerate`, `filter`, and `@property` are the next batch of
    CORE_FEATURES-gated builtins the port opts into selectively (
    MICROPY_PY_BUILTINS_ENUMERATE / FILTER / PROPERTY in
    mpconfigport.h, while staying at ROM_LEVEL_MINIMUM). Each pulls
    in self-contained .c that's already in the multi-TU compile."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from uc386.dos_emu import run

    res = run(micropython_bin,
              stdin_bytes=(
                  b"print(list(enumerate(['a','b','c'])))\n"
                  b"print(list(filter(lambda x: x>2, [1,2,3,4])))\n"
                  b"\x04"
              ),
              timeout_seconds=15.0,
              instruction_limit=4_000_000_000)
    assert not res.timed_out, "REPL didn't exit"
    assert res.error is None, f"dos_emu reported error: {res.error}"
    assert res.exit_code == 0
    assert "[(0, 'a'), (1, 'b'), (2, 'c')]" in res.stdout, (
        f"expected enumerate output, got: {res.stdout!r}"
    )
    assert "[3, 4]" in res.stdout, (
        f"expected filter output `[3, 4]`, got: {res.stdout!r}"
    )


def test_micropython_property_decorator(micropython_bin: Path) -> None:
    """`@property` decorator wraps an instance method as a read-only
    attribute. Pins the descriptor protocol + decorator dispatch."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from uc386.dos_emu import run

    res = run(
        micropython_bin,
        stdin_bytes=(
            b"class C:\n"
            b"    @property\n"
            b"    def x(self): return 42\n"
            b"\n"
            b"print(C().x)\n"
            b"\x04"
        ),
        timeout_seconds=15.0,
        instruction_limit=4_000_000_000,
    )
    assert not res.timed_out, "REPL didn't exit"
    assert res.error is None, f"dos_emu reported error: {res.error}"
    assert res.exit_code == 0
    assert "\n42\n" in res.stdout, (
        f"expected `C().x` → `42`, got: {res.stdout!r}"
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


def test_micropython_core_features_bytearray(micropython_bin: Path) -> None:
    """`bytearray(b'abc')` is gated at CORE_FEATURES (default-off at
    MINIMUM via MICROPY_PY_BUILTINS_BYTEARRAY). Pins the type as
    runnable end-to-end on the uc386-built bin."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from uc386.dos_emu import run

    res = run(micropython_bin,
              stdin_bytes=b"print(bytearray(b'abc'))\n\x04",
              timeout_seconds=15.0,
              instruction_limit=4_000_000_000)
    assert not res.timed_out, "REPL didn't exit"
    assert res.error is None, f"dos_emu reported error: {res.error}"
    assert res.exit_code == 0
    assert "bytearray(b'abc')" in res.stdout, (
        f"expected bytearray repr, got: {res.stdout!r}"
    )


def test_micropython_core_features_set(micropython_bin: Path) -> None:
    """`set` literal + binary `|` exercises CORE_FEATURES gates
    (MICROPY_PY_BUILTINS_SET). The dedup output `{1, 2, 3}` from
    `set([1,2,2,3])` proves objset.c's hash-based de-duplication is
    wired correctly across the multi-TU build."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from uc386.dos_emu import run

    res = run(micropython_bin,
              stdin_bytes=b"print(set([1,2,2,3]))\n\x04",
              timeout_seconds=15.0,
              instruction_limit=4_000_000_000)
    assert not res.timed_out, "REPL didn't exit"
    assert res.error is None, f"dos_emu reported error: {res.error}"
    assert res.exit_code == 0
    assert "{1, 2, 3}" in res.stdout, (
        f"expected set output `{{1, 2, 3}}`, got: {res.stdout!r}"
    )


def test_micropython_core_features_named_error(micropython_bin: Path) -> None:
    """At CORE_FEATURES `MICROPY_ERROR_REPORTING_DETAILED` includes
    the offending qstr name in NameError messages (vs MINIMUM's
    `name not defined` placeholder). Pin the rich form so a future
    ROM-level downgrade is caught loudly."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from uc386.dos_emu import run

    res = run(micropython_bin,
              stdin_bytes=b"print(undefined_name)\n\x04",
              timeout_seconds=15.0,
              instruction_limit=4_000_000_000)
    assert not res.timed_out, "REPL didn't exit"
    assert res.error is None, f"dos_emu reported error: {res.error}"
    assert res.exit_code == 0
    assert "undefined_name" in res.stdout, (
        f"expected qstr name in NameError, got: {res.stdout!r}"
    )
    assert "NameError" in res.stdout, (
        f"expected NameError, got: {res.stdout!r}"
    )


def test_micropython_core_features_str_modulo(micropython_bin: Path) -> None:
    """C-style `%` string formatting (`'%d-%s' % (5, 'x')`) is gated
    on `MICROPY_PY_BUILTINS_STR_OP_MODULO` which default-enables at
    CORE_FEATURES. Pins the formatter end-to-end."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from uc386.dos_emu import run

    res = run(micropython_bin,
              stdin_bytes=b"print('%d-%s' % (5, 'x'))\n\x04",
              timeout_seconds=15.0,
              instruction_limit=4_000_000_000)
    assert not res.timed_out, "REPL didn't exit"
    assert res.error is None, f"dos_emu reported error: {res.error}"
    assert res.exit_code == 0
    assert "5-x" in res.stdout, (
        f"expected `5-x` from str %% formatting, got: {res.stdout!r}"
    )


def test_micropython_import_sys(micropython_bin: Path) -> None:
    """`import sys` exercises the module-table lookup path. Until
    moduledefs.h registered `mp_module_sys`, this raised
    `ImportError: no module named 'sys'`. Pins both the registration
    + the sys module's static-init."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from uc386.dos_emu import run

    res = run(micropython_bin,
              stdin_bytes=b"import sys\nprint(sys.implementation.name)\n\x04",
              timeout_seconds=15.0,
              instruction_limit=4_000_000_000)
    assert not res.timed_out, "REPL didn't exit"
    assert res.error is None, f"dos_emu reported error: {res.error}"
    assert res.exit_code == 0
    assert "micropython" in res.stdout, (
        f"expected `micropython` from sys.implementation.name, got: "
        f"{res.stdout!r}"
    )


def test_micropython_import_gc(micropython_bin: Path) -> None:
    """`import gc; gc.collect()` exercises the gc module + its
    `gc.collect` entry. Gated on `MICROPY_PY_GC` which default-on
    at CORE_FEATURES."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from uc386.dos_emu import run

    res = run(micropython_bin,
              stdin_bytes=b"import gc\ngc.collect()\nprint('ok')\n\x04",
              timeout_seconds=15.0,
              instruction_limit=4_000_000_000)
    assert not res.timed_out, "REPL didn't exit"
    assert res.error is None, f"dos_emu reported error: {res.error}"
    assert res.exit_code == 0
    assert "ok" in res.stdout, (
        f"expected `ok` after gc.collect(), got: {res.stdout!r}"
    )


def test_micropython_import_collections(micropython_bin: Path) -> None:
    """`import collections; OrderedDict` exercises the collections
    module — gated on `MICROPY_PY_COLLECTIONS` (CORE_FEATURES)
    and a uc386-side moduledefs.h registration."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from uc386.dos_emu import run

    res = run(micropython_bin,
              stdin_bytes=(
                  b"from collections import OrderedDict\n"
                  b"d = OrderedDict()\n"
                  b"d['a'] = 1\n"
                  b"d['b'] = 2\n"
                  b"print(list(d.keys()))\n"
                  b"\x04"
              ),
              timeout_seconds=15.0,
              instruction_limit=4_000_000_000)
    assert not res.timed_out, "REPL didn't exit"
    assert res.error is None, f"dos_emu reported error: {res.error}"
    assert res.exit_code == 0
    assert "['a', 'b']" in res.stdout, (
        f"expected ordered keys `['a', 'b']`, got: {res.stdout!r}"
    )


def test_micropython_import_struct(micropython_bin: Path) -> None:
    """`import struct; struct.pack` exercises struct module — gated
    on `MICROPY_PY_STRUCT` (CORE_FEATURES). Pinned because struct's
    little-endian byte layout is sensitive to misaligned codegen."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from uc386.dos_emu import run

    res = run(micropython_bin,
              stdin_bytes=(
                  b"import struct\n"
                  b"print(struct.pack('<I', 0x12345678))\n"
                  b"\x04"
              ),
              timeout_seconds=15.0,
              instruction_limit=4_000_000_000)
    assert not res.timed_out, "REPL didn't exit"
    assert res.error is None, f"dos_emu reported error: {res.error}"
    assert res.exit_code == 0
    # 0x12345678 little-endian = 78 56 34 12
    assert "b'xV4\\x12'" in res.stdout, (
        f"expected packed bytes `b'xV4\\\\x12'`, got: {res.stdout!r}"
    )


def test_micropython_import_errno(micropython_bin: Path) -> None:
    """`import errno; errno.EINVAL` exercises the errno module —
    pulled in via explicit `MICROPY_PY_ERRNO=1` opt-in. The X-macro
    qstrs (`MP_QSTR_EPERM`, `MP_QSTR_EINVAL`, etc.) are pre-emitted
    into qstrdefs.generated.h by build.sh's X-macro-aware grep —
    without that, uc386 fails compile with `float init must be a
    constant expression (got Identifier)` because the qstrs aren't
    enum constants the static-init can resolve. Pin so a future
    config change doesn't silently drop the surface."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from uc386.dos_emu import run

    res = run(micropython_bin,
              stdin_bytes=b"import errno\nprint(errno.EINVAL)\n\x04",
              timeout_seconds=15.0,
              instruction_limit=4_000_000_000)
    assert not res.timed_out, "REPL didn't exit"
    assert res.error is None, f"dos_emu reported error: {res.error}"
    assert res.exit_code == 0
    assert "22" in res.stdout, (
        f"expected EINVAL=22 from errno module, got: {res.stdout!r}"
    )


def test_micropython_import_math(micropython_bin: Path) -> None:
    """`import math; math.sqrt(2.0)` exercises the math module —
    requires `MICROPY_FLOAT_IMPL=DOUBLE` (so MicroPython's mp_float_t
    is a real double, lowered through uc386's x87 FPU path) plus
    libc-side implementations of the math functions modmath.c
    references at CORE_FEATURES (sin/cos/tan/asin/acos/atan/atan2/
    exp/log/pow/sqrt/floor/ceil/trunc/fmod/copysign/fabs/ldexp/
    nearbyint, plus modf for repr).

    Also pin the constant pi (gated on MICROPY_PY_MATH_CONSTANTS at
    EXTRA_FEATURES — we don't have it, so this test only checks
    sqrt). Result `1.414...` proves the FPU path works end-to-end."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from uc386.dos_emu import run

    res = run(micropython_bin,
              stdin_bytes=b"import math\nprint(math.sqrt(2.0))\n\x04",
              timeout_seconds=15.0,
              instruction_limit=4_000_000_000)
    assert not res.timed_out, "REPL didn't exit"
    assert res.error is None, f"dos_emu reported error: {res.error}"
    assert res.exit_code == 0
    assert "1.41421" in res.stdout, (
        f"expected sqrt(2) ≈ 1.41421..., got: {res.stdout!r}"
    )


def test_micropython_float_arithmetic(micropython_bin: Path) -> None:
    """`1.5 + 2.5` exercises the FPU path end-to-end inside the REPL:
    lex → parse FloatLiteral → BINARY_OP + → mp_float_t add → repr
    → print. Without MICROPY_PY_BUILTINS_FLOAT (i.e. before the
    FLOAT_IMPL=DOUBLE bump) this raised SyntaxError on the dot.

    Tolerates upstream's APPROX float formatter quirk: at
    MICROPY_FLOAT_IMPL=DOUBLE without a wider-than-double
    `mp_large_float_t`, the formatter accumulates rounding error
    across digit-extraction multiplies and `4.0` repr-prints as
    `3.999999999999997` instead. We assert the parsed value is
    close to 4.0, not the exact literal."""
    import re
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from uc386.dos_emu import run

    res = run(micropython_bin,
              stdin_bytes=b"print(1.5 + 2.5)\n\x04",
              timeout_seconds=15.0,
              instruction_limit=4_000_000_000)
    assert not res.timed_out, "REPL didn't exit"
    assert res.error is None, f"dos_emu reported error: {res.error}"
    assert res.exit_code == 0
    # Match the printed numeric line between the two `>>> ` prompts.
    m = re.search(r"\n([0-9.]+)\n", res.stdout)
    assert m is not None, (
        f"expected a numeric result line, got: {res.stdout!r}"
    )
    val = float(m.group(1))
    assert abs(val - 4.0) < 1e-6, (
        f"expected ≈ 4.0, got {val!r} from stdout {res.stdout!r}"
    )


def test_micropython_math_special_functions(micropython_bin: Path) -> None:
    """`MICROPY_PY_MATH_SPECIAL_FUNCTIONS` opens hyperbolic +
    log2 + expm1 + erf/erfc surface. Pin a small subset
    (sinh/cosh ≈ identity at 0, log2(8) = 3, atanh(0.5) ≈ 0.549,
    erf(0.5) ≈ 0.520) so a future libc-side regression in any of
    the FPU primitives shows up here."""
    import math as cmath_local
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from uc386.dos_emu import run

    res = run(micropython_bin,
              stdin_bytes=(
                  b"import math\n"
                  b"print(round(math.sinh(0.0), 4))\n"
                  b"print(round(math.cosh(0.0), 4))\n"
                  b"print(round(math.log2(8.0), 4))\n"
                  b"print(round(math.atanh(0.5), 4))\n"
                  b"print(round(math.erf(0.5), 4))\n"
                  b"\x04"
              ),
              timeout_seconds=15.0,
              instruction_limit=4_000_000_000)
    assert not res.timed_out, "REPL didn't exit"
    assert res.error is None, f"dos_emu reported error: {res.error}"
    assert res.exit_code == 0
    # Each result is rounded to 4 decimal digits in-Python before
    # printing, dodging the formatter-precision quirk for sub-1
    # results (those don't need MAX_MANTISSA_DIGITS=19 anyway).
    expected = [
        ("sinh(0)", 0.0),
        ("cosh(0)", 1.0),
        ("log2(8)", 3.0),
        ("atanh(0.5)", round(cmath_local.atanh(0.5), 4)),
        ("erf(0.5)", round(cmath_local.erf(0.5), 4)),
    ]
    out_lines = [l for l in res.stdout.splitlines() if l and l[0].isdigit() or (l and l[0] == '-')]
    assert len(out_lines) >= 5, (
        f"expected 5 numeric lines, got: {res.stdout!r}"
    )
    for (label, exp), actual_str in zip(expected, out_lines):
        actual = float(actual_str)
        # Abramowitz erf is good to ~1.5e-7; round-to-4 leaves
        # plenty of room for that.
        assert abs(actual - exp) < 1e-3, (
            f"{label}: expected ≈ {exp}, got {actual}"
        )


def test_micropython_import_array(micropython_bin: Path) -> None:
    """`import array; array.array('i', ...)` — gated on
    `MICROPY_PY_ARRAY` (CORE_FEATURES). Pins typed-array storage
    + iteration."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from uc386.dos_emu import run

    res = run(micropython_bin,
              stdin_bytes=(
                  b"import array\n"
                  b"a = array.array('i', [1,2,3])\n"
                  b"print(sum(a))\n"
                  b"\x04"
              ),
              timeout_seconds=15.0,
              instruction_limit=4_000_000_000)
    assert not res.timed_out, "REPL didn't exit"
    assert res.error is None, f"dos_emu reported error: {res.error}"
    assert res.exit_code == 0
    assert "\n6\n" in res.stdout, (
        f"expected sum 6 from int array, got: {res.stdout!r}"
    )
