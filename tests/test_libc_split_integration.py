"""libc_split↔uc386 integration tests.

End-to-end tests that exercise upeep386's libc_split parser against
uc386's actual `lib/i386_dos_libc.asm` file. The pure parser tests
live in upeep386's own test suite — what lives here is the
verification that the bundled libc parses cleanly and the typical
closures (just `_printf`, just `_abort`) produce sensible output.

Extracted from uc386/tests/test_libc_split.py when libc_split.py
was moved to the upeep386 package (2026-05-23).
"""

from pathlib import Path

import pytest

from upeep386.libc_split import parse_libc


LIBC_PATH = Path(__file__).resolve().parents[1] / "src" / "uc386" / "lib" / "i386_dos_libc.asm"


@pytest.fixture(scope="module")
def real_libc():
    return parse_libc(LIBC_PATH.read_text())


def test_real_libc_parses_without_error(real_libc):
    """The bundled libc itself should parse cleanly."""
    assert len(real_libc.functions) > 50
    for name in ["_printf", "_abort", "_putchar", "_malloc", "_free", "_strlen"]:
        assert name in real_libc.functions, f"{name} missing from parsed libc"


def test_real_libc_printf_minimal_deps(real_libc):
    needed = real_libc.transitive_closure({"_printf"})
    assert "_printf" in needed


def test_real_libc_abort_no_deps(real_libc):
    """`_abort` is a leaf — `int 0x21` exits via the harness."""
    needed = real_libc.transitive_closure({"_abort"})
    assert needed == {"_abort"}


def test_real_libc_emit_minimal_size(real_libc):
    """Emitting just printf produces dramatically less asm than the
    full libc."""
    out = real_libc.emit({"_printf"})
    full_lines = sum(1 for _ in LIBC_PATH.read_text().splitlines())
    out_lines = len(out.splitlines())
    assert out_lines < full_lines * 0.05  # < 5% of original
