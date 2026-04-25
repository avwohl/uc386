"""Smoke tests: verify the uc_core -> uc386 pipeline is wired up."""

import subprocess
import sys

import pytest

from uc_core.backend import CodeGenerator as CodeGeneratorProtocol
from uc_core.lexer import Lexer
from uc_core.parser import Parser

from uc386.codegen import CodeGenerator, CodegenError


def _compile(src: str) -> str:
    tokens = list(Lexer(src, "test.c").tokenize())
    unit = Parser(tokens).parse()
    return CodeGenerator().generate(unit)


def test_backend_implements_protocol():
    assert isinstance(CodeGenerator(), CodeGeneratorProtocol)


def test_emits_dos_exit_stub():
    asm = _compile("int main(void) { return 0; }")
    assert "bits 32" in asm
    assert "_start:" in asm
    assert "_main:" in asm
    assert "call    _main" in asm
    assert "int     21h" in asm
    assert "mov     ah, 4Ch" in asm


def test_return_value_passed_in_eax():
    asm = _compile("int main(void) { return 42; }")
    assert "mov     eax, 42" in asm


def test_empty_main_falls_through_to_zero():
    asm = _compile("int main(void) { }")
    # C99: falling off main returns 0. We emit xor eax,eax before the epilogue.
    assert "xor     eax, eax" in asm


def test_bare_return_is_zero():
    asm = _compile("int main(void) { return; }")
    assert "xor     eax, eax" in asm


def test_missing_main_rejected():
    with pytest.raises(CodegenError, match="main"):
        _compile("int other(void) { return 0; }")


def test_arithmetic_return_rejected():
    # Binary ops aren't lowered yet.
    with pytest.raises(CodegenError):
        _compile("int main(void) { return 1 + 2; }")


def test_local_int_with_literal_init():
    asm = _compile("int main(void) { int x = 7; return x; }")
    assert "sub     esp, 4" in asm
    assert "mov     eax, 7" in asm
    assert "mov     [ebp - 4], eax" in asm
    assert "mov     eax, [ebp - 4]" in asm


def test_multiple_locals_get_distinct_slots():
    asm = _compile("int main(void) { int x = 1; int y = 2; return y; }")
    assert "sub     esp, 8" in asm
    assert "mov     [ebp - 4], eax" in asm
    assert "mov     [ebp - 8], eax" in asm
    # Return reads y, which is at -8.
    assert "mov     eax, [ebp - 8]\n        jmp     .epilogue" in asm


def test_uninitialized_local_allocated_no_store():
    asm = _compile("int main(void) { int x; return 0; }")
    # Frame is reserved but no init store is emitted.
    assert "sub     esp, 4" in asm
    assert "mov     [ebp - 4], eax" not in asm


def test_unknown_identifier_rejected():
    with pytest.raises(CodegenError, match="unknown identifier"):
        _compile("int main(void) { return x; }")


def test_non_int_local_rejected():
    with pytest.raises(CodegenError, match="only `int`"):
        _compile("int main(void) { char c; return 0; }")


def test_end_to_end_driver(tmp_path):
    src = tmp_path / "hi.c"
    src.write_text("int main(void) { return 0; }\n")
    out = tmp_path / "hi.asm"
    r = subprocess.run(
        [sys.executable, "-m", "uc386.main", str(src), "-o", str(out)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    text = out.read_text()
    assert "_start:" in text
    assert "_main:" in text
    assert "int     21h" in text
