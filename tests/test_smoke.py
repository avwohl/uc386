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


def test_arithmetic_returns_emit_two_loads_and_op():
    asm = _compile("int main(void) { int x = 1; int y = 2; return x + y; }")
    # x → eax → push, y → eax → ecx, pop eax, add eax,ecx
    assert "push    eax" in asm
    assert "pop     eax" in asm
    assert "add     eax, ecx" in asm


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


@pytest.mark.parametrize(
    "op,instr",
    [
        ("+",  "add     eax, ecx"),
        ("-",  "sub     eax, ecx"),
        ("*",  "imul    eax, ecx"),
        ("&",  "and     eax, ecx"),
        ("|",  "or      eax, ecx"),
        ("^",  "xor     eax, ecx"),
    ],
)
def test_simple_binops(op, instr):
    asm = _compile(f"int main(void) {{ int x = 1; int y = 2; return x {op} y; }}")
    assert instr in asm


def test_division_uses_idiv():
    asm = _compile("int main(void) { int x = 10; int y = 3; return x / y; }")
    assert "cdq" in asm
    assert "idiv    ecx" in asm


def test_modulo_takes_remainder_from_edx():
    asm = _compile("int main(void) { int x = 10; int y = 3; return x % y; }")
    assert "idiv    ecx" in asm
    assert "mov     eax, edx" in asm


def test_left_shift_uses_cl():
    asm = _compile("int main(void) { int x = 1; int y = 2; return x << y; }")
    assert "shl     eax, cl" in asm


def test_signed_right_shift_is_arithmetic():
    asm = _compile("int main(void) { int x = 8; int y = 1; return x >> y; }")
    assert "sar     eax, cl" in asm


@pytest.mark.parametrize(
    "op,setcc",
    [
        ("==", "sete"),
        ("!=", "setne"),
        ("<",  "setl"),
        (">",  "setg"),
        ("<=", "setle"),
        (">=", "setge"),
    ],
)
def test_comparisons(op, setcc):
    asm = _compile(f"int main(void) {{ int x = 1; int y = 2; return x {op} y; }}")
    assert "cmp     eax, ecx" in asm
    assert f"{setcc}    al" in asm
    assert "movzx   eax, al" in asm


def test_unary_minus_emits_neg():
    asm = _compile("int main(void) { int x = 5; return -x; }")
    assert "neg     eax" in asm


def test_unary_bitnot_emits_not():
    asm = _compile("int main(void) { int x = 5; return ~x; }")
    assert "not     eax" in asm


def test_logical_not_emits_test_sete():
    asm = _compile("int main(void) { int x = 5; return !x; }")
    assert "test    eax, eax" in asm
    assert "sete    al" in asm


def test_assignment_writes_back_to_slot():
    asm = _compile("int main(void) { int x = 1; x = 5; return x; }")
    # The assignment becomes a `mov eax, 5` followed by a write to [ebp-4].
    # The earlier init also writes [ebp-4]; both must be present.
    assert asm.count("mov     [ebp - 4], eax") >= 2


def test_assignment_to_non_lvalue_rejected():
    with pytest.raises(CodegenError, match="must be an identifier"):
        _compile("int main(void) { 1 = 2; return 0; }")


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
