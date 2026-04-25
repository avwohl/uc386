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


# ---- control flow ----------------------------------------------------------

def test_if_without_else():
    asm = _compile("int main(void) { int x = 0; if (1) { x = 7; } return x; }")
    assert "test    eax, eax" in asm
    assert "_endif:" in asm
    # No else label / jmp-to-endif stitching for else-less if.
    assert "_else:" not in asm


def test_if_else_emits_else_label():
    asm = _compile(
        "int main(void) { int x = 0; if (1) { x = 7; } else { x = 9; } return x; }"
    )
    assert "_else:" in asm
    assert "_endif:" in asm


def test_while_loop_has_top_and_end_labels():
    asm = _compile("int main(void) { int i = 0; while (i < 10) { i = i + 1; } return i; }")
    assert "_while_top:" in asm
    assert "_while_end:" in asm
    # Body jumps back to the top.
    assert "jmp     .L1_while_top" in asm or "jmp     .L2_while_top" in asm \
        or any("jmp     " in line and "_while_top" in line for line in asm.splitlines())


def test_do_while_branches_at_bottom():
    asm = _compile("int main(void) { int i = 0; do { i = i + 1; } while (i < 5); return i; }")
    assert "_do_top:" in asm
    assert "_do_cont:" in asm
    assert "_do_end:" in asm
    assert any("jnz     " in line and "_do_top" in line for line in asm.splitlines())


def test_for_loop_has_init_cond_step_end():
    asm = _compile(
        "int main(void) { int s = 0; for (int i = 0; i < 5; i = i + 1) { s = s + i; } return s; }"
    )
    assert "_for_top:" in asm
    assert "_for_step:" in asm
    assert "_for_end:" in asm


def test_break_jumps_to_loop_end():
    asm = _compile(
        "int main(void) { int i = 0; while (1) { if (i > 3) break; i = i + 1; } return i; }"
    )
    # `break` lowers to a jmp at any of the *_end labels.
    end_labels = [line for line in asm.splitlines() if line.endswith("_end:")]
    assert end_labels, "no loop-end label found"
    label = end_labels[0].rstrip(":").lstrip()
    assert any(f"jmp     {label}" in line for line in asm.splitlines())


def test_continue_jumps_to_step_in_for():
    asm = _compile(
        "int main(void) { int s = 0; for (int i = 0; i < 5; i = i + 1) { "
        "if (i == 2) continue; s = s + i; } return s; }"
    )
    step_label = next(line for line in asm.splitlines() if line.endswith("_for_step:"))
    label = step_label.rstrip(":").lstrip()
    assert any(f"jmp     {label}" in line for line in asm.splitlines())


def test_break_outside_loop_rejected():
    with pytest.raises(CodegenError, match="break.*outside"):
        _compile("int main(void) { break; return 0; }")


def test_continue_outside_loop_rejected():
    with pytest.raises(CodegenError, match="continue.*outside"):
        _compile("int main(void) { continue; return 0; }")


def test_logical_and_short_circuits():
    asm = _compile("int main(void) { int a = 1; int b = 2; return a && b; }")
    assert "_and_false:" in asm
    assert "_and_end:" in asm
    # Two `jz` to the false label — one per operand.
    assert sum(1 for line in asm.splitlines() if "jz      .L" in line and "_and_false" in line) == 2


def test_logical_or_short_circuits():
    asm = _compile("int main(void) { int a = 0; int b = 1; return a || b; }")
    assert "_or_true:" in asm
    assert "_or_end:" in asm
    assert sum(1 for line in asm.splitlines() if "jnz     .L" in line and "_or_true" in line) == 2


# ---- function calls and parameters -----------------------------------------

def test_parameter_read_uses_positive_offset():
    # cdecl: first param at [ebp + 8], second at [ebp + 12].
    asm = _compile("int add(int a, int b) { return a + b; } int main(void) { return 0; }")
    assert "mov     eax, [ebp + 8]" in asm
    assert "mov     eax, [ebp + 12]" in asm


def test_parameter_can_be_assigned():
    asm = _compile("int f(int x) { x = x + 1; return x; } int main(void) { return 0; }")
    assert "mov     [ebp + 8], eax" in asm


def test_call_pushes_args_right_to_left_and_cleans_up():
    asm = _compile(
        "int add(int a, int b) { return a + b; } "
        "int main(void) { return add(2, 3); }"
    )
    lines = asm.splitlines()
    # Find the call site in main; arg pushes happen just above it.
    call_idx = next(i for i, l in enumerate(lines) if l.strip() == "call    _add")
    pushes = [l for l in lines[call_idx - 4:call_idx] if "push    eax" in l]
    assert len(pushes) == 2
    # Last arg (3) is loaded first, pushed first → ends up at higher address.
    assert "mov     eax, 3" in lines[call_idx - 4]
    assert "mov     eax, 2" in lines[call_idx - 2]
    # Caller cleans up 2 args * 4 bytes.
    assert lines[call_idx + 1].strip() == "add     esp, 8"


def test_call_with_no_args_no_cleanup():
    asm = _compile("int g(void) { return 5; } int main(void) { return g(); }")
    assert "call    _g" in asm
    # No `add esp, ...` should follow the call (zero-arg cleanup is elided).
    assert "add     esp" not in asm.split("call    _g")[1].splitlines()[0]


def test_indirect_call_rejected():
    # Calling a function pointer (or any non-Identifier callee) isn't supported yet.
    with pytest.raises(CodegenError, match="direct calls"):
        _compile(
            "int main(void) { int x = 0; return (x ? main : main)(); }"
        )


def test_duplicate_parameter_rejected():
    with pytest.raises(CodegenError, match="duplicate parameter"):
        _compile("int f(int x, int x) { return x; } int main(void) { return 0; }")


def test_param_and_local_share_namespace():
    # A local can't shadow a parameter (flat scope).
    with pytest.raises(CodegenError, match="redeclaration"):
        _compile("int f(int x) { int x = 7; return x; } int main(void) { return 0; }")


# ---- compound assign / ++ / -- / ternary -----------------------------------

@pytest.mark.parametrize(
    "op,instr",
    [
        ("+=", "add     eax, ecx"),
        ("-=", "sub     eax, ecx"),
        ("*=", "imul    eax, ecx"),
        ("&=", "and     eax, ecx"),
        ("|=", "or      eax, ecx"),
        ("^=", "xor     eax, ecx"),
    ],
)
def test_compound_assign_simple(op, instr):
    asm = _compile(f"int main(void) {{ int x = 1; x {op} 2; return x; }}")
    assert instr in asm
    # Result is written back to the lvalue's slot.
    assert "mov     [ebp - 4], eax" in asm


def test_compound_div_uses_idiv():
    asm = _compile("int main(void) { int x = 10; x /= 3; return x; }")
    assert "idiv    ecx" in asm
    assert "mov     [ebp - 4], eax" in asm


def test_prefix_increment_loads_new_value():
    asm = _compile("int main(void) { int x = 5; int y = ++x; return y; }")
    lines = asm.splitlines()
    # The prefix bump emits `inc dword [ebp-4]` *before* loading EAX from
    # that slot. Find them and verify ordering.
    inc_idx = next(i for i, l in enumerate(lines) if "inc     dword [ebp - 4]" in l)
    load_idx = next(i for i, l in enumerate(lines) if i > inc_idx and l.strip() == "mov     eax, [ebp - 4]")
    assert load_idx == inc_idx + 1


def test_postfix_increment_loads_old_value():
    asm = _compile("int main(void) { int x = 5; int y = x++; return y; }")
    lines = asm.splitlines()
    # Postfix loads the old value, then bumps in place.
    load_idx = next(
        i for i, l in enumerate(lines)
        if l.strip() == "mov     eax, [ebp - 4]"
        and i + 1 < len(lines)
        and "inc     dword [ebp - 4]" in lines[i + 1]
    )
    assert load_idx >= 0


def test_decrement_uses_dec():
    asm = _compile("int main(void) { int x = 5; --x; return x; }")
    assert "dec     dword [ebp - 4]" in asm


def test_inc_dec_on_non_identifier_rejected():
    with pytest.raises(CodegenError, match="must be an identifier"):
        _compile("int main(void) { int x = 0; ++(x + 1); return 0; }")


def test_ternary_emits_cond_branch_and_join():
    asm = _compile("int main(void) { int x = 1; return x ? 7 : 9; }")
    assert "_tern_false:" in asm
    assert "_tern_end:" in asm
    # True branch loads 7, false branch loads 9.
    assert "mov     eax, 7" in asm
    assert "mov     eax, 9" in asm


# ---- pointers --------------------------------------------------------------

def test_address_of_emits_lea():
    asm = _compile("int main(void) { int x = 0; int *p = &x; return 0; }")
    assert "lea     eax, [ebp - 4]" in asm


def test_dereference_loads_through_eax():
    asm = _compile("int main(void) { int x = 7; int *p = &x; return *p; }")
    # Load pointer value, then read through it.
    assert "mov     eax, [ebp - 8]" in asm
    assert "mov     eax, [eax]" in asm


def test_store_through_pointer_writes_via_ecx():
    asm = _compile("int main(void) { int x = 0; int *p = &x; *p = 42; return x; }")
    # Pointer evaluated first (push eax), rhs into eax, pop into ecx, store.
    assert "push    eax" in asm
    assert "pop     ecx" in asm
    assert "mov     [ecx], eax" in asm


def test_address_of_non_identifier_rejected():
    with pytest.raises(CodegenError, match="`&` operand"):
        _compile("int main(void) { return &(1 + 2); }")


def test_pointer_param_passed_as_int():
    # int*-typed param shares a 4-byte slot at [ebp+8].
    asm = _compile(
        "int deref(int *p) { return *p; } "
        "int main(void) { int x = 5; return deref(&x); }"
    )
    assert "mov     eax, [ebp + 8]" in asm
    assert "mov     eax, [eax]" in asm
    # Caller passes &x as the argument.
    assert "lea     eax, [ebp - 4]" in asm


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
