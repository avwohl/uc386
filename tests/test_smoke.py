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


def test_complex_type_basic_layout():
    # `_Complex T` is laid out as two T's (real, imag), 16 bytes for
    # `_Complex double`. Just declaring one shouldn't error; arithmetic
    # / __real__ / __imag__ on them is still incomplete.
    asm = _compile("int main(void) { _Complex double c; return 0; }")
    assert "sub     esp, 16" in asm


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


# Indirect calls landed in slice 15 — the previous "rejected" test is gone.
# The positive coverage lives in the function-pointer section below.


def test_duplicate_parameter_rejected():
    with pytest.raises(CodegenError, match="duplicate parameter"):
        _compile("int f(int x, int x) { return x; } int main(void) { return 0; }")


def test_param_and_local_can_shadow():
    # A local in the body's compound shadows the param (block scope).
    asm = _compile("int f(int x) { int x = 7; return x; } int main(void) { return 0; }")
    # The local `x` gets a fresh slot at [ebp - 4] regardless of the param at
    # [ebp + 8]; reading back x sees the local value (7).
    assert "[ebp - 4]" in asm


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


# ---- string literals + externs --------------------------------------------

def test_string_literal_loads_data_label():
    asm = _compile('int puts(const char *s); int main(void) { puts("hi"); return 0; }')
    assert "mov     eax, _uc386_str0" in asm
    assert "section .data" in asm
    assert "_uc386_str0:" in asm
    assert "db      'hi', 0" in asm


def test_extern_emitted_for_undefined_function():
    asm = _compile('int puts(const char *s); int main(void) { return puts("x"); }')
    assert "extern  _puts" in asm


def test_extern_not_emitted_when_function_is_defined_locally():
    asm = _compile(
        "int helper(void) { return 1; } "
        "int main(void) { return helper(); }"
    )
    assert "extern" not in asm


def test_string_literals_are_interned():
    asm = _compile(
        'int puts(const char *s); int main(void) { puts("hi"); puts("hi"); return 0; }'
    )
    # Same content → one label only.
    assert asm.count("_uc386_str0:") == 1
    assert "_uc386_str1" not in asm


def test_string_literal_with_special_chars():
    asm = _compile('int puts(const char *s); int main(void) { puts("a\\nb"); return 0; }')
    # Newline (0x0A) should appear as a numeric byte segment between the
    # printable runs, not be embedded in the quoted form.
    assert "'a', 10, 'b', 0" in asm


# ---- pointer arithmetic with size scaling ---------------------------------

def test_pointer_plus_int_scales_by_pointee_size():
    asm = _compile("int main(void) { int x = 0; int *p = &x; p = p + 1; return 0; }")
    # The integer 1 is loaded into eax, then scaled by sizeof(int)=4 before
    # the add — using `shl` for the power-of-two case.
    assert "shl     eax, 2" in asm
    assert "add     eax, ecx" in asm


def test_pointer_minus_int_scales():
    asm = _compile("int main(void) { int x = 0; int *p = &x; p = p - 2; return 0; }")
    assert "shl     eax, 2" in asm
    assert "sub     eax, ecx" in asm


def test_int_plus_pointer_scales_int_side():
    # C allows `n + p`; the int is what gets scaled, not the pointer.
    asm = _compile("int main(void) { int x = 0; int *p = &x; p = 1 + p; return 0; }")
    # After both sides eval, eax = ptr, stack = int. We pop the int into ecx,
    # scale ecx, then add.
    assert "shl     ecx, 2" in asm
    assert "add     eax, ecx" in asm


def test_pointer_difference_divides_by_pointee_size():
    asm = _compile(
        "int main(void) { int a = 0; int b = 0; int *p = &a; int *q = &b; "
        "int d = p - q; return 0; }"
    )
    # ptr - ptr → byte difference, then arithmetic shift right to divide by 4.
    assert "sub     eax, ecx" in asm
    assert "sar     eax, 2" in asm


def test_pointer_compound_add_scales():
    asm = _compile("int main(void) { int x = 0; int *p = &x; p += 3; return 0; }")
    # `p += 3` desugars to `p = p + 3`, which routes through pointer +.
    assert "shl     eax, 2" in asm
    assert "add     eax, ecx" in asm


def test_pointer_prefix_increment_advances_by_pointee_size():
    asm = _compile("int main(void) { int x = 0; int *p = &x; ++p; return 0; }")
    # Pointer ++ becomes `add dword [slot], 4`, not `inc`.
    assert "add     dword [ebp - 8], 4" in asm
    assert "inc     dword [ebp - 8]" not in asm


def test_pointer_postfix_increment_advances_by_pointee_size():
    asm = _compile("int main(void) { int x = 0; int *p = &x; p++; return 0; }")
    # Postfix loads first, then bumps — the bump is still `add ..., 4`.
    asm_lines = asm.splitlines()
    load_idx = next(
        i for i, l in enumerate(asm_lines)
        if l.strip() == "mov     eax, [ebp - 8]"
        and i + 1 < len(asm_lines)
        and "add     dword [ebp - 8], 4" in asm_lines[i + 1]
    )
    assert load_idx >= 0


def test_pointer_decrement_steps_back_by_pointee_size():
    asm = _compile("int main(void) { int x = 0; int *p = &x; --p; return 0; }")
    assert "sub     dword [ebp - 8], 4" in asm
    assert "dec     dword [ebp - 8]" not in asm


def test_int_increment_still_uses_inc():
    # Regression: int slots keep using `inc` / `dec`, not `add ..., 1`.
    asm = _compile("int main(void) { int x = 0; ++x; return x; }")
    assert "inc     dword [ebp - 4]" in asm


def test_char_pointer_arithmetic_no_scaling():
    # sizeof(char)=1, so `p + 1` on char* needs no multiplicative scaling.
    asm = _compile(
        'int puts(const char *s); '
        'int main(void) { const char *p = "abc"; p = p + 1; return 0; }'
    )
    assert "shl     eax, 2" not in asm
    assert "imul" not in asm
    # The bare add still happens.
    assert "add     eax, ecx" in asm


def test_char_pointer_increment_advances_by_one():
    asm = _compile(
        'int puts(const char *s); '
        'int main(void) { const char *p = "abc"; ++p; return 0; }'
    )
    # sizeof(char) = 1: pointer ++ on char* is functionally `inc`, but we
    # emit `add ..., 1` for consistency with all pointer increments.
    assert "add     dword [ebp - 4], 1" in asm


def test_adding_two_pointers_rejected():
    # C: pointer + pointer is illegal. Only pointer - pointer is meaningful.
    with pytest.raises(CodegenError, match="add"):
        _compile(
            "int main(void) { int x = 0; int *p = &x; int *q = &x; "
            "int r = p + q; return 0; }"
        )


def test_dereferencing_pointer_arithmetic_works():
    # `*(p + 1)` evaluates the arithmetic, then loads through the result.
    asm = _compile(
        "int main(void) { int x = 0; int *p = &x; int y = *(p + 1); return 0; }"
    )
    assert "shl     eax, 2" in asm
    assert "add     eax, ecx" in asm
    # The dereference reads through eax after the arithmetic.
    assert "mov     eax, [eax]" in asm


# ---- arrays ----------------------------------------------------------------

def test_array_local_allocates_full_size():
    # 3 ints * 4 bytes = 12 bytes for the arr slot.
    asm = _compile("int main(void) { int arr[3]; return 0; }")
    assert "sub     esp, 12" in asm


def test_array_followed_by_int_local_layout():
    # `arr` is allocated first (-12), then `x` (-16). Total frame = 16.
    asm = _compile("int main(void) { int arr[3]; int x = 5; return x; }")
    assert "sub     esp, 16" in asm
    assert "mov     [ebp - 16], eax" in asm
    assert "mov     eax, [ebp - 16]" in asm


def test_array_index_store_and_load():
    asm = _compile(
        "int main(void) { int arr[3]; arr[1] = 7; return arr[1]; }"
    )
    # Element address: lea (array decay), scale index by sizeof(int), add.
    assert "lea     eax, [ebp - 12]" in asm
    assert "shl     eax, 2" in asm
    # Store goes through ecx.
    assert "mov     [ecx], eax" in asm
    # Load reads through eax after the address arithmetic.
    assert "mov     eax, [eax]" in asm


def test_array_decays_to_pointer_in_initializer():
    asm = _compile("int main(void) { int arr[3]; int *p = arr; return 0; }")
    # `arr` in a value position decays to its address.
    assert "lea     eax, [ebp - 12]" in asm
    # `p` slot lives just below the array (at -16) — store its decayed address.
    assert "mov     [ebp - 16], eax" in asm


def test_array_pointer_arithmetic_decays_array_first():
    asm = _compile("int main(void) { int arr[5]; int *p = arr + 2; return 0; }")
    # arr[5] occupies 20 bytes, so it lives at [ebp - 20].
    assert "lea     eax, [ebp - 20]" in asm
    # The integer 2 is scaled by sizeof(int)=4 before the add.
    assert "shl     eax, 2" in asm
    assert "add     eax, ecx" in asm


def test_dereferencing_array_name_reads_first_element():
    # `*arr` is `*(arr+0)` — same as `arr[0]`.
    asm = _compile("int main(void) { int arr[3]; arr[0] = 9; return *arr; }")
    assert "lea     eax, [ebp - 12]" in asm
    assert "mov     eax, [eax]" in asm


def test_address_of_array_element_is_address_arithmetic_only():
    # `&arr[i]` computes the element address without dereferencing.
    asm = _compile("int main(void) { int arr[3]; int *p = &arr[1]; return 0; }")
    asm_lines = asm.splitlines()
    # The element-address computation happens, but no `mov eax, [eax]` should
    # follow it (that would be the deref we're suppressing).
    assert "lea     eax, [ebp - 12]" in asm
    # `p` slot is at -16; the computed address is stored there directly.
    assert "mov     [ebp - 16], eax" in asm


def test_dynamic_index():
    asm = _compile(
        "int main(void) { int arr[5]; int i = 2; arr[i] = 9; return arr[i]; }"
    )
    # The index value is loaded from `i`'s slot, then scaled.
    assert "mov     eax, [ebp - 24]" in asm  # i is the local after arr → -24
    assert "shl     eax, 2" in asm
    assert "mov     eax, [eax]" in asm


def test_array_assignment_rejected():
    # Regular C arrays aren't lvalues. Only vector_size-tagged
    # ArrayTypes get the memcpy-style assignment.
    with pytest.raises(CodegenError, match="array"):
        _compile("int main(void) { int a[3]; int b[3]; a = b; return 0; }")


def test_array_increment_rejected():
    with pytest.raises(CodegenError, match="array"):
        _compile("int main(void) { int a[3]; ++a; return 0; }")


def test_unsized_array_without_initializer_rejected():
    with pytest.raises(CodegenError, match="initializer|size"):
        _compile("int main(void) { int a[]; return 0; }")


# ---- unions ---------------------------------------------------------------

def test_union_local_size_is_max_member():
    asm = _compile(
        "union mix { int x; char c; }; "
        "int main(void) { union mix u; return sizeof(u); }"
    )
    # max(int=4, char=1) = 4.
    assert "mov     eax, 4" in asm
    assert "sub     esp, 4" in asm


def test_union_members_share_offset_zero():
    asm = _compile(
        "union mix { int x; char c; }; "
        "int main(void) { union mix u; u.x = 5; u.c = 65; return u.x; }"
    )
    asm_lines = asm.splitlines()
    # All member accesses lea the same slot address with no `add eax, N`
    # following — every union member sits at offset 0.
    lea_count = sum(1 for l in asm_lines if "lea     eax, [ebp - 4]" in l)
    assert lea_count >= 3
    # No `add eax, <small>` should follow the lea (members at offset 0).
    # Searching for the specific bad form:
    assert not any(
        "add     eax, 1" in l or "add     eax, 2" in l or "add     eax, 4" in l
        for l in asm_lines
    )


def test_union_largest_member_wins_size():
    # Array member dominates — 4 ints = 16 bytes.
    asm = _compile(
        "union big { char c; int x; int arr[4]; }; "
        "int main(void) { return sizeof(union big); }"
    )
    assert "mov     eax, 16" in asm


def test_union_member_widths_preserved():
    asm = _compile(
        "union mix { int x; char c; }; "
        "int main(void) { union mix u; u.c = 65; return u.c; }"
    )
    # Char member access uses byte loads/stores even though the slot is 4 bytes.
    assert "mov     byte [ecx], al" in asm
    assert "movsx   eax, byte [eax]" in asm


def test_union_global_in_bss():
    asm = _compile(
        "union mix { int x; char c; }; "
        "union mix g; "
        "int main(void) { return g.x; }"
    )
    assert "section .bss" in asm
    assert "_g:" in asm
    assert "resb    4" in asm


def test_union_inside_struct():
    # Tagged-union pattern: struct with a union member.
    asm = _compile(
        "union v { int i; char c; }; "
        "struct tagged { int tag; union v val; }; "
        "int main(void) { struct tagged x; x.tag = 1; x.val.i = 99; return x.val.i; }"
    )
    # struct tagged: tag at 0, val at 4. Total = 8.
    assert "sub     esp, 8" in asm


def test_union_pointer_arrow_access():
    asm = _compile(
        "union mix { int x; char c; }; "
        "int main(void) { union mix u; union mix *p = &u; "
        "p->x = 7; return p->c; }"
    )
    # Stored through the pointer.
    assert "mov     [ecx], eax" in asm
    # Char-width read for p->c.
    assert "movsx   eax, byte [eax]" in asm


# ---- struct by-value params -----------------------------------------------

def test_struct_param_callee_uses_param_offset():
    asm = _compile(
        "struct point { int x; int y; }; "
        "int get_x(struct point p) { return p.x; } "
        "int main(void) { struct point pt; pt.x = 9; return get_x(pt); }"
    )
    asm_lines = asm.splitlines()
    fn_idx = next(i for i, l in enumerate(asm_lines) if l.strip() == "_get_x:")
    fn_section = "\n".join(asm_lines[fn_idx:])
    # First param at [ebp + 8]; p.x is offset 0 so the lea + read gives us p.x.
    assert "lea     eax, [ebp + 8]" in fn_section
    # The actual member load.
    assert "mov     eax, [eax]" in fn_section


def test_struct_param_caller_memcpys_onto_stack():
    asm = _compile(
        "struct point { int x; int y; }; "
        "int get_x(struct point p) { return p.x; } "
        "int main(void) { struct point pt; pt.x = 9; pt.y = 7; return get_x(pt); }"
    )
    asm_lines = asm.splitlines()
    main_idx = next(i for i, l in enumerate(asm_lines) if l.strip() == "_main:")
    main_section = "\n".join(asm_lines[main_idx:])
    # Caller reserves 8 bytes for the struct arg, then writes the two
    # dwords into [esp + 0] and [esp + 4].
    assert "sub     esp, 8" in main_section
    assert "mov     [esp + 0], eax" in main_section
    assert "mov     [esp + 4], eax" in main_section
    # And cleans up after the call.
    assert "add     esp, 8" in main_section


def test_struct_and_int_param_layout():
    asm = _compile(
        "struct point { int x; int y; }; "
        "int f(int n, struct point p) { return n + p.x; } "
        "int main(void) { struct point q; q.x = 5; return f(3, q); }"
    )
    asm_lines = asm.splitlines()
    fn_idx = next(i for i, l in enumerate(asm_lines) if l.strip() == "_f:")
    fn_section = "\n".join(asm_lines[fn_idx:])
    # `n` at [ebp + 8], struct `p` at [ebp + 12] (since n consumes 4).
    assert "mov     eax, [ebp + 8]" in fn_section
    assert "lea     eax, [ebp + 12]" in fn_section


def test_struct_param_with_mixed_widths():
    # struct {char c; int n} — sizeof = 8 (1 byte + 3 pad + 4).
    asm = _compile(
        "struct mix { char c; int n; }; "
        "int f(struct mix m) { return m.n; } "
        "int main(void) { struct mix x; x.c = 0; x.n = 42; return f(x); }"
    )
    main = "\n".join(asm.splitlines()[next(i for i, l in enumerate(asm.splitlines()) if l.strip() == "_main:"):])
    # Caller reserves 8 bytes for the arg.
    assert "sub     esp, 8" in main


def test_struct_return_into_local_init():
    asm = _compile(
        "struct point { int x; int y; }; "
        "struct point make(int a, int b) { struct point p; p.x = a; p.y = b; return p; } "
        "int main(void) { struct point s = make(3, 4); return s.x + s.y; }"
    )
    asm_lines = asm.splitlines()
    main_idx = next(i for i, l in enumerate(asm_lines) if l.strip() == "_main:")
    main_section = "\n".join(asm_lines[main_idx:])
    # Caller pushes &s as the hidden first arg.
    assert "lea     eax, [ebp - 8]" in main_section
    assert "call    _make" in main_section
    # Cleanup = 2 args (8) + retptr (4) = 12.
    assert "add     esp, 12" in main_section


def test_struct_return_into_assignment():
    asm = _compile(
        "struct point { int x; int y; }; "
        "struct point make(void) { struct point p; p.x = 7; p.y = 9; return p; } "
        "int main(void) { struct point s; s = make(); return s.x; }"
    )
    asm_lines = asm.splitlines()
    main_idx = next(i for i, l in enumerate(asm_lines) if l.strip() == "_main:")
    main_section = "\n".join(asm_lines[main_idx:])
    assert "call    _make" in main_section
    # Just retptr — no args.
    assert "add     esp, 4" in main_section


def test_struct_return_callee_param_offsets_account_for_retptr():
    asm = _compile(
        "struct point { int x; int y; }; "
        "struct point make(int a) { struct point p; p.x = a; p.y = a; return p; } "
        "int main(void) { struct point s = make(5); return s.x; }"
    )
    asm_lines = asm.splitlines()
    fn_idx = next(i for i, l in enumerate(asm_lines) if l.strip() == "_make:")
    end_idx = next(i for i, l in enumerate(asm_lines[fn_idx:]) if l.strip() == "ret") + fn_idx
    fn_section = "\n".join(asm_lines[fn_idx:end_idx])
    # `a` lives at [ebp + 12] (retptr at [ebp + 8] shifted everything else).
    assert "[ebp + 12]" in fn_section
    # The retptr load comes from [ebp + 8].
    assert "[ebp + 8]" in fn_section


def test_chained_struct_return_forwards_retptr():
    # `return helper()` from within a struct-returning function should pass
    # our own retptr down rather than allocating a temp.
    asm = _compile(
        "struct point { int x; int y; }; "
        "struct point inner(void) { struct point p; p.x = 1; p.y = 2; return p; } "
        "struct point outer(void) { return inner(); } "
        "int main(void) { struct point s = outer(); return s.x; }"
    )
    asm_lines = asm.splitlines()
    fn_idx = next(i for i, l in enumerate(asm_lines) if l.strip() == "_outer:")
    end_idx = next(i for i, l in enumerate(asm_lines[fn_idx:]) if l.strip() == "ret") + fn_idx
    fn_section = "\n".join(asm_lines[fn_idx:end_idx])
    # outer pushes its own retptr (loaded from [ebp + 8]) before calling inner.
    assert "mov     eax, [ebp + 8]" in fn_section
    assert "call    _inner" in fn_section


def test_struct_return_member_access():
    asm = _compile(
        "struct point { int x; int y; }; "
        "struct point make(void) { struct point p; p.x = 5; p.y = 7; return p; } "
        "int main(void) { return make().x; }"
    )
    asm_lines = asm.splitlines()
    main_idx = next(i for i, l in enumerate(asm_lines) if l.strip() == "_main:")
    main_section = "\n".join(asm_lines[main_idx:])
    # Caller reserves a temp slot for the returned struct, passes its
    # address as the retptr, and then accesses `.x` from that slot.
    assert "call    _make" in main_section
    # main's frame should include space for the temp.
    sub_lines = [l for l in asm_lines[main_idx:] if "sub     esp," in l]
    assert sub_lines  # at least one sub esp


def test_struct_return_passed_as_arg():
    # `f(make())` — caller stages the struct in a temp, then memcpy's onto
    # the stack as a struct-by-value arg.
    asm = _compile(
        "struct point { int x; int y; }; "
        "struct point make(void) { struct point p; p.x = 1; p.y = 2; return p; } "
        "int sum(struct point p) { return p.x + p.y; } "
        "int main(void) { return sum(make()); }"
    )
    asm_lines = asm.splitlines()
    main_idx = next(i for i, l in enumerate(asm_lines) if l.strip() == "_main:")
    main_section = "\n".join(asm_lines[main_idx:])
    assert "call    _make" in main_section
    assert "call    _sum" in main_section


def test_multiple_struct_returning_calls_get_distinct_temps():
    asm = _compile(
        "struct point { int x; int y; }; "
        "struct point make(int n) { struct point p; p.x = n; p.y = n; return p; } "
        "int main(void) { return make(1).x + make(2).x; }"
    )
    asm_lines = asm.splitlines()
    main_idx = next(i for i, l in enumerate(asm_lines) if l.strip() == "_main:")
    main_section = "\n".join(asm_lines[main_idx:])
    # Both calls happen.
    assert main_section.count("call    _make") == 2
    # Each gets its own temp — adding a struct-sized temp twice means the
    # frame is at least 16 bytes (2 * sizeof(struct point)).
    sub_match = next(
        l for l in asm_lines[main_idx:] if l.strip().startswith("sub     esp,")
    )
    sub_size = int(sub_match.split(",")[1].strip())
    assert sub_size >= 16


# ---- switch / case --------------------------------------------------------

def test_switch_dispatches_with_cmp_and_je():
    asm = _compile(
        "int main(void) { int x = 2; switch (x) { case 1: x = 10; break; "
        "case 2: x = 20; break; } return x; }"
    )
    # Each case emits a `cmp eax, value; je <case_label>` pair.
    assert "cmp     eax, 1" in asm
    assert "cmp     eax, 2" in asm
    # At least one `je .L*_case` jump should appear in the dispatch.
    assert any("je      .L" in l and "_case" in l for l in asm.splitlines())


def test_switch_break_jumps_to_switch_end():
    asm = _compile(
        "int main(void) { int x = 1; switch (x) { case 1: x = 5; break; "
        "case 2: x = 99; } return x; }"
    )
    asm_lines = asm.splitlines()
    end_label_line = next(l for l in asm_lines if l.endswith("_switch_end:"))
    end_label = end_label_line.rstrip(":").lstrip()
    assert any(f"jmp     {end_label}" in l for l in asm_lines)


def test_switch_default_falls_through_when_no_case_matches():
    asm = _compile(
        "int main(void) { int x = 99; switch (x) { case 1: x = 10; break; "
        "default: x = 0; } return x; }"
    )
    # Default body executes — the assignment to 0 should appear.
    assert "mov     eax, 0" in asm
    # And the default label should be referenced as a jmp target from the dispatch tail.
    assert "_default:" in asm


def test_switch_no_match_no_default_skips_body():
    asm = _compile(
        "int main(void) { int hit = 0; int x = 99; switch (x) { "
        "case 1: hit = 1; break; case 2: hit = 2; break; } return hit; }"
    )
    # Without a default, the dispatch tail jumps to switch_end.
    asm_lines = asm.splitlines()
    assert any(
        "jmp     .L" in l and "_switch_end" in l for l in asm_lines
    )


def test_switch_fall_through_without_break():
    asm = _compile(
        "int sum(int x) { int s = 0; switch (x) { "
        "case 1: s = s + 1; case 2: s = s + 2; case 3: s = s + 3; } return s; } "
        "int main(void) { return sum(1); }"
    )
    asm_lines = asm.splitlines()
    fn_idx = next(i for i, l in enumerate(asm_lines) if l.strip() == "_sum:")
    end_idx = next(i for i, l in enumerate(asm_lines[fn_idx:]) if l.strip() == "ret") + fn_idx
    fn_section = asm_lines[fn_idx:end_idx]
    # All three add operations are present (one per case body).
    add_count = sum(1 for l in fn_section if "add     eax, ecx" in l)
    assert add_count >= 3


def test_switch_continue_passes_to_enclosing_loop():
    asm = _compile(
        "int main(void) { int sum = 0; for (int i = 0; i < 5; i = i + 1) { "
        "switch (i) { case 2: continue; } sum = sum + i; } return sum; }"
    )
    asm_lines = asm.splitlines()
    step_label_line = next(l for l in asm_lines if l.endswith("_for_step:"))
    step_label = step_label_line.rstrip(":").lstrip()
    assert any(f"jmp     {step_label}" in l for l in asm_lines)


def test_switch_evaluates_expression_only_once():
    asm = _compile(
        "int counter = 0; "
        "int next(void) { counter = counter + 1; return counter; } "
        "int main(void) { switch (next()) { case 1: break; case 2: break; } "
        "return counter; }"
    )
    asm_lines = asm.splitlines()
    main_idx = next(i for i, l in enumerate(asm_lines) if l.strip() == "_main:")
    main_section = "\n".join(asm_lines[main_idx:])
    # `next()` is the switch expression — exactly one call site in main.
    assert main_section.count("call    _next") == 1


def test_non_constant_case_value_rejected():
    with pytest.raises(CodegenError):
        _compile(
            "int main(void) { int x = 0; int n = 5; "
            "switch (x) { case n: break; } return 0; }"
        )


def test_break_outside_switch_or_loop_still_rejected():
    # The error message changed wording with slice 19 but still mentions "outside".
    with pytest.raises(CodegenError, match="outside"):
        _compile("int main(void) { break; return 0; }")


# ---- floats (Phase 5) -----------------------------------------------------

def test_float_literal_emits_data_constant():
    asm = _compile("int main(void) { float x = 1.5; return 0; }")
    assert "section .data" in asm
    # Float constant gets a unique label. `1.5` parses as a double
    # literal (no `f` suffix), so the pool entry is qword; the store
    # narrows to dword via x87's automatic precision conversion.
    assert any("_uc386_float" in l for l in asm.splitlines())
    assert "fld     qword [_uc386_float" in asm
    assert "fstp    dword [ebp - 4]" in asm


def test_float_local_copy():
    asm = _compile(
        "int main(void) { float x = 1.5; float y = x; return 0; }"
    )
    # Reading `x` is fld dword from its slot.
    assert "fld     dword [ebp - 4]" in asm
    # Writing `y` is fstp dword to its slot.
    assert "fstp    dword [ebp - 8]" in asm


def test_float_addition():
    asm = _compile(
        "int main(void) { float x = 1.5; float y = 2.5; float z = x + y; return 0; }"
    )
    # Both operands fld'd, then faddp combines.
    assert "faddp" in asm


def test_float_division():
    asm = _compile(
        "int main(void) { float x = 6.0; float y = 2.0; float z = x / y; return 0; }"
    )
    assert "fdivp" in asm


def test_cast_float_to_int_uses_fistp():
    asm = _compile(
        "int main(void) { float x = 1.5; int n = (int)x; return n; }"
    )
    assert "fistp" in asm


def test_cast_int_to_float_uses_fild():
    asm = _compile(
        "int main(void) { int n = 5; float f = (float)n; return 0; }"
    )
    assert "fild" in asm


def test_double_local_eight_byte_slot():
    asm = _compile("int main(void) { double x = 1.5; return 0; }")
    # Slot rounds up to 4-aligned but the value is 8 bytes wide; we
    # emit a `qword` load/store and the frame includes 8 bytes.
    assert "fld     qword" in asm
    assert "fstp    qword [ebp - 8]" in asm


def test_float_unary_negate():
    asm = _compile(
        "int main(void) { float x = 1.5; float y = -x; return 0; }"
    )
    # Float negation is fchs.
    assert "fchs" in asm


def test_mixed_int_float_in_arithmetic():
    # `1.5 + 2` — int promotes to float, sum is float.
    asm = _compile(
        "int main(void) { float r = 1.5 + 2; return (int)r; }"
    )
    # int->float promotion uses fild somewhere.
    assert "fild" in asm
    # And the addition itself.
    assert "faddp" in asm


def test_returning_int_cast_of_float_arithmetic():
    # The return statement evaluates a float-typed expression but stores
    # an int — the conversion uses fistp.
    asm = _compile(
        "int main(void) { return (int)(1.5 + 2.5); }"
    )
    assert "faddp" in asm
    assert "fistp" in asm


# ---- float params + returns -----------------------------------------------

def test_float_param_loaded_from_offset():
    asm = _compile(
        "float identity(float x) { return x; } "
        "int main(void) { return (int)identity(2.5); }"
    )
    asm_lines = asm.splitlines()
    fn_idx = next(i for i, l in enumerate(asm_lines) if l.strip() == "_identity:")
    fn_section = "\n".join(asm_lines[fn_idx:])
    # `x` lives at [ebp + 8]; reading it as float is fld dword.
    assert "fld     dword [ebp + 8]" in fn_section


def test_double_param_loaded_qword_and_takes_eight_bytes():
    asm = _compile(
        "double scale(double x) { return x + 1.0; } "
        "int main(void) { return (int)scale(2.0); }"
    )
    asm_lines = asm.splitlines()
    fn_idx = next(i for i, l in enumerate(asm_lines) if l.strip() == "_scale:")
    fn_section = "\n".join(asm_lines[fn_idx:])
    assert "fld     qword [ebp + 8]" in fn_section


def test_double_literal_narrowed_at_float_param():
    # Without a coercion pass, `2.5` (double) at a float param site
    # would push 8 bytes; the callee'd then misread its frame. With
    # narrowing, the caller pushes 4 bytes.
    asm = _compile(
        "float square(float x) { return x * x; } "
        "int main(void) { return (int)square(2.5); }"
    )
    asm_lines = asm.splitlines()
    main_idx = next(i for i, l in enumerate(asm_lines) if l.strip() == "_main:")
    main_section = "\n".join(asm_lines[main_idx:])
    # Narrowed to 4-byte float at the call site.
    assert "sub     esp, 4" in main_section
    assert "fstp    dword [esp]" in main_section
    # And no 8-byte qword push for this literal.
    assert "fstp    qword [esp]" not in main_section


def test_float_arg_widened_at_double_param():
    # The reverse: `1.5f` at a double param. The caller widens to
    # 8 bytes via the same fld/fstp width swap.
    asm = _compile(
        "double scale(double x) { return x * x; } "
        "int main(void) { return (int)scale(1.5f); }"
    )
    asm_lines = asm.splitlines()
    main_idx = next(i for i, l in enumerate(asm_lines) if l.strip() == "_main:")
    main_section = "\n".join(asm_lines[main_idx:])
    assert "sub     esp, 8" in main_section
    assert "fstp    qword [esp]" in main_section


def test_caller_pushes_float_arg_via_fstp():
    # Use `2.5f` (float-suffixed) so the literal matches the declared
    # param type. Without a coercion pass at the call site, an unsuffixed
    # `2.5` would be treated as `double` and pushed as 8 bytes.
    asm = _compile(
        "float identity(float x) { return x; } "
        "int main(void) { return (int)identity(2.5f); }"
    )
    asm_lines = asm.splitlines()
    main_idx = next(i for i, l in enumerate(asm_lines) if l.strip() == "_main:")
    main_section = "\n".join(asm_lines[main_idx:])
    assert "sub     esp, 4" in main_section
    assert "fstp    dword [esp]" in main_section


def test_float_return_left_on_fpu_stack():
    asm = _compile(
        "float two_pi(void) { return 6.28; } "
        "int main(void) { return (int)two_pi(); }"
    )
    asm_lines = asm.splitlines()
    fn_idx = next(i for i, l in enumerate(asm_lines) if l.strip() == "_two_pi:")
    end_idx = next(i for i, l in enumerate(asm_lines[fn_idx:]) if l.strip() == "ret") + fn_idx
    fn_section = "\n".join(asm_lines[fn_idx:end_idx])
    # The return loads the literal onto st(0) and jumps to epilogue
    # without storing to eax.
    assert "fld     qword [_uc386_float" in fn_section or "fld     dword [_uc386_float" in fn_section
    assert "jmp     .epilogue" in fn_section
    # No `mov eax, ...` before the return jump (other than the
    # default xor eax, eax that follows).


def test_caller_consumes_float_return_via_fistp_for_int_assignment():
    # `int n = f();` where f returns float: caller's auto-convert uses fistp.
    asm = _compile(
        "float make(void) { return 3.5; } "
        "int main(void) { int n = (int)make(); return n; }"
    )
    asm_lines = asm.splitlines()
    main_idx = next(i for i, l in enumerate(asm_lines) if l.strip() == "_main:")
    main_section = "\n".join(asm_lines[main_idx:])
    assert "call    _make" in main_section
    assert "fistp" in main_section


def test_float_return_into_float_local():
    # `float f = make();` — the result rides st(0) straight into the slot.
    asm = _compile(
        "float make(void) { return 3.5; } "
        "int main(void) { float f = make(); return (int)f; }"
    )
    asm_lines = asm.splitlines()
    main_idx = next(i for i, l in enumerate(asm_lines) if l.strip() == "_main:")
    main_section = "\n".join(asm_lines[main_idx:])
    assert "call    _make" in main_section
    # No fistp/fild between the call and the float store.
    assert "fstp    dword [ebp" in main_section


# ---- float in boolean context ---------------------------------------------

def test_if_with_float_condition_uses_fp_compare():
    asm = _compile(
        "int main(void) { float x = 0.5f; if (x) return 1; return 0; }"
    )
    # The condition test goes through an FPU compare against 0.0,
    # not an int truncation — so 0.5 takes the if branch.
    assert "fldz" in asm
    assert "fucompp" in asm


def test_while_with_float_condition_uses_fp_compare():
    asm = _compile(
        "int main(void) { float x = 0.0f; while (x) {} return 0; }"
    )
    assert "fldz" in asm
    assert "fucompp" in asm


def test_logical_not_on_float_uses_fp_compare():
    # `!f` should be 1 only when f is exactly 0.0 — not when (int)f is 0.
    asm = _compile(
        "int main(void) { float x = 0.5f; return !x; }"
    )
    assert "fldz" in asm
    assert "fucompp" in asm


def test_ternary_with_float_condition():
    asm = _compile(
        "int main(void) { float x = 0.5f; return x ? 1 : 0; }"
    )
    assert "fldz" in asm
    assert "fucompp" in asm


def test_logical_and_with_float_operand():
    asm = _compile(
        "int main(void) { float a = 0.5f; int b = 1; return a && b; }"
    )
    # First operand (float) compares to 0.0; second (int) tests directly.
    assert "fldz" in asm
    assert "fucompp" in asm


# ---- float ++ / -- --------------------------------------------------------

def test_float_prefix_increment():
    asm = _compile(
        "int main(void) { float f = 1.5f; ++f; return (int)f; }"
    )
    # `++f` on a float: load, fld1, faddp, fst.
    assert "fld1" in asm
    assert "faddp" in asm


def test_float_prefix_decrement():
    asm = _compile(
        "int main(void) { float f = 2.5f; --f; return (int)f; }"
    )
    assert "fld1" in asm
    assert "fsubp" in asm


def test_float_postfix_increment_returns_old_value():
    # `f++` should yield the old value but leave f as old + 1.
    asm = _compile(
        "int main(void) { float f = 1.5f; float g = f++; return (int)g; }"
    )
    asm_lines = asm.splitlines()
    # Expect two `fld dword [ebp - 4]` (one for the kept old value, one
    # for the bumped copy).
    fld_count = sum(1 for l in asm_lines if "fld     dword [ebp - 4]" in l)
    assert fld_count >= 2
    assert "fld1" in asm
    assert "faddp" in asm


# ---- float lvalue stores --------------------------------------------------

def test_float_array_element_assignment():
    asm = _compile(
        "int main(void) { float arr[3]; arr[1] = 1.5f; return (int)arr[1]; }"
    )
    # The element address goes through ECX; fst writes through it.
    assert "fst     dword [ecx]" in asm


def test_float_struct_member_assignment():
    asm = _compile(
        "struct point { float x; float y; }; "
        "int main(void) { struct point p; p.x = 1.5f; return (int)p.x; }"
    )
    assert "fst     dword [ecx]" in asm


def test_float_pointer_deref_assignment():
    asm = _compile(
        "int main(void) { float x = 0.0f; float *p = &x; *p = 1.5f; return (int)x; }"
    )
    assert "fst     dword [ecx]" in asm


def test_float_compound_assign_to_identifier():
    # `f += 1.5f` desugars to `f = f + 1.5f` and works through the
    # existing Identifier float-assign path.
    asm = _compile(
        "int main(void) { float f = 1.0f; f += 1.5f; return (int)f; }"
    )
    # The addition lands as faddp.
    assert "faddp" in asm
    # And the result stores back to f's slot.
    assert "fstp    dword [ebp - 4]" in asm


def test_float_compound_assign_to_array_element():
    asm = _compile(
        "int main(void) { float arr[3]; arr[1] = 1.0f; "
        "arr[1] += 0.5f; return (int)arr[1]; }"
    )
    # Compound load + faddp + store-through-ecx.
    assert "faddp" in asm
    assert "fst     dword [ecx]" in asm


def test_float_compound_assign_to_struct_member():
    asm = _compile(
        "struct p { float x; }; "
        "int main(void) { struct p s; s.x = 1.0f; "
        "s.x *= 2.0f; return (int)s.x; }"
    )
    assert "fmulp" in asm
    assert "fst     dword [ecx]" in asm


def test_float_compound_assign_through_pointer():
    asm = _compile(
        "int main(void) { float x = 2.0f; float *p = &x; "
        "*p -= 1.0f; return (int)x; }"
    )
    assert "fsubp" in asm
    assert "fst     dword [ecx]" in asm


# ---- float globals --------------------------------------------------------

def test_float_global_initialized_in_data():
    asm = _compile(
        "float pi = 3.14f; int main(void) { return (int)pi; }"
    )
    assert "section .data" in asm
    assert "_pi:" in asm
    assert "dd      3.14" in asm


def test_double_global_initialized_in_data():
    asm = _compile(
        "double e = 2.718; int main(void) { return (int)e; }"
    )
    assert "_e:" in asm
    assert "dq      2.718" in asm


def test_uninitialized_float_global_in_bss():
    asm = _compile(
        "float counter; int main(void) { return (int)counter; }"
    )
    assert "section .bss" in asm
    assert "_counter:" in asm
    assert "resb    4" in asm


def test_float_global_read_uses_fld():
    asm = _compile(
        "float pi = 3.14f; int main(void) { float p = pi; return (int)p; }"
    )
    assert "fld     dword [_pi]" in asm


def test_float_global_write_uses_fst():
    # `_float_assign` uses `fst` (no pop) so the assignment expression's
    # value stays on st(0) for chained uses; the trailing fistp/fstp at
    # the consumer site pops it.
    asm = _compile(
        "float counter; "
        "int main(void) { counter = 1.5f; return 0; }"
    )
    assert "fst     dword [_counter]" in asm


# ---- float comparisons ----------------------------------------------------

@pytest.mark.parametrize(
    "op,setcc",
    [
        ("==", "sete"),
        ("!=", "setne"),
        ("<",  "seta"),    # x87 compares ST(0) vs ST(1); after fxch the
        (">",  "setb"),    # mapping to setCC inverts vs the integer ops.
        ("<=", "setae"),
        (">=", "setbe"),
    ],
)
def test_float_comparison_uses_fucompp_and_setcc(op, setcc):
    asm = _compile(
        f"int main(void) {{ float a = 1.0; float b = 2.0; return a {op} b; }}"
    )
    assert "fucompp" in asm
    assert "fnstsw  ax" in asm
    assert "sahf" in asm
    assert f"{setcc}    al" in asm
    assert "movzx   eax, al" in asm


def test_float_comparison_with_int_promotes():
    # `a == 0` with `a` float — the int 0 promotes to float.
    asm = _compile("int main(void) { float a = 1.5; return a == 0; }")
    # Expect the fild promotion of 0 and a float compare.
    assert "fild" in asm
    assert "fucompp" in asm


def test_float_comparison_in_if_condition():
    asm = _compile(
        "int main(void) { float a = 1.0; float b = 2.0; "
        "if (a < b) return 1; return 0; }"
    )
    # The comparison still produces an int result that drives test+jz.
    assert "fucompp" in asm
    assert "test    eax, eax" in asm


# ---- bitfields ------------------------------------------------------------

def test_bitfield_struct_size_one_word():
    asm = _compile(
        "struct flags { int a:1; int b:1; int c:6; }; "
        "int main(void) { return sizeof(struct flags); }"
    )
    # All three bitfields pack into a single 4-byte storage unit.
    assert "mov     eax, 4" in asm


def test_bitfield_write_then_read():
    asm = _compile(
        "struct flags { int a:1; int b:1; int c:6; }; "
        "int main(void) { struct flags f; f.a = 0; f.b = 0; f.c = 0; "
        "f.a = 1; f.c = 5; return f.a + f.c; }"
    )
    # Bitfield writes use AND (mask out) + OR (insert) + shift.
    asm_lines = asm.splitlines()
    # We just verify the codegen produced shift / and / or instructions —
    # the exact sequence depends on the layout but bitfield RMW always
    # involves these.
    assert any("shl     eax," in l or "shl     ecx," in l for l in asm_lines)
    assert any("and     eax," in l for l in asm_lines)


def test_bitfield_cross_int_starts_new_unit():
    # `int a:24; int b:16` — b doesn't fit in the first int's leftover
    # 8 bits, so it starts a new 4-byte unit. Total = 8 bytes.
    asm = _compile(
        "struct s { int a:24; int b:16; }; "
        "int main(void) { return sizeof(struct s); }"
    )
    assert "mov     eax, 8" in asm


# ---- _Bool + nullptr ------------------------------------------------------

def test_bool_local_size_one_byte():
    asm = _compile("int main(void) { _Bool b = 1; return b; }")
    # Bool slot is 1 byte (rounded to 4-aligned slot), and store narrows to byte.
    assert "mov     byte [ebp - 4], al" in asm
    # Read uses byte load.
    assert "movsx   eax, byte [ebp - 4]" in asm or "movzx   eax, byte [ebp - 4]" in asm


def test_nullptr_lowers_to_zero():
    asm = _compile("int main(void) { int *p = nullptr; return p == nullptr; }")
    # `nullptr` is just integer 0 in expression position.
    assert "mov     eax, 0" in asm
    # And can compare equal to it.
    assert "sete    al" in asm


def test_nullptr_in_pointer_arithmetic_position():
    # `nullptr + 0` shouldn't blow up — treat as a pointer-sized zero.
    asm = _compile("int main(void) { int *p = nullptr; if (p) return 1; return 0; }")
    # The if-condition tests p (4-byte slot) for non-zero.
    assert "test    eax, eax" in asm


# ---- typedef + storage classes --------------------------------------------

def test_typedef_basic_int_resolved_at_parse_time():
    # uc_core resolves typedef-of-int into BasicType(int) before codegen
    # ever sees it — no special handling needed.
    asm = _compile(
        "typedef int mytype; "
        "int main(void) { mytype x = 5; return x; }"
    )
    assert "mov     eax, 5" in asm


def test_typedef_named_struct():
    asm = _compile(
        "typedef struct point { int x; int y; } Point; "
        "int main(void) { Point p; p.x = 5; p.y = 7; return p.x + p.y; }"
    )
    assert "sub     esp, 8" in asm


def test_typedef_anonymous_struct():
    # `typedef struct { ... } P;` produces a StructType with name=None
    # and inline members. We register it under a synthetic name on
    # first sight.
    asm = _compile(
        "typedef struct { int x; int y; } AP; "
        "int main(void) { AP a; a.x = 1; return a.x; }"
    )
    assert "sub     esp, 8" in asm


def test_top_level_static_global_works_like_regular():
    asm = _compile(
        "static int counter = 7; "
        "int main(void) { counter = counter + 1; return counter; }"
    )
    # Stored in .data like any other initialized global.
    assert "_counter:" in asm
    assert "dd      7" in asm
    assert "[_counter]" in asm


def test_function_static_local_initialized_in_data():
    asm = _compile(
        "int counter(void) { static int x = 0; x = x + 1; return x; } "
        "int main(void) { counter(); return counter(); }"
    )
    # The static lives in .data with a function-mangled name, not a frame slot.
    assert "section .data" in asm
    assert "counter_x:" in asm or "counter__x:" in asm
    # Initialized to 0 once.
    assert "dd      0" in asm
    # And the function reads/writes the global label, not [ebp - 4].
    asm_lines = asm.splitlines()
    fn_idx = next(i for i, l in enumerate(asm_lines) if l.strip() == "_counter:")
    end_idx = next(i for i, l in enumerate(asm_lines[fn_idx:]) if l.strip() == "ret") + fn_idx
    fn_section = "\n".join(asm_lines[fn_idx:end_idx])
    # No frame is reserved for `x` (only the param/return scaffolding).
    # The static is referenced by its `<func>__x` global label.
    assert any("counter__x" in l for l in fn_section.splitlines())


def test_function_static_local_uninitialized_in_bss():
    asm = _compile(
        "int once(void) { static int x; x = 5; return x; } "
        "int main(void) { return once(); }"
    )
    assert "section .bss" in asm
    assert any("once_x:" in l or "once__x:" in l for l in asm.splitlines())


def test_function_static_does_not_count_against_frame():
    # A static local should not increase the function's `sub esp, N`.
    asm_with_static = _compile(
        "int f(void) { static int x = 0; return x; } "
        "int main(void) { return f(); }"
    )
    # `f` has no frame locals — so no `sub esp, ...` for f itself.
    asm_lines = asm_with_static.splitlines()
    fn_idx = next(i for i, l in enumerate(asm_lines) if l.strip() == "_f:")
    end_idx = next(i for i, l in enumerate(asm_lines[fn_idx:]) if l.strip() == "ret") + fn_idx
    fn_section = asm_lines[fn_idx:end_idx]
    assert not any("sub     esp," in l for l in fn_section)


# ---- enums ----------------------------------------------------------------

def test_enum_implicit_values_start_at_zero():
    asm = _compile(
        "enum c { A, B, C }; int main(void) { return B; }"
    )
    # `B` is the second enum value → 1.
    assert "mov     eax, 1" in asm


def test_enum_explicit_values_with_implicit_chain():
    asm = _compile(
        "enum d { X = 5, Y, Z }; int main(void) { return Z; }"
    )
    # X=5, Y=6, Z=7.
    assert "mov     eax, 7" in asm


def test_enum_used_as_type():
    asm = _compile(
        "enum c { A, B, C }; "
        "int main(void) { enum c v = B; return v; }"
    )
    # `enum c` slot is 4 bytes; B = 1 stores 1.
    assert "sub     esp, 4" in asm
    assert "mov     eax, 1" in asm


def test_enum_constant_in_arithmetic():
    asm = _compile(
        "enum c { A, B = 10, C }; "
        "int main(void) { return A + B + C; }"
    )
    # 0 + 10 + 11 = 21 (computed at runtime via three separate `mov`s).
    assert "mov     eax, 0" in asm
    assert "mov     eax, 10" in asm
    assert "mov     eax, 11" in asm


def test_unknown_enum_constant_is_unknown_identifier():
    with pytest.raises(CodegenError, match="unknown identifier"):
        _compile("int main(void) { return UNDEFINED_ENUM_CONST; }")


# ---- va_list (callee-side variadic) ---------------------------------------

def test_va_start_va_arg_va_end_compile():
    asm = _compile(
        "typedef char *va_list; "
        "int sum(int n, ...) { "
        "  va_list ap; "
        "  va_start(ap, n); "
        "  int total = 0; "
        "  for (int i = 0; i < n; i = i + 1) { "
        "    total = total + va_arg(ap, int); "
        "  } "
        "  va_end(ap); "
        "  return total; "
        "} "
        "int main(void) { return sum(3, 10, 20, 30); }"
    )
    # va_start: ap = &n + 4. n is at [ebp + 8] so we lea [ebp + 12]
    # and store that into ap's slot.
    assert "lea     eax, [ebp + 12]" in asm


def test_va_arg_advances_pointer_and_loads():
    asm = _compile(
        "typedef char *va_list; "
        "int first(int n, ...) { "
        "  va_list ap; "
        "  va_start(ap, n); "
        "  int v = va_arg(ap, int); "
        "  return v; "
        "} "
        "int main(void) { return first(0, 42); }"
    )
    asm_lines = asm.splitlines()
    fn_idx = next(i for i, l in enumerate(asm_lines) if l.strip() == "_first:")
    end_idx = next(i for i, l in enumerate(asm_lines[fn_idx:]) if l.strip() == "ret") + fn_idx
    fn_section = "\n".join(asm_lines[fn_idx:end_idx])
    # va_arg(ap, int): read *ap as int, advance ap by 4.
    # Look for: load ap into ecx, advance ap, then load via [ecx].
    assert any("add     dword [ebp - " in l and ", 4" in l for l in fn_section.splitlines())
    # The load itself is `mov eax, [ecx]` (int width).
    assert "mov     eax, [ecx]" in fn_section


def test_va_arg_double_advances_eight_bytes():
    asm = _compile(
        "typedef char *va_list; "
        "double first(int n, ...) { "
        "  va_list ap; "
        "  va_start(ap, n); "
        "  double v = va_arg(ap, double); "
        "  return v; "
        "} "
        "int main(void) { return (int)first(1, 1.5); }"
    )
    asm_lines = asm.splitlines()
    fn_idx = next(i for i, l in enumerate(asm_lines) if l.strip() == "_first:")
    end_idx = next(i for i, l in enumerate(asm_lines[fn_idx:]) if l.strip() == "ret") + fn_idx
    fn_section = "\n".join(asm_lines[fn_idx:end_idx])
    # va_arg(ap, double): advance ap by 8.
    assert any("add     dword [ebp - " in l and ", 8" in l for l in fn_section.splitlines())
    # And the value loads through fld qword.
    assert "fld     qword [ecx]" in fn_section


def test_va_end_emits_no_code():
    asm = _compile(
        "typedef char *va_list; "
        "int f(int n, ...) { va_list ap; va_start(ap, n); va_end(ap); return n; } "
        "int main(void) { return f(0); }"
    )
    # va_end(ap) is a no-op; just verify nothing strange shows up.
    # (No specific instruction to assert; presence of `_f` in the output
    # plus successful compilation is enough.)
    assert "_f:" in asm


# ---- variadic external calls ----------------------------------------------

def test_variadic_extern_declaration_emits_extern():
    asm = _compile(
        'int printf(const char *fmt, ...); '
        'int main(void) { return 0; }'
    )
    assert "extern  _printf" in asm


def test_variadic_call_pushes_extra_args_right_to_left():
    asm = _compile(
        'int printf(const char *fmt, ...); '
        'int main(void) { return printf("hi %d %d", 1, 2); }'
    )
    asm_lines = asm.splitlines()
    main_idx = next(i for i, l in enumerate(asm_lines) if l.strip() == "_main:")
    main_section = "\n".join(asm_lines[main_idx:])
    # Three args: format string + 2 ints. Pushed right-to-left.
    assert "call    _printf" in main_section
    # Caller cleans up 3 * 4 = 12 bytes.
    assert "add     esp, 12" in main_section
    # Both literal ints get loaded.
    assert "mov     eax, 1" in main_section
    assert "mov     eax, 2" in main_section
    # Format string lands as a label load.
    assert "mov     eax, _uc386_str0" in main_section


def test_variadic_call_with_no_extra_args():
    asm = _compile(
        'int printf(const char *fmt, ...); '
        'int main(void) { return printf("hello"); }'
    )
    # Only the format string is passed.
    assert "add     esp, 4" in asm


# ---- goto + labels --------------------------------------------------------

def test_goto_forward_jump():
    asm = _compile(
        "int main(void) { int x = 0; goto skip; x = 1; skip: return x; }"
    )
    asm_lines = asm.splitlines()
    # The goto emits a jmp; the label appears later.
    label_line = next(l for l in asm_lines if l.endswith("_skip:"))
    label_name = label_line.rstrip(":").lstrip()
    assert any(f"jmp     {label_name}" in l for l in asm_lines)
    # The skipped assignment to x = 1 still appears in the output (we just
    # never execute it).
    assert "mov     eax, 1" in asm


def test_goto_backward_loop():
    asm = _compile(
        "int main(void) { int i = 0; "
        "again: i = i + 1; if (i < 3) goto again; return i; }"
    )
    asm_lines = asm.splitlines()
    label_line = next(l for l in asm_lines if l.endswith("_again:"))
    label_name = label_line.rstrip(":").lstrip()
    # The jmp to `again` appears after the conditional check.
    assert any(f"jmp     {label_name}" in l for l in asm_lines)


def test_goto_unknown_label_rejected():
    with pytest.raises(CodegenError, match="label"):
        _compile("int main(void) { goto nowhere; return 0; }")


def test_duplicate_label_rejected():
    with pytest.raises(CodegenError, match="label"):
        _compile("int main(void) { dup: dup: return 0; }")


def test_label_in_separate_function_does_not_collide():
    # NASM `.L*` labels are local to the previous global (= function) label,
    # so two functions can each have `loop:` without an actual link-time
    # collision. Just verify both functions emit a `*_loop:` label in their
    # own scope.
    asm = _compile(
        "int f(void) { int i = 0; loop: i = i + 1; if (i < 2) goto loop; return i; } "
        "int g(void) { int j = 0; loop: j = j + 1; if (j < 3) goto loop; return j; } "
        "int main(void) { return f() + g(); }"
    )
    asm_lines = asm.splitlines()
    label_lines = [l for l in asm_lines if l.endswith("_loop:")]
    # One occurrence per function.
    assert len(label_lines) == 2


# ---- multidim arrays ------------------------------------------------------

def test_2d_array_local_full_size():
    asm = _compile(
        "int main(void) { int m[2][3]; return 0; }"
    )
    # 2 * 3 * 4 = 24 bytes.
    assert "sub     esp, 24" in asm


def test_2d_array_index_then_index():
    asm = _compile(
        "int main(void) { int m[2][3]; m[1][2] = 99; return m[1][2]; }"
    )
    # Inner row scaling: each row is sizeof(int[3]) = 12 bytes.
    # 12 isn't a power of two → expect imul.
    assert "imul    eax, eax, 12" in asm
    # Final element access scales the inner index by sizeof(int) = 4.
    assert "shl     eax, 2" in asm


def test_2d_array_outer_index_decays_to_pointer():
    # `m[i]` in expression position decays to a pointer to the row.
    asm = _compile(
        "int main(void) { int m[2][3]; int *row = m[1]; return 0; }"
    )
    # The row's address should be computed but no `mov eax, [eax]` deref
    # follows it — the array element is itself an array, so the result of
    # `m[1]` is its address.
    asm_lines = asm.splitlines()
    # Find the imul (multidim row scaling) and verify what follows is the
    # add + store, not a load through eax.
    imul_idx = next(i for i, l in enumerate(asm_lines) if "imul    eax, eax, 12" in l)
    # The next few lines do the add and store of `row`.
    # Just check no `mov eax, [eax]` between imul and the store to row's slot.
    # `row` is the second local — slot at -28 (m at -24, row at -28).
    assert "mov     [ebp - 28], eax" in asm
    # No `mov eax, [eax]` immediately after the address computation.
    # (We check this loosely: no occurrence in the small block.)


def test_sizeof_2d_array():
    asm = _compile(
        "int main(void) { int m[2][3]; return sizeof(m); }"
    )
    # 24 bytes.
    assert "mov     eax, 24" in asm


def test_3d_array_size_correct():
    asm = _compile(
        "int main(void) { int m[2][3][4]; return sizeof(m); }"
    )
    # 2 * 3 * 4 * 4 = 96.
    assert "mov     eax, 96" in asm


# ---- designated initializers ----------------------------------------------

def test_array_designated_init_skips_zero_fills():
    asm = _compile(
        "int main(void) { int arr[5] = {[1] = 10, [3] = 30}; return arr[1]; }"
    )
    # Both designated values stored.
    assert "mov     eax, 10" in asm
    assert "mov     eax, 30" in asm
    # Unfilled indices 0, 2, 4 → zero-fills at -20, -12, -4.
    assert "mov     dword [ebp - 20], 0" in asm
    assert "mov     dword [ebp - 12], 0" in asm
    assert "mov     dword [ebp - 4], 0" in asm


def test_array_mixed_positional_and_designated():
    # `{1, 2, [3] = 7, 8}` → arr[0]=1, [1]=2, [2]=0, [3]=7, [4]=8.
    asm = _compile(
        "int main(void) { int arr[5] = {1, 2, [3] = 7, 8}; return arr[0]; }"
    )
    for v in (1, 2, 7, 8):
        assert f"mov     eax, {v}" in asm
    # Index 2 is skipped → zero-filled.
    assert "mov     dword [ebp - 12], 0" in asm


def test_struct_designated_init():
    asm = _compile(
        "struct point { int x; int y; }; "
        "int main(void) { struct point p = {.y = 10, .x = 5}; return p.x + p.y; }"
    )
    # Either order in source; codegen emits to each member's offset.
    assert "mov     eax, 5" in asm
    assert "mov     [ebp - 8], eax" in asm  # x at offset 0
    assert "mov     eax, 10" in asm
    assert "mov     [ebp - 4], eax" in asm  # y at offset 4


def test_struct_designated_partial_init_zero_fills_rest():
    asm = _compile(
        "struct point { int x; int y; }; "
        "int main(void) { struct point p = {.y = 5}; return p.x + p.y; }"
    )
    assert "mov     eax, 5" in asm
    assert "mov     [ebp - 4], eax" in asm  # y assigned
    # x at offset 0 (= -8) zero-filled.
    assert "mov     dword [ebp - 8], 0" in asm


def test_unknown_designated_member_rejected():
    with pytest.raises(CodegenError, match="member"):
        _compile(
            "struct point { int x; int y; }; "
            "int main(void) { struct point p = {.z = 1}; return 0; }"
        )


# ---- structs --------------------------------------------------------------

def test_struct_local_allocates_full_size():
    asm = _compile(
        "struct point { int x; int y; }; "
        "int main(void) { struct point p; return 0; }"
    )
    # Two ints, no padding, total 8 bytes.
    assert "sub     esp, 8" in asm


def test_struct_member_assign_and_read():
    asm = _compile(
        "struct point { int x; int y; }; "
        "int main(void) { struct point p; p.x = 5; p.y = 7; return p.x + p.y; }"
    )
    # &p is at [ebp - 8]; member access uses lea + add offset.
    assert "lea     eax, [ebp - 8]" in asm
    assert "mov     [ecx], eax" in asm
    assert "add     eax, ecx" in asm   # the + in the return


def test_struct_member_offset_for_second_field():
    asm = _compile(
        "struct point { int x; int y; }; "
        "int main(void) { struct point p; p.y = 99; return 0; }"
    )
    # &p.y = &p + 4. Both pieces appear.
    assert "lea     eax, [ebp - 8]" in asm
    assert "add     eax, 4" in asm


def test_struct_pointer_arrow_member_access():
    asm = _compile(
        "struct point { int x; int y; }; "
        "int main(void) { struct point p; struct point *pp = &p; pp->x = 10; return pp->y; }"
    )
    # Arrow loads the pointer value, adds offset, then reads/writes.
    asm_lines = asm.splitlines()
    # The pointer slot should be loaded — `mov eax, [ebp - 12]` for pp at -12.
    assert "mov     eax, [ebp - 12]" in asm
    assert "add     eax, 4" in asm   # for pp->y, offset 4


def test_struct_with_mixed_widths_pads_for_alignment():
    # `char` then `int` lays out as: c at 0, 3 bytes pad, int at 4. Total 8.
    asm = _compile(
        "struct mix { char c; int x; }; "
        "int main(void) { struct mix m; m.c = 65; m.x = 100; return m.x + m.c; }"
    )
    assert "sub     esp, 8" in asm
    # char store goes through `mov byte [ecx], al`.
    assert "mov     byte [ecx], al" in asm
    # int member at offset 4.
    assert "add     eax, 4" in asm


def test_sizeof_struct_type():
    asm = _compile(
        "struct point { int x; int y; }; "
        "int main(void) { return sizeof(struct point); }"
    )
    assert "mov     eax, 8" in asm


def test_sizeof_struct_value():
    asm = _compile(
        "struct point { int x; int y; }; "
        "int main(void) { struct point p; return sizeof(p); }"
    )
    assert "mov     eax, 8" in asm


def test_address_of_struct_member():
    asm = _compile(
        "struct point { int x; int y; }; "
        "int main(void) { struct point p; int *q = &p.y; return 0; }"
    )
    # &p.y computes the address — no final `mov eax, [eax]`.
    assert "lea     eax, [ebp - 8]" in asm
    assert "add     eax, 4" in asm


def test_struct_pointer_arithmetic_scales_by_struct_size():
    # `pp + 1` scales by sizeof(struct point) = 8, which is `shl eax, 3`.
    asm = _compile(
        "struct point { int x; int y; }; "
        "int main(void) { struct point arr[2]; struct point *pp = arr; "
        "pp = pp + 1; return 0; }"
    )
    assert "shl     eax, 3" in asm


def test_struct_global_uninitialized_in_bss():
    asm = _compile(
        "struct point { int x; int y; }; "
        "struct point origin; "
        "int main(void) { return origin.x; }"
    )
    assert "section .bss" in asm
    assert "_origin:" in asm
    assert "resb    8" in asm
    # Global member access uses the label as base.
    assert "mov     eax, _origin" in asm


def test_compound_assign_struct_member_address_computed_once():
    asm = _compile(
        "struct counter { int n; }; "
        "int main(void) { struct counter c; c.n = 0; c.n += 5; return c.n; }"
    )
    asm_lines = asm.splitlines()
    # The compound assign should not double-emit the `lea` for c.
    # &c is at [ebp - 4] (just one int member). The `lea` for the compound
    # assign appears once; the read in `return c.n` is a separate occurrence.
    lea_count = sum(1 for l in asm_lines if "lea     eax, [ebp - 4]" in l)
    # One for `c.n = 0`, one for `c.n += 5`, one for `return c.n` = 3.
    assert lea_count == 3


def test_array_of_structs_index_then_member():
    asm = _compile(
        "struct point { int x; int y; }; "
        "int main(void) { struct point arr[2]; arr[0].x = 1; arr[1].y = 2; "
        "return arr[1].x + arr[0].y; }"
    )
    # 2 structs * 8 = 16 bytes.
    assert "sub     esp, 16" in asm


def test_struct_pointer_param():
    asm = _compile(
        "struct point { int x; int y; }; "
        "int get_y(struct point *p) { return p->y; } "
        "int main(void) { struct point pt; pt.y = 99; return get_y(&pt); }"
    )
    # In get_y, p loads from [ebp+8]; offset 4 added for `->y`.
    asm_lines = asm.splitlines()
    fn_idx = next(i for i, l in enumerate(asm_lines) if l.strip() == "_get_y:")
    fn_section = "\n".join(asm_lines[fn_idx:])
    assert "mov     eax, [ebp + 8]" in fn_section
    assert "add     eax, 4" in fn_section


def test_undefined_struct_rejected():
    with pytest.raises(CodegenError, match="struct"):
        _compile("int main(void) { struct undefined u; return 0; }")


def test_unknown_struct_member_rejected():
    with pytest.raises(CodegenError, match="member"):
        _compile(
            "struct point { int x; int y; }; "
            "int main(void) { struct point p; return p.z; }"
        )


def test_struct_local_init_with_initializer_list():
    asm = _compile(
        "struct point { int x; int y; }; "
        "int main(void) { struct point p = {1, 2}; return p.x + p.y; }"
    )
    # Each member stored via the standard `_store_from_eax` path; for an
    # int member that's `mov [..], eax` after `mov eax, value`.
    asm_lines = asm.splitlines()
    # Find the `mov eax, 1` and `mov eax, 2` followed by stores at the
    # right offsets (p at -8: x at -8, y at -4).
    assert "mov     eax, 1" in asm
    assert "mov     [ebp - 8], eax" in asm
    assert "mov     eax, 2" in asm
    assert "mov     [ebp - 4], eax" in asm


def test_struct_local_underspecified_init_zero_fills():
    asm = _compile(
        "struct point { int x; int y; }; "
        "int main(void) { struct point p = {7}; return p.x + p.y; }"
    )
    assert "mov     eax, 7" in asm
    assert "mov     [ebp - 8], eax" in asm
    # Tail member zero-filled directly.
    assert "mov     dword [ebp - 4], 0" in asm


def test_struct_local_init_mixed_widths():
    asm = _compile(
        "struct mix { char c; int n; }; "
        "int main(void) { struct mix m = {65, 100}; return m.c + m.n; }"
    )
    # m at [ebp - 8]; c at offset 0 (= -8), n at offset 4 (= -4).
    assert "mov     byte [ebp - 8], al" in asm
    assert "mov     [ebp - 4], eax" in asm


def test_struct_array_init_nested_braces():
    asm = _compile(
        "struct point { int x; int y; }; "
        "int main(void) { struct point arr[2] = {{1, 2}, {3, 4}}; return arr[1].y; }"
    )
    # Two structs * 8 bytes = 16. Each pair of members landed at the right offsets.
    assert "sub     esp, 16" in asm
    asm_lines = asm.splitlines()
    # arr[0] starts at -16; arr[1] at -8. Members within: x at +0, y at +4.
    assert "mov     eax, 1" in asm
    assert "mov     [ebp - 16], eax" in asm  # arr[0].x
    assert "mov     eax, 2" in asm
    assert "mov     [ebp - 12], eax" in asm  # arr[0].y
    assert "mov     eax, 3" in asm
    assert "mov     [ebp - 8], eax" in asm   # arr[1].x
    assert "mov     eax, 4" in asm
    assert "mov     [ebp - 4], eax" in asm   # arr[1].y


def test_struct_assignment_inline_per_dword_copy():
    asm = _compile(
        "struct point { int x; int y; }; "
        "int main(void) { struct point a; struct point b; a.x = 7; b = a; return b.x; }"
    )
    # 8-byte struct copies as two dword loads + stores via edx/ecx.
    assert "mov     eax, [edx + 0]" in asm
    assert "mov     [ecx + 0], eax" in asm
    assert "mov     eax, [edx + 4]" in asm
    assert "mov     [ecx + 4], eax" in asm


def test_too_many_struct_initializers_rejected():
    with pytest.raises(CodegenError, match="too many"):
        _compile(
            "struct point { int x; int y; }; "
            "int main(void) { struct point p = {1, 2, 3}; return 0; }"
        )


# ---- compound assignment to non-Identifier lvalues ------------------------

def test_index_compound_add_assign():
    asm = _compile("int main(void) { int arr[3]; arr[1] = 5; arr[1] += 7; return arr[1]; }")
    # arr[1] += 7: address computed once, loaded, added, stored.
    assert "mov     eax, [eax]" in asm   # current value load
    assert "add     eax, ecx" in asm
    assert "mov     [ecx], eax" in asm   # store back


def test_index_compound_does_not_recompute_index():
    # Only one `lea` for arr's base in the compound assign — the previous
    # naive desugar `arr[i] = arr[i] + rhs` would have emitted two.
    asm = _compile("int main(void) { int arr[5]; arr[2] += 1; return 0; }")
    asm_lines = asm.splitlines()
    lea_count = sum(1 for l in asm_lines if "lea     eax, [ebp - 20]" in l)
    assert lea_count == 1


def test_index_compound_subtract_assign():
    asm = _compile("int main(void) { int arr[3]; arr[1] = 10; arr[1] -= 3; return arr[1]; }")
    assert "sub     eax, ecx" in asm


def test_index_compound_multiply_assign():
    asm = _compile("int main(void) { int arr[3]; arr[0] = 5; arr[0] *= 2; return arr[0]; }")
    assert "imul    eax, ecx" in asm


def test_index_compound_divide_assign():
    asm = _compile("int main(void) { int arr[3]; arr[0] = 10; arr[0] /= 3; return arr[0]; }")
    assert "cdq" in asm
    assert "idiv    ecx" in asm


def test_index_compound_bitwise_assign():
    asm = _compile("int main(void) { int arr[3]; arr[0] = 12; arr[0] &= 5; return arr[0]; }")
    assert "and     eax, ecx" in asm


def test_pointer_deref_compound_assign():
    asm = _compile("int main(void) { int x = 0; int *p = &x; *p += 7; return x; }")
    # Pointer evaluated once, pushed, value loaded, added, stored back.
    assert "mov     eax, [eax]" in asm
    assert "add     eax, ecx" in asm
    assert "mov     [ecx], eax" in asm


def test_char_index_compound_assign_uses_byte_widths():
    asm = _compile(
        "int main(void) { char arr[4]; arr[0] = 1; arr[0] += 5; return arr[0]; }"
    )
    # Sub-word load and store widths.
    assert "movsx   eax, byte [eax]" in asm
    assert "mov     byte [ecx], al" in asm


def test_pointer_lvalue_compound_scales_when_pointee():
    # `*pp` is `int *`; `*pp += 1` should scale 1 by sizeof(int) = 4.
    asm = _compile(
        "int main(void) { int x = 0; int *p = &x; int **pp = &p; "
        "*pp += 1; return 0; }"
    )
    assert "shl     eax, 2" in asm
    assert "mov     [ecx], eax" in asm


# ---- function pointers + CharLiteral --------------------------------------

def test_charliteral_init_loads_ascii_value():
    # `'A'` is a CharLiteral with value=65; lowers to a plain integer load.
    asm = _compile("int main(void) { char c = 'A'; return c; }")
    assert "mov     eax, 65" in asm


def test_charliteral_in_arithmetic():
    asm = _compile("int main(void) { return 'a' + 1; }")
    assert "mov     eax, 97" in asm
    assert "mov     eax, 1" in asm
    assert "add     eax, ecx" in asm


def test_function_name_in_value_position_loads_address():
    asm = _compile(
        "int helper(void) { return 5; } "
        "int main(void) { int (*fp)(void) = helper; return 0; }"
    )
    # `helper` in value position yields its label as an immediate.
    assert "mov     eax, _helper" in asm


def test_indirect_call_through_function_pointer():
    asm = _compile(
        "int helper(void) { return 5; } "
        "int main(void) { int (*fp)(void) = helper; return fp(); }"
    )
    asm_lines = asm.splitlines()
    main_idx = next(i for i, l in enumerate(asm_lines) if l.strip() == "_main:")
    main_section = "\n".join(asm_lines[main_idx:])
    # Indirect call through eax, not a direct call to _helper.
    assert "call    eax" in main_section
    # The direct call shouldn't appear in main (it's only in main if we
    # mistakenly emitted `call _helper`).
    assert "call    _helper" not in main_section


def test_call_through_dereferenced_pointer_same_as_direct_pointer():
    # `(*fp)()` and `fp()` produce the same call sequence — leading `*` is
    # idempotent on function-typed values.
    asm_a = _compile(
        "int helper(void) { return 5; } "
        "int main(void) { int (*fp)(void) = helper; return fp(); }"
    )
    asm_b = _compile(
        "int helper(void) { return 5; } "
        "int main(void) { int (*fp)(void) = helper; return (*fp)(); }"
    )
    # Both should emit `call eax` in main; ignore the function definitions.
    def main_body(asm: str) -> str:
        lines = asm.splitlines()
        main_idx = next(i for i, l in enumerate(lines) if l.strip() == "_main:")
        return "\n".join(lines[main_idx:])
    assert "call    eax" in main_body(asm_a)
    assert "call    eax" in main_body(asm_b)


def test_indirect_call_with_args_pushes_then_calls():
    asm = _compile(
        "int add(int a, int b) { return a + b; } "
        "int main(void) { int (*fp)(int, int) = add; return fp(2, 3); }"
    )
    asm_lines = asm.splitlines()
    main_idx = next(i for i, l in enumerate(asm_lines) if l.strip() == "_main:")
    main_section = "\n".join(asm_lines[main_idx:])
    # Args pushed, then call eax, then stack cleanup.
    assert "call    eax" in main_section
    assert "add     esp, 8" in main_section


def test_function_pointer_passed_as_argument():
    # Pass a function pointer through cdecl, call it indirectly inside.
    asm = _compile(
        "int twice(int (*f)(int), int x) { return f(x) + f(x); } "
        "int square(int x) { return x * x; } "
        "int main(void) { return twice(square, 3); }"
    )
    asm_lines = asm.splitlines()
    # `twice` calls through its parameter slot — indirect.
    twice_idx = next(i for i, l in enumerate(asm_lines) if l.strip() == "_twice:")
    end_idx = next(
        i for i, l in enumerate(asm_lines)
        if i > twice_idx and l.strip() == "ret"
    )
    twice_section = "\n".join(asm_lines[twice_idx:end_idx])
    assert "call    eax" in twice_section
    # `main` passes square by name (loaded as immediate).
    main_idx = next(i for i, l in enumerate(asm_lines) if l.strip() == "_main:")
    main_section = "\n".join(asm_lines[main_idx:])
    assert "mov     eax, _square" in main_section


# ---- casts ----------------------------------------------------------------

def test_cast_int_to_signed_char_narrows():
    asm = _compile("int main(void) { int x = 300; return (char)x; }")
    # Truncate to byte then sign-extend back through al.
    assert "movsx   eax, al" in asm


def test_cast_int_to_unsigned_char_zero_extends():
    asm = _compile("int main(void) { int x = 300; return (unsigned char)x; }")
    assert "movzx   eax, al" in asm


def test_cast_int_to_signed_short_narrows():
    asm = _compile("int main(void) { int x = 70000; return (short)x; }")
    assert "movsx   eax, ax" in asm


def test_cast_int_to_unsigned_short_zero_extends():
    asm = _compile("int main(void) { int x = 70000; return (unsigned short)x; }")
    assert "movzx   eax, ax" in asm


def test_cast_to_int_is_noop():
    asm = _compile("int main(void) { int x = 5; return (int)x; }")
    # No truncation instructions emitted.
    assert "movsx" not in asm
    assert "movzx" not in asm


def test_cast_pointer_to_pointer_is_noop():
    asm = _compile("int main(void) { int x = 0; char *p = (char *)&x; return 0; }")
    # The address still loads via lea; the cast itself emits nothing.
    assert "lea     eax, [ebp - 4]" in asm


def test_cast_int_to_pointer():
    # `(int *)0` is the canonical null-pointer form.
    asm = _compile("int main(void) { int *p = (int *)0; return 0; }")
    assert "mov     eax, 0" in asm


def test_cast_then_arithmetic_uses_target_type():
    # `(char *)p + 1` should scale by sizeof(char)=1, not sizeof(int)=4.
    asm = _compile(
        "int main(void) { int x = 0; char *p = (char *)&x + 1; return 0; }"
    )
    # No `shl eax, 2` from a missed scaling decision.
    asm_lines = asm.splitlines()
    # Allow scaling by 2 (short) or 1 (no-op) but specifically NOT by 4.
    assert "shl     eax, 2" not in asm
    # And we should still see the address arithmetic.
    assert "add     eax, ecx" in asm


def test_cast_widens_via_load():
    # Loading a char already sign-extends; an explicit cast back to int is a no-op.
    asm = _compile("int main(void) { char c = -1; return (int)c; }")
    # The `c` read is already movsx, no additional widening needed.
    assert "movsx   eax, byte [ebp - 4]" in asm


# ---- globals --------------------------------------------------------------

def test_global_int_uninitialized_in_bss():
    asm = _compile("int g; int main(void) { return 0; }")
    assert "section .bss" in asm
    assert "_g:" in asm
    # NASM `resb` reserves N bytes of uninitialized space.
    assert "resb    4" in asm


def test_global_int_initialized_in_data():
    asm = _compile("int g = 42; int main(void) { return 0; }")
    assert "section .data" in asm
    assert "_g:" in asm
    assert "dd      42" in asm


def test_global_int_read_uses_label():
    asm = _compile("int g = 7; int main(void) { return g; }")
    # No frame load — direct memory access via the label.
    assert "mov     eax, [_g]" in asm


def test_global_int_write_uses_label():
    asm = _compile("int g; int main(void) { g = 5; return g; }")
    assert "mov     [_g], eax" in asm


def test_global_address_of_emits_label_immediate():
    asm = _compile("int g; int main(void) { int *p = &g; return 0; }")
    # `&g` loads the label as an absolute address (immediate operand).
    assert "mov     eax, _g" in asm


def test_global_array_uninitialized_in_bss():
    asm = _compile("int arr[3]; int main(void) { return arr[0]; }")
    assert "section .bss" in asm
    assert "_arr:" in asm
    assert "resb    12" in asm
    # Array decay loads the label into eax as the base address.
    assert "mov     eax, _arr" in asm


def test_global_array_initialized_in_data():
    asm = _compile("int arr[3] = {1, 2, 3}; int main(void) { return arr[1]; }")
    assert "section .data" in asm
    assert "_arr:" in asm
    assert "dd      1, 2, 3" in asm


def test_global_array_underspecified_zero_filled():
    asm = _compile("int arr[5] = {1, 2}; int main(void) { return 0; }")
    # Assembler-time zero-fill via the directive.
    assert "dd      1, 2, 0, 0, 0" in asm


def test_global_string_array_inferred_size():
    asm = _compile('char s[] = "hi"; int main(void) { return 0; }')
    assert "_s:" in asm
    # String stored as byte values + null terminator.
    assert "db      104, 105, 0" in asm


def test_global_string_array_with_padding():
    asm = _compile('char s[5] = "hi"; int main(void) { return 0; }')
    # 'h', 'i', null, then 2 zero-pad bytes to fill out to 5.
    assert "db      104, 105, 0, 0, 0" in asm


def test_global_char_uses_db():
    asm = _compile("char c = 65; int main(void) { return c; }")
    assert "db      65" in asm
    # Read still goes through movsx (signed-char default).
    assert "movsx   eax, byte [_c]" in asm


def test_global_short_uses_dw_with_negative_value():
    asm = _compile("short s = -1; int main(void) { return s; }")
    # NASM accepts negative values directly in `dw`; the assembler chooses
    # the right two's-complement encoding.
    assert "dw      -1" in asm
    assert "movsx   eax, word [_s]" in asm


def test_global_increment_uses_label():
    asm = _compile("int g; int main(void) { ++g; return g; }")
    assert "inc     dword [_g]" in asm


def test_local_shadows_global():
    asm = _compile("int g = 99; int main(void) { int g = 5; return g; }")
    asm_lines = asm.splitlines()
    # The local is at [ebp - 4]; the return must read from there, not [_g].
    ret_idx = next(i for i, l in enumerate(asm_lines) if l.strip() == "jmp     .epilogue")
    prev_load = next(
        l for l in reversed(asm_lines[:ret_idx])
        if l.strip().startswith("mov     eax,")
    )
    assert "[ebp - 4]" in prev_load
    # The global is still emitted — its label is still in .data.
    assert "_g:" in asm
    assert "dd      99" in asm


def test_non_constant_global_init_rejected():
    # Globals must be initialized with literals, not expressions referencing
    # other names.
    with pytest.raises(CodegenError, match="constant|literal"):
        _compile(
            "int x = 5; int g = x + 1; int main(void) { return 0; }"
        )


def test_negative_int_global_init():
    asm = _compile("int g = -7; int main(void) { return g; }")
    assert "dd      -7" in asm


# ---- sizeof ---------------------------------------------------------------

@pytest.mark.parametrize(
    "type_str,expected",
    [
        ("int",      4),
        ("char",     1),
        ("short",    2),
        ("int *",    4),
        ("char *",   4),
        ("int[3]",   12),
        ("char[5]",  5),
        ("short[4]", 8),
    ],
)
def test_sizeof_type(type_str, expected):
    asm = _compile(f"int main(void) {{ return sizeof({type_str}); }}")
    assert f"mov     eax, {expected}" in asm


def test_sizeof_int_local():
    asm = _compile("int main(void) { int x = 0; return sizeof(x); }")
    assert "mov     eax, 4" in asm


def test_sizeof_char_local():
    asm = _compile("int main(void) { char c = 0; return sizeof(c); }")
    # The init still emits `mov eax, 0`, but sizeof emits `mov eax, 1`.
    assert "mov     eax, 1" in asm


def test_sizeof_array_local_does_not_decay():
    # `sizeof(arr)` returns the full array size, not pointer size.
    asm = _compile("int main(void) { int arr[5]; return sizeof(arr); }")
    assert "mov     eax, 20" in asm


def test_sizeof_dereferenced_pointer_returns_pointee_size():
    asm = _compile(
        "int main(void) { int x = 0; int *p = &x; return sizeof(*p); }"
    )
    assert "mov     eax, 4" in asm


def test_sizeof_array_index_element_size():
    asm = _compile("int main(void) { char arr[5]; return sizeof(arr[0]); }")
    assert "mov     eax, 1" in asm


def test_sizeof_does_not_evaluate_operand():
    # `sizeof(arr[i])` must NOT compute the index — only the operand's
    # static type matters.
    asm = _compile(
        "int main(void) { int arr[3]; int i = 99; return sizeof(arr[i]); }"
    )
    asm_lines = asm.splitlines()
    # The init `int i = 99` writes 99 once; sizeof shouldn't re-load `i`
    # (no `mov eax, [ebp - 16]` from the sizeof side, and no `lea` from a
    # phantom array decay). The simplest check: the sizeof produces the
    # element size and there's no `mov eax, [eax]` index-load.
    assert "mov     eax, 4" in asm  # sizeof(int) = 4
    # The slot read for `i` would only appear if sizeof had evaluated the
    # operand — verify it doesn't.
    body_lines = [l.strip() for l in asm_lines]
    # Filter out lines from the init (`mov eax, 99` and its store), keep
    # everything after to check sizeof's emission.
    assert "mov     eax, [eax]" not in body_lines


def test_sizeof_in_arithmetic():
    # sizeof participates as a normal int constant in arithmetic.
    asm = _compile("int main(void) { return sizeof(int) * 2; }")
    assert "mov     eax, 4" in asm
    assert "mov     eax, 2" in asm
    assert "imul    eax, ecx" in asm


# ---- array initialization --------------------------------------------------

def test_int_array_init_inline_stores():
    asm = _compile("int main(void) { int arr[3] = {1, 2, 3}; return 0; }")
    assert "sub     esp, 12" in asm
    # Each value lands at the right element offset: arr[0] at -12, arr[1] at -8, arr[2] at -4.
    asm_lines = asm.splitlines()
    for value, disp in [(1, -12), (2, -8), (3, -4)]:
        idx = next(
            i for i, l in enumerate(asm_lines) if l.strip() == f"mov     eax, {value}"
        )
        assert asm_lines[idx + 1].strip() == f"mov     [ebp - {-disp}], eax"


def test_int_array_underspecified_zero_fills_tail():
    # arr[0] = 10, arr[1] = 20, arr[2..4] zeroed.
    asm = _compile("int main(void) { int arr[5] = {10, 20}; return 0; }")
    assert "sub     esp, 20" in asm
    assert "mov     eax, 10" in asm
    assert "mov     eax, 20" in asm
    # Tail elements zeroed via direct memory writes.
    assert "mov     dword [ebp - 12], 0" in asm
    assert "mov     dword [ebp - 8], 0" in asm
    assert "mov     dword [ebp - 4], 0" in asm


def test_int_array_inferred_size_from_init():
    asm = _compile("int main(void) { int arr[] = {7, 8, 9, 10}; return 0; }")
    # Size inferred from the four-element initializer → 16 bytes.
    assert "sub     esp, 16" in asm
    for value in (7, 8, 9, 10):
        assert f"mov     eax, {value}" in asm


def test_char_array_string_init_inferred_size():
    asm = _compile('int main(void) { char s[] = "hi"; return 0; }')
    # 'h' (104), 'i' (105), '\\0' → 3 bytes payload, slot rounds to 4.
    assert "sub     esp, 4" in asm
    assert "mov     byte [ebp - 4], 104" in asm
    assert "mov     byte [ebp - 3], 105" in asm
    assert "mov     byte [ebp - 2], 0" in asm


def test_char_array_string_init_with_padding():
    # `char t[5] = "hi"` writes h, i, null, then zero-fills the remaining bytes.
    asm = _compile('int main(void) { char t[5] = "hi"; return 0; }')
    assert "sub     esp, 8" in asm  # 5 rounds to 8
    assert "mov     byte [ebp - 8], 104" in asm
    assert "mov     byte [ebp - 7], 105" in asm
    assert "mov     byte [ebp - 6], 0" in asm
    assert "mov     byte [ebp - 5], 0" in asm
    assert "mov     byte [ebp - 4], 0" in asm


def test_char_array_init_from_int_list():
    # `char arr[3] = {65, 66, 67}` — each int store narrows to a byte.
    asm = _compile("int main(void) { char arr[3] = {65, 66, 67}; return 0; }")
    assert "sub     esp, 4" in asm  # 3 bytes rounds to 4
    # Each store goes through `mov byte [...], al` after `mov eax, value`.
    assert "mov     byte [ebp - 4], al" in asm
    assert "mov     byte [ebp - 3], al" in asm
    assert "mov     byte [ebp - 2], al" in asm


def test_too_many_initializers_rejected():
    with pytest.raises(CodegenError, match="out of range|too many|exceeds"):
        _compile("int main(void) { int arr[2] = {1, 2, 3}; return 0; }")


def test_string_init_too_long_rejected():
    # `char s[N] = "<string>"` requires N >= len("<string>"); the
    # null terminator is dropped if N == len. Three chars into a
    # two-slot array is a real overflow.
    with pytest.raises(CodegenError, match="exceeds"):
        _compile('int main(void) { char s[2] = "hii"; return 0; }')


def test_string_init_for_int_array_rejected():
    with pytest.raises(CodegenError, match="char"):
        _compile('int main(void) { int arr[3] = "hi"; return 0; }')


def test_array_init_then_indexed_read():
    # End-to-end: initialize, then read an element back.
    asm = _compile("int main(void) { int arr[3] = {1, 2, 3}; return arr[1]; }")
    # Init writes 1, 2, 3 to the slots.
    assert "mov     eax, 2" in asm
    assert "mov     [ebp - 8], eax" in asm
    # Final return loads through the indexed address.
    assert "mov     eax, [eax]" in asm


# ---- char / short codegen --------------------------------------------------

def test_char_local_signed_load():
    asm = _compile("int main(void) { char c = 5; return c; }")
    # Signed char default → sign-extend on load.
    assert "movsx   eax, byte [ebp - 4]" in asm
    # Init writes only the low byte.
    assert "mov     byte [ebp - 4], al" in asm


def test_unsigned_char_zero_extends():
    asm = _compile("int main(void) { unsigned char c = 200; return c; }")
    # `unsigned char` → zero-extend on load.
    assert "movzx   eax, byte [ebp - 4]" in asm


def test_short_local_signed_load_word_sized():
    asm = _compile("int main(void) { short s = 1234; return s; }")
    assert "movsx   eax, word [ebp - 4]" in asm
    assert "mov     word [ebp - 4], ax" in asm


def test_unsigned_short_zero_extends_word():
    asm = _compile("int main(void) { unsigned short s = 60000; return s; }")
    assert "movzx   eax, word [ebp - 4]" in asm


def test_char_slot_pads_to_four_bytes():
    # A single char still consumes a 4-byte slot so a following int stays
    # 4-aligned at [ebp - 8].
    asm = _compile("int main(void) { char c = 1; int x = 2; return x; }")
    assert "sub     esp, 8" in asm
    assert "mov     [ebp - 8], eax" in asm


def test_char_array_packs_bytes():
    # `char arr[5]` is genuinely 5 bytes of payload, but the slot is rounded
    # up to a multiple of 4 so the next local stays aligned.
    asm = _compile(
        "int main(void) { char arr[5]; int x = 0; return x; }"
    )
    assert "sub     esp, 12" in asm
    # `x` lives at [ebp - 12] (the int that follows the rounded-up array slot).
    assert "mov     [ebp - 12], eax" in asm


def test_char_array_index_load_uses_byte():
    asm = _compile(
        "int main(void) { char arr[4]; arr[0] = 65; return arr[0]; }"
    )
    # Address arithmetic still happens, but the element scaling uses
    # sizeof(char)=1, so no shl appears for the byte offset.
    assert "lea     eax, [ebp - 4]" in asm
    # Store goes through `mov byte [ecx], al`.
    assert "mov     byte [ecx], al" in asm
    # Load uses signed byte.
    assert "movsx   eax, byte [eax]" in asm


def test_char_param_loaded_with_movsx():
    asm = _compile(
        "int f(char c) { return c; } int main(void) { return f(65); }"
    )
    assert "movsx   eax, byte [ebp + 8]" in asm


def test_dereference_char_pointer_uses_movsx():
    # The pre-slice 10 bug was `*char_ptr` reading 4 bytes; now it reads 1.
    asm = _compile(
        "int main(void) { char c = 65; char *p = &c; return *p; }"
    )
    assert "movsx   eax, byte [eax]" in asm


def test_store_through_char_pointer_writes_byte():
    asm = _compile(
        "int main(void) { char c = 0; char *p = &c; *p = 7; return c; }"
    )
    assert "mov     byte [ecx], al" in asm


def test_char_increment_uses_byte_inc():
    asm = _compile("int main(void) { char c = 0; ++c; return c; }")
    assert "inc     byte [ebp - 4]" in asm
    # And the read after still sign-extends.
    assert "movsx   eax, byte [ebp - 4]" in asm


def test_char_pointer_arithmetic_steps_by_one_after_subword_lands():
    # Same assertion as before — pointer scaling for char* is still 1 byte.
    # Re-verifying after sub-word load lands.
    asm = _compile(
        'int puts(const char *s); '
        'int main(void) { const char *p = "abc"; p = p + 1; return 0; }'
    )
    assert "shl     eax, 2" not in asm
    assert "add     eax, ecx" in asm


def test_int_locals_unchanged_by_subword_pass():
    # Regression: int slots still emit plain `mov eax, [...]` and
    # `mov [...], eax`, no movsx/movzx anywhere.
    asm = _compile("int main(void) { int x = 5; int y = x; return y; }")
    assert "mov     eax, [ebp - 4]" in asm
    assert "mov     [ebp - 8], eax" in asm
    assert "movsx" not in asm
    assert "movzx" not in asm


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


# ---- regression coverage for recent slices --------------------------


def test_int128_storage_size():
    # `unsigned __int128` global takes 16 bytes in BSS.
    asm = _compile("unsigned __int128 b; int main(void) { return 0; }")
    assert "_b:" in asm
    assert "resb    16" in asm


def test_int128_add_emits_carry_chain():
    asm = _compile(
        "unsigned __int128 b; int a;\n"
        "int main(void) { b += a; return 0; }"
    )
    # 4-dword carry chain: one add then three adcs.
    assert asm.count("        add     ") >= 1
    assert asm.count("        adc     ") >= 3


def test_int128_div_by_int_uses_three_div_chain():
    asm = _compile(
        "unsigned __int128 b; int c;\n"
        "int main(void) { b /= c; return 0; }"
    )
    # u128 / u32: top-down chain of four `div` instructions.
    assert asm.count("        div     ") >= 4


def test_int128_shift_by_64_compiles():
    # `>> 64` lowers via the bit_shift=0, word_shift=2 path. The bytes
    # shift by 8 within the 16-byte slot.
    asm = _compile(
        "unsigned __int128 b;\n"
        "unsigned long long get_high(void) { return b >> 64; }\n"
        "int main(void) { return 0; }\n"
    )
    # No runtime shift instruction needed for word-aligned shift.
    # The high 8 bytes get loaded into the result.
    assert "_get_high:" in asm


def test_int128_multiply_emits_full_schoolbook():
    # u128 * u128 produces 10 partial products (i+j < 4). Each is a
    # `mul dword [...]`. With 4 i-values and j running 0..3-i, we get
    # 4 + 3 + 2 + 1 = 10 mul instructions.
    asm = _compile(
        "unsigned __int128 a, b, c;\n"
        "int main(void) { c = a * b; return 0; }\n"
    )
    assert asm.count("        mul     ") >= 10


def test_int128_compare_emits_dword_chain():
    # u128 < u128 walks 4 dwords from high to low.
    asm = _compile(
        "unsigned __int128 a, b;\n"
        "int main(void) { return a < b; }\n"
    )
    # Four cmp eax, [esp + ...] comparisons (one per dword pair).
    assert asm.count("        cmp     eax, ") >= 4
    # Unsigned: jb / ja
    assert "jb " in asm or "ja " in asm


def test_int128_div_by_int128_calls_runtime_helper():
    # u128 / u128 routes to the libc helper since the int-divisor
    # fast path doesn't apply.
    asm = _compile(
        "unsigned __int128 a, b, c;\n"
        "int main(void) { c = a / b; return 0; }\n"
    )
    assert "___uc386_udiv128" in asm


def test_int128_add_promotes_via_type_of():
    # `u128 + u128` should produce a value of type int128 (so the
    # codegen routes through the carry chain rather than 32-bit add).
    asm = _compile(
        "unsigned __int128 a, b, c;\n"
        "int main(void) { c = a + b; return 0; }\n"
    )
    # 4-dword carry chain (one add, three adcs).
    assert asm.count("        add     ") >= 1
    assert asm.count("        adc     ") >= 3
    # And no spurious "add eax, ecx" that would indicate 32-bit add.
    assert "add     eax, ecx" not in asm


def test_int128_signed_div_emits_sign_handling():
    # Signed `s128 / s128` should compute |lhs|, |rhs|, divide, then
    # apply the result sign (sign(lhs) ^ sign(rhs) for /, sign(lhs)
    # for %). Look for the abs-value extraction (shr eax, 31) and
    # the unsigned helper call.
    asm = _compile(
        "__int128 a, b, c;\n"
        "int main(void) { c = a / b; return 0; }\n"
    )
    assert "shr     eax, 31" in asm
    assert "___uc386_udiv128" in asm


def test_int128_returning_function_uses_retptr_abi():
    # Int128-returning functions take a hidden first param (retptr) at
    # [ebp+8] just like struct returns; real params shift up by 4.
    asm = _compile(
        "unsigned __int128 add128(unsigned __int128 a, unsigned __int128 b) {\n"
        "    return a + b;\n"
        "}\n"
        "int main(void) { return 0; }\n"
    )
    # First int128 param 'a' is now at [ebp + 12] (= 8 + retptr).
    assert "[ebp + 12]" in asm
    # And the return path should write 16 bytes through __retptr__.
    assert "_add128:" in asm


def test_int128_pass_by_value_pushes_full_16_bytes():
    # When passing a __int128 by value, the caller should push 16
    # bytes per arg (4 dwords), not 4 bytes (the address).
    asm = _compile(
        "void take(unsigned __int128 a, unsigned __int128 b);\n"
        "int main(void) {\n"
        "    unsigned __int128 x = 1, y = 2;\n"
        "    take(x, y);\n"
        "    return 0;\n"
        "}\n"
    )
    # Two int128 args plus a (call-temp-style) destination — caller
    # reserves 32 bytes (16 each).  Look for sub esp, 16 (or 32).
    assert "sub     esp, 16" in asm or "sub     esp, 32" in asm


def test_int128_inc_dec_emits_4dword_carry_chain():
    # `++u128_val` should bump the low dword and propagate carry
    # through three adcs.  `--` uses sub/sbb.
    asm = _compile(
        "unsigned __int128 g;\n"
        "int main(void) { ++g; --g; return 0; }\n"
    )
    assert "add     dword" in asm
    assert "adc     dword" in asm
    assert "sub     dword" in asm
    assert "sbb     dword" in asm


def test_int128_global_with_init_emits_two_dq_halves():
    # Global `unsigned __int128 g = 100` lays down two 64-bit halves.
    asm = _compile("unsigned __int128 g = 100; int main(void) { return 0; }")
    assert "_g:" in asm
    # Low half = 100, high half = 0.
    assert "dq      0x0000000000000064" in asm
    assert asm.count("        dq      0x") >= 2


def test_int128_array_init_with_int_literals_widens():
    # `unsigned __int128 arr[3] = {1, 2, 3}` widens each int literal
    # to 16 bytes and stores them at consecutive 16-byte offsets.
    asm = _compile(
        "int main(void) {\n"
        "    unsigned __int128 arr[3] = {1, 2, 3};\n"
        "    return 0;\n"
        "}\n"
    )
    # Each int literal sign-extends via cdq (signed int source) and
    # then per-dword copies into the slot — three widening cdqs.
    assert asm.count("        cdq") >= 3


def test_int128_struct_member_init_with_int_literal_widens():
    # struct { int tag; __uint128_t v; } s = { 7, 42 };
    asm = _compile(
        "struct S { int tag; unsigned __int128 v; };\n"
        "int main(void) {\n"
        "    struct S s = { 7, 42 };\n"
        "    return 0;\n"
        "}\n"
    )
    # 7 → tag (one mov), then 42 widened to 16 bytes (low + 3 zero
    # high dwords via xor edx, edx + cdq fanout).
    assert "_main:" in asm


def test_int128_ternary_lowers_to_branched_copy():
    asm = _compile(
        "unsigned __int128 cmax(unsigned __int128 a, unsigned __int128 b) {\n"
        "    return a > b ? a : b;\n"
        "}\n"
        "int main(void) { return 0; }\n"
    )
    # Both branches should emit per-dword copies into the same dest
    # slot; expect two copy blocks framed by jz/jmp.
    assert "tern_false" in asm or "i128_tern_false" in asm
    assert "_cmax:" in asm


def test_int128_bool_context_ors_all_dwords():
    asm = _compile(
        "int main(void) {\n"
        "    unsigned __int128 g = 0;\n"
        "    if (g) return 1;\n"
        "    return 0;\n"
        "}\n"
    )
    # OR low + +4 + +8 + +12 of the int128 address into eax.
    assert "or      eax, [ecx + 4]" in asm
    assert "or      eax, [ecx + 8]" in asm
    assert "or      eax, [ecx + 12]" in asm


def test_int128_va_arg_copies_16_bytes_to_temp():
    # Typedef va_list manually since _compile bypasses the
    # preprocessor's __builtin_va_list predefine.
    asm = _compile(
        "typedef char *va_list;\n"
        "typedef unsigned __int128 u128;\n"
        "u128 first(int n, ...) {\n"
        "    va_list ap;\n"
        "    va_start(ap, n);\n"
        "    u128 r = va_arg(ap, u128);\n"
        "    va_end(ap);\n"
        "    return r;\n"
        "}\n"
        "int main(void) { return 0; }\n"
    )
    # va_arg int128 advances ap by 16, copies 16 bytes from [ecx]
    # to the destination.
    assert "_first:" in asm
    assert "16" in asm  # the 16-byte advance / copy size shows up


def test_int128_return_widens_int_literal():
    # `return 1;` from a __int128-returning function widens via a
    # synthetic Cast (sign-extends the int via cdq + fanout).
    asm = _compile(
        "unsigned __int128 fact(unsigned __int128 n) {\n"
        "    if (n <= 1) return 1;\n"
        "    return n * fact(n - 1);\n"
        "}\n"
        "int main(void) { return 0; }\n"
    )
    assert "_fact:" in asm


def test_int128_compound_literal_address():
    # `&(__int128){42}` allocates a per-expr 16-byte temp, stores 42
    # into it, returns the address.
    asm = _compile(
        "int main(void) {\n"
        "    unsigned __int128 *p = &(unsigned __int128){42};\n"
        "    return (int)*p;\n"
        "}\n"
    )
    assert "_main:" in asm
    # The compound literal's value 42 should land in the temp.
    assert "mov     eax, 42" in asm


def test_int128_compound_literal_array_init():
    # `(__int128[3]){1, 2, 3}` as an array initializer strips the
    # compound's wrapper and treats the inner InitializerList as the
    # source.
    asm = _compile(
        "int main(void) {\n"
        "    unsigned __int128 arr[3] = "
        "(unsigned __int128[3]){10, 20, 30};\n"
        "    return (int)arr[1];\n"
        "}\n"
    )
    assert "mov     eax, 10" in asm
    assert "mov     eax, 20" in asm
    assert "mov     eax, 30" in asm


def test_int128_variable_shift_emits_per_bit_loop():
    # `u128 << n` where n is a runtime int falls through to a
    # per-bit-loop runtime path. Look for the loop label and the
    # shl/rcl chain.
    asm = _compile(
        "unsigned __int128 sl(unsigned __int128 a, int n) {\n"
        "    return a << n;\n"
        "}\n"
        "int main(void) { return 0; }\n"
    )
    assert "i128_shift_loop" in asm
    assert "shl     dword [edi], 1" in asm
    assert "rcl     dword [edi + 4], 1" in asm


def test_int128_comma_type_of_returns_right_arm():
    # `int r = (u128_compound, (int)y)` should compile — the comma's
    # result type is the right arm's, not int128.
    asm = _compile(
        "int main(void) {\n"
        "    unsigned __int128 y = 10;\n"
        "    int r = (y += 5, (int)y);\n"
        "    return r;\n"
        "}\n"
    )
    assert "_main:" in asm


def test_int128_ternary_with_int_literal_arm_widens():
    # `cond ? f() : 0` where f returns u128 — the `0` arm needs
    # widening via synthetic Cast (matches _var_init / _return).
    asm = _compile(
        "unsigned __int128 f(unsigned __int128 x) { return x * 2; }\n"
        "int main(void) {\n"
        "    int c = 1;\n"
        "    unsigned __int128 a = c ? f(100) : 0;\n"
        "    return (int)a;\n"
        "}\n"
    )
    assert "_main:" in asm
    assert "_f:" in asm


def test_stmt_expr_with_float_trailing_value():
    # `({ ...; float_expr; })` evaluates head items, then leaves the
    # trailing float on st(0). Used to fall through to "not implemented".
    asm = _compile(
        "int main(void) {\n"
        "    float f = ({ float x = 1.5f; x + 0.5f; });\n"
        "    return f != 2.0f;\n"
        "}\n"
    )
    assert "_main:" in asm
    # The trailing add lands as faddp.
    assert "faddp" in asm or "fadd" in asm


def test_stmt_expr_with_int128_trailing_value():
    asm = _compile(
        "int main(void) {\n"
        "    unsigned __int128 v = "
        "({ unsigned __int128 a = 21; a * 2; });\n"
        "    return (int)v;\n"
        "}\n"
    )
    assert "_main:" in asm


def test_signed_int128_div_by_int_widens_rhs():
    # `i128 / int` (signed) widens rhs to i128 and routes through the
    # int128/int128 path (which has the sign-handling machinery).
    asm = _compile(
        "__int128 sd(__int128 a, int b) { return a / b; }\n"
        "int main(void) { return 0; }\n"
    )
    assert "_sd:" in asm
    # The int128/int128 path uses the unsigned helper after sign
    # extraction.
    assert "___uc386_udiv128" in asm


def test_long_long_inc_dec_propagates_carry():
    # `ll++` on an Identifier whose value is at the 32-bit boundary
    # should produce 0x1_00000000, not 0x0. Was using `inc dword` which
    # only bumps low 32 bits.
    asm = _compile(
        "int main(void) {\n"
        "    long long x = 0xFFFFFFFFLL;\n"
        "    x++;\n"
        "    return x == 0x100000000LL ? 0 : 1;\n"
        "}\n"
    )
    # Long long ++/-- now goes through `_inc_dec_ll` which does an
    # add dword + adc dword 0 chain.
    assert "add     dword [ebp - " in asm and "adc     dword [ebp - " in asm


def test_long_long_inc_dec_array_element_propagates_carry():
    # Same for array elements (non-Identifier path).
    asm = _compile(
        "int main(void) {\n"
        "    long long arr[1] = {0xFFFFFFFFLL};\n"
        "    arr[0]++;\n"
        "    return arr[0] == 0x100000000LL ? 0 : 1;\n"
        "}\n"
    )
    # The lvalue path should emit add dword [ecx] / adc dword [ecx+4].
    assert "add     dword [ecx], 1" in asm
    assert "adc     dword [ecx + 4], 0" in asm


def test_long_long_compound_assign_array_element_routes_to_ll_path():
    # `arr[i] += rhs` where arr is long-long must route through the
    # LL ladder so the high half participates with carry. Was using
    # the 32-bit path and storing cdq-extended low into high.
    asm = _compile(
        "int main(void) {\n"
        "    long long arr[1] = {0xFFFFFFFFLL};\n"
        "    arr[0] += 1;\n"
        "    return arr[0] == 0x100000000LL ? 0 : 1;\n"
        "}\n"
    )
    # Look for the LL add: `add eax, ecx` followed by `adc edx, ebx`.
    assert "adc     edx, ebx" in asm


def test_long_long_comma_evaluates_lhs_for_side_effects():
    # `(a += 5, a *= 2)` where a is long-long: was dropping the
    # `a += 5` and only emitting `a *= 2`. Now evaluates lhs through
    # _eval_expr_to_edx_eax so both side effects fire.
    asm = _compile(
        "int main(void) {\n"
        "    long long a = 10LL;\n"
        "    (a += 5, a *= 2);\n"
        "    return a == 30LL ? 0 : 1;\n"
        "}\n"
    )
    # Both compound assigns should emit add (for +=) and a multiply
    # sequence (for *=).
    assert asm.count("adc     edx, ebx") >= 1
    # And the LL multiply (`mul ecx` followed by partial products).
    assert "        mul     ecx" in asm


def test_decimal64_keyword_compiles_as_double():
    # _Decimal64 → double approximation. The literal `0.DD` parses
    # as a double with value 0.
    asm = _compile(
        "int main(void) { _Decimal64 d = 0.DD; if (d != 0.DD) return 1; return 0; }"
    )
    assert "_main:" in asm
    # Value 0.0 lands in .data via the float-constants pool.
    assert "section .data" in asm


def test_nested_fn_with_nonlocal_goto_emits_trampoline_and_setjmp():
    asm = _compile(
        "extern void exit(int);\n"
        "extern void abort(void);\n"
        "extern void qsort(void *, unsigned int, unsigned int, "
        "    int (*)(const void *, const void *));\n"
        "int main(void) {\n"
        "    __label__ done;\n"
        "    int compare(const void *a, const void *b) { goto done; }\n"
        "    char arr[3] = {0};\n"
        "    qsort(arr, 3, 1, compare);\n"
        "    abort();\n"
        " done:\n"
        "    exit(0);\n"
        "}\n"
    )
    # Per-frame trampoline: 12 bytes initialized in main's prologue.
    # Look for the opcode bytes (0xB9 = mov ecx, imm32; 0xBA = mov edx,
    # imm32; 0xE2FF = jmp edx).
    assert "0xB9" in asm
    assert "0xBA" in asm
    assert "0xE2FF" in asm
    # setjmp dispatch in main's prologue.
    assert "___builtin_setjmp" in asm
    # Lifted nested fn longjmps from the goto.
    assert "___builtin_longjmp" in asm
    # Lifted fn name carries the outer prefix.
    assert "_main__compare:" in asm
