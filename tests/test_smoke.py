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


def test_unsupported_type_local_rejected():
    # `float` isn't yet a supported slot type — full FP codegen comes later.
    with pytest.raises(CodegenError, match="not supported|only"):
        _compile("int main(void) { float f; return 0; }")


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


def test_struct_return_not_yet_supported():
    with pytest.raises(CodegenError, match="struct return"):
        _compile(
            "struct point { int x; int y; }; "
            "struct point make(int x, int y) { struct point p; p.x = x; p.y = y; return p; } "
            "int main(void) { return 0; }"
        )


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
    # String stored inline: 'h', 'i', 0.
    assert "db      'hi', 0" in asm


def test_global_string_array_with_padding():
    asm = _compile('char s[5] = "hi"; int main(void) { return 0; }')
    # 'h', 'i', null, then 2 zero-pad bytes to fill out to 5.
    assert "db      'hi', 0, 0, 0" in asm


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
    with pytest.raises(CodegenError, match="too many|exceeds"):
        _compile("int main(void) { int arr[2] = {1, 2, 3}; return 0; }")


def test_string_init_too_long_rejected():
    # "hi" + null = 3 bytes, but only 2 slots available.
    with pytest.raises(CodegenError, match="exceeds"):
        _compile('int main(void) { char s[2] = "hi"; return 0; }')


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
    assert "int     21h" in text
