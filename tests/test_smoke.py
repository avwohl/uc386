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


def test_array_initialization_not_yet_supported():
    with pytest.raises(CodegenError, match="initializ"):
        _compile("int main(void) { int a[3] = {1, 2, 3}; return 0; }")


def test_unsized_array_local_rejected():
    with pytest.raises(CodegenError, match="size"):
        _compile("int main(void) { int a[]; return 0; }")


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
