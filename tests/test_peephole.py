"""Tests for the asm-level peephole optimizer."""

from uc386.peephole import PeepholeOptimizer, optimize


# ── P1: dead-after-terminator ────────────────────────────────────


def test_drops_dead_xor_after_jmp():
    """The dead `xor eax, eax` after `jmp .epilogue` is removed.
    The LIVE `mov eax, 0` before the jmp also gets rewritten to
    `xor eax, eax` (mov_zero_to_xor), so the only xor in the
    output is the rewritten one immediately before the jmp."""
    asm = (
        "_main:\n"
        "        mov     eax, 0\n"
        "        jmp     .epilogue\n"
        "        xor     eax, eax\n"
        ".epilogue:\n"
        "        ret\n"
    )
    out = optimize(asm)
    # Exactly one `xor eax, eax` should remain — the rewritten
    # live one. The dead one after the jmp is dropped.
    assert out.count("xor     eax, eax") == 1
    # And the dead `mov eax, 0` doesn't appear (rewritten to xor).
    assert "mov     eax, 0" not in out
    assert ".epilogue:" in out  # Label preserved
    assert "ret" in out         # Real epilogue preserved


def test_drops_multiple_dead_instructions():
    asm = (
        "_f:\n"
        "        jmp     .end\n"
        "        mov     eax, 1\n"
        "        mov     ebx, 2\n"
        "        xor     ecx, ecx\n"
        ".end:\n"
        "        ret\n"
    )
    out = optimize(asm)
    assert "mov     eax, 1" not in out
    assert "mov     ebx, 2" not in out
    assert "xor     ecx, ecx" not in out
    assert ".end:" in out
    assert "ret" in out


def test_drops_dead_after_ret():
    asm = (
        "_f:\n"
        "        ret\n"
        "        mov     eax, 99\n"
        ".unused:\n"
        "        ret\n"
    )
    out = optimize(asm)
    # The mov between `ret` and `.unused:` is dead.
    assert "mov     eax, 99" not in out
    # The `.unused:` label still survives (we don't remove unreachable
    # labels — that's asm DCE's job).
    assert ".unused:" in out


def test_does_not_drop_after_conditional_jump():
    asm = (
        "_f:\n"
        "        test    eax, eax\n"
        "        jz      .skip\n"
        "        mov     eax, 1\n"
        ".skip:\n"
        "        ret\n"
    )
    out = optimize(asm)
    # `jz` is conditional — fall-through is live.
    assert "mov     eax, 1" in out


def test_preserves_data_after_terminator():
    """Data definitions in .data shouldn't be touched by P1."""
    asm = (
        "_f:\n"
        "        ret\n"
        "\n"
        "        section .data\n"
        "_str:\n"
        "        db      'hi', 0\n"
    )
    out = optimize(asm)
    assert "_str:" in out
    assert "db      'hi', 0" in out


def test_preserves_label_directly_after_terminator():
    asm = (
        "_f:\n"
        "        jmp     .target\n"
        ".target:\n"
        "        ret\n"
    )
    # No dead instructions to drop, but the jmp-to-next-label pass
    # will remove the redundant jmp.
    out = optimize(asm)
    assert ".target:" in out
    assert "ret" in out


# ── P-jmp-to-next: redundant `jmp X; X:` ─────────────────────────


def test_drops_jmp_to_next_label():
    asm = (
        "_f:\n"
        "        mov     eax, 0\n"
        "        jmp     .epilogue\n"
        ".epilogue:\n"
        "        ret\n"
    )
    out = optimize(asm)
    assert "jmp     .epilogue" not in out
    assert ".epilogue:" in out
    assert "ret" in out


def test_keeps_jmp_to_distant_label():
    asm = (
        "_f:\n"
        "        jmp     .far\n"
        ".near:\n"
        "        ret\n"
        ".far:\n"
        "        ret\n"
    )
    out = optimize(asm)
    # The jmp goes to .far, but the next label is .near. Don't drop.
    assert "jmp     .far" in out


def test_keeps_indirect_jmp():
    asm = (
        "_f:\n"
        "        jmp     eax\n"
        ".epilogue:\n"
        "        ret\n"
    )
    out = optimize(asm)
    # Indirect jmp can target anywhere — never elide.
    assert "jmp     eax" in out


def test_keeps_jmp_through_memory():
    asm = (
        "_f:\n"
        "        jmp     [_table + eax*4]\n"
        ".epilogue:\n"
        "        ret\n"
    )
    out = optimize(asm)
    assert "jmp     [_table + eax*4]" in out


# ── Statistics ───────────────────────────────────────────────────


def test_stats_count_dead_after_terminator():
    asm = (
        "_f:\n"
        "        jmp     .end\n"
        "        mov     eax, 1\n"
        "        mov     ebx, 2\n"
        ".end:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("dead_after_terminator", 0) == 2


def test_stats_count_jmp_to_next():
    asm = (
        "_f:\n"
        "        jmp     .a\n"
        ".a:\n"
        "        jmp     .b\n"
        ".b:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("jmp_to_next_label", 0) >= 2


# ── Combined: real-world end-of-function pattern ────────────────


def test_real_world_function_end():
    """Every function ends with this pattern when the last statement
    is `return X;`. All three of dead-after-jmp, jmp-to-next, and
    leave-collapse should fire."""
    asm = (
        "_f:\n"
        "        push    ebp\n"
        "        mov     ebp, esp\n"
        "        mov     eax, 42\n"
        "        jmp     .epilogue\n"
        "        xor     eax, eax\n"
        ".epilogue:\n"
        "        mov     esp, ebp\n"
        "        pop     ebp\n"
        "        ret\n"
    )
    out = optimize(asm)
    # Dead xor + redundant jmp gone.
    assert "xor     eax, eax" not in out
    assert "jmp     .epilogue" not in out
    # Epilogue collapsed to leave + ret.
    assert "        leave" in out
    assert "mov     esp, ebp" not in out
    assert "pop     ebp" not in out
    # Real return value + label still there.
    assert "mov     eax, 42" in out
    assert ".epilogue:" in out
    assert "ret" in out


# ── binop_collapse ───────────────────────────────────────────────


def test_binop_collapse_with_immediate():
    """The canonical binop right-operand pattern collapses, and the
    follow-up imm_binop_collapse folds the `mov ecx, 1; cmp eax, ecx`
    into a single `cmp eax, 1`."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        push    eax\n"
        "        mov     eax, 1\n"
        "        mov     ecx, eax\n"
        "        pop     eax\n"
        "        cmp     eax, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # The 4 stack-machine lines collapse to one `mov ecx, 1`,
    # which is then folded into the cmp's immediate operand.
    assert "push    eax" not in out
    assert "mov     ecx, eax" not in out
    assert "pop     eax" not in out
    assert "mov     ecx, 1" not in out
    assert "cmp     eax, 1" in out
    assert opt.stats.get("binop_collapse", 0) == 1
    assert opt.stats.get("imm_binop_collapse", 0) == 1


def test_binop_collapse_with_memory_operand():
    """The 4-line collapse fires; then imm_binop_collapse (now also
    accepting memory sources) folds `mov ecx, [...]; add eax, ecx`
    into `add eax, [...]`."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp - 4]\n"
        "        push    eax\n"
        "        mov     eax, [ebp - 8]\n"
        "        mov     ecx, eax\n"
        "        pop     eax\n"
        "        add     eax, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # The intermediate `mov ecx, [ebp - 8]` gets folded into the add.
    assert "mov     ecx, [ebp - 8]" not in out
    assert "add     eax, [ebp - 8]" in out
    assert opt.stats.get("binop_collapse", 0) == 1
    assert opt.stats.get("imm_binop_collapse", 0) == 1


def test_binop_collapse_skips_esp_relative_source():
    """`push eax` shifts ESP, so a subsequent `[esp + N]` read targets
    a different byte after collapse. Both binop_collapse and the
    later right_operand_retarget must skip — dropping the push/pop
    scaffold changes which byte the [esp + N] reads."""
    asm = (
        "_f:\n"
        "        push    eax\n"
        "        mov     eax, [esp + 4]\n"
        "        mov     ecx, eax\n"
        "        pop     eax\n"
        "        cmp     eax, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # Neither pattern may fire.
    assert opt.stats.get("binop_collapse", 0) == 0
    assert opt.stats.get("right_operand_retarget", 0) == 0
    assert "push    eax" in out
    assert "pop     eax" in out


def test_store_collapse_skips_esp_relative_source():
    asm = (
        "_f:\n"
        "        mov     eax, [ebp - 4]\n"
        "        push    eax\n"
        "        mov     eax, [esp + 8]\n"
        "        pop     ecx\n"
        "        mov     [ecx], eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("store_collapse", 0) == 0


def test_binop_collapse_does_not_fire_on_store_pattern():
    """The store pattern uses `pop ecx` (not `mov ecx, eax; pop eax`),
    so binop_collapse must skip it."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp - 4]\n"
        "        push    eax\n"
        "        mov     eax, 100\n"
        "        pop     ecx\n"
        "        mov     [ecx], eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # binop_collapse should NOT have fired (no mov ecx, eax marker).
    assert opt.stats.get("binop_collapse", 0) == 0


def test_binop_collapse_handles_chain_of_binops():
    """Two adjacent binops should both collapse."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        push    eax\n"
        "        mov     eax, 1\n"
        "        mov     ecx, eax\n"
        "        pop     eax\n"
        "        add     eax, ecx\n"
        "        push    eax\n"
        "        mov     eax, 2\n"
        "        mov     ecx, eax\n"
        "        pop     eax\n"
        "        add     eax, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("binop_collapse", 0) == 2


# ── store_collapse ───────────────────────────────────────────────


def test_store_collapse_with_immediate():
    """Store-through-pointer with literal value drops push/pop pair."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp - 4]\n"
        "        push    eax\n"
        "        mov     eax, 42\n"
        "        pop     ecx\n"
        "        mov     [ecx], eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "push    eax" not in out
    assert "pop     ecx" not in out
    # The value-load and store survive.
    assert "mov     eax, 42" in out
    assert "mov     [ecx], eax" in out
    # And we picked up a `mov ecx, eax` to save the address.
    assert "mov     ecx, eax" in out
    assert opt.stats.get("store_collapse", 0) == 1


def test_store_collapse_skips_when_src_reads_ecx():
    """If the value-load reads ECX, collapsing would change semantics
    (the new ECX value would be the saved address rather than caller-
    state)."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp - 4]\n"
        "        push    eax\n"
        "        mov     eax, [ecx]\n"
        "        pop     ecx\n"
        "        mov     [ecx], eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("store_collapse", 0) == 0


def test_store_collapse_does_not_fire_on_binop_pattern():
    """The binop pattern uses `mov ecx, eax; pop eax` (not `pop ecx;
    mov [ecx], eax`), so store_collapse must skip it."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp - 4]\n"
        "        push    eax\n"
        "        mov     eax, 1\n"
        "        mov     ecx, eax\n"
        "        pop     eax\n"
        "        add     eax, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("store_collapse", 0) == 0


# ── leave_collapse ───────────────────────────────────────────────


def test_leave_collapse_basic():
    asm = (
        "_f:\n"
        "        push    ebp\n"
        "        mov     ebp, esp\n"
        "        mov     eax, 42\n"
        "        mov     esp, ebp\n"
        "        pop     ebp\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "        leave" in out
    assert "mov     esp, ebp" not in out
    assert "pop     ebp" not in out
    # The prologue's `push ebp; mov ebp, esp` is left alone.
    assert "push    ebp" in out
    assert "mov     ebp, esp" in out
    assert opt.stats.get("leave_collapse", 0) == 1


def test_leave_collapse_does_not_fire_without_pop_ebp():
    asm = (
        "_f:\n"
        "        mov     esp, ebp\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("leave_collapse", 0) == 0


def test_leave_collapse_with_intervening_blank():
    asm = (
        "_f:\n"
        "        mov     esp, ebp\n"
        "\n"
        "        pop     ebp\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "        leave" in out
    assert opt.stats.get("leave_collapse", 0) == 1


# ── imm_store_collapse ───────────────────────────────────────────


def test_imm_store_collapse_zero():
    """`mov eax, 0; mov [...], eax; mov eax, 1` collapses the first
    two lines into a direct memory-immediate store."""
    asm = (
        "_f:\n"
        "        mov     eax, 0\n"
        "        mov     [ebp - 4], eax\n"
        "        mov     eax, 1\n"
        "        mov     [ebp - 8], eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # First store collapses (line C = `mov eax, 1` is the witness).
    assert "        mov     dword [ebp - 4], 0" in out
    # Second store doesn't collapse (no witness EAX overwrite after).
    # That's fine — the witness requirement is the safety guarantee.
    assert opt.stats.get("imm_store_collapse", 0) >= 1


def test_imm_store_collapse_label_source():
    """A label-immediate (e.g., a string literal address) is also a
    valid constant for the direct memory store form."""
    asm = (
        "_f:\n"
        "        mov     eax, _str0\n"
        "        mov     [ebp - 4], eax\n"
        "        mov     eax, 0\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "        mov     dword [ebp - 4], _str0" in out
    assert opt.stats.get("imm_store_collapse", 0) == 1


def test_imm_store_collapse_skips_memory_source():
    """`mov eax, [src]; mov [dst], eax` is NOT a `mem-imm` candidate
    because mem-mem moves don't exist on x86. NASM would reject the
    rewrite."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp - 4]\n"
        "        mov     [ebp - 8], eax\n"
        "        mov     eax, 0\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("imm_store_collapse", 0) == 0


def test_imm_store_collapse_requires_eax_overwrite_witness():
    """Without a `mov eax, X` after the store, EAX might be live
    (e.g., the assignment expression's value is being used). Skip."""
    asm = (
        "_f:\n"
        "        mov     eax, 0\n"
        "        mov     [ebp - 4], eax\n"
        "        cmp     eax, 0\n"  # reads EAX!
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("imm_store_collapse", 0) == 0


def test_imm_store_collapse_skips_register_source():
    """`mov eax, ecx` to a temporary EAX shouldn't be folded — ECX
    might be live and the immediate-form mov can't take a register."""
    asm = (
        "_f:\n"
        "        mov     eax, ecx\n"
        "        mov     [ebp - 4], eax\n"
        "        mov     eax, 0\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("imm_store_collapse", 0) == 0


def test_imm_store_collapse_skips_self_rmw_witness():
    """Regression: `mov eax, [eax]` is a self-RMW (reads EAX before
    writing). It cannot serve as the EAX-overwrite witness for
    imm_store_collapse, because dropping the original `mov eax, IMM`
    would change what address `[eax]` reads.

    Bug observed in c-testsuite 00163: `int *b = &(struct.b);
    printf("%d", *b);` produced this 3-line pattern after
    label_offset_fold + store_load_collapse, and imm_store_collapse
    was incorrectly dropping the IMM load.
    """
    asm = (
        "_f:\n"
        "        mov     eax, _g + 4\n"
        "        mov     [ebp - 8], eax\n"
        "        mov     eax, [eax]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # The IMM load must survive — `mov eax, [eax]` reads EAX.
    assert "mov     eax, _g + 4" in out
    assert opt.stats.get("imm_store_collapse", 0) == 0


# ── setcc_jcc_collapse ───────────────────────────────────────────


def test_setcc_jcc_collapse_setle_jz():
    """setle + jz means "jump if NOT (<=)", i.e., jump if strictly
    greater. Emit `jg`."""
    asm = (
        "_f:\n"
        "        cmp     eax, ecx\n"
        "        setle    al\n"
        "        movzx   eax, al\n"
        "        test    eax, eax\n"
        "        jz      .target\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "        jg     " in out
    assert ".target" in out
    assert "setle" not in out
    assert "movzx" not in out
    assert "test" not in out
    assert "jz" not in out
    assert opt.stats.get("setcc_jcc_collapse", 0) == 1


def test_setcc_jcc_collapse_setne_jz():
    """setne + jz means "jump if NOT (!=)", i.e., jump if equal.
    Emit `je`."""
    asm = (
        "_f:\n"
        "        cmp     eax, ecx\n"
        "        setne    al\n"
        "        movzx   eax, al\n"
        "        test    eax, eax\n"
        "        jz      .target\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "        je     " in out
    assert opt.stats.get("setcc_jcc_collapse", 0) == 1


def test_setcc_jcc_collapse_setl_jnz():
    """setl + jnz means "jump if (< was true)", i.e., jump if strictly
    less. Emit `jl` directly (no inversion)."""
    asm = (
        "_f:\n"
        "        cmp     eax, ecx\n"
        "        setl    al\n"
        "        movzx   eax, al\n"
        "        test    eax, eax\n"
        "        jnz     .target\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "        jl     " in out
    assert opt.stats.get("setcc_jcc_collapse", 0) == 1


def test_setcc_jcc_collapse_unsigned():
    """Unsigned conditions (a/ae/b/be) work the same way."""
    asm = (
        "_f:\n"
        "        cmp     eax, ecx\n"
        "        seta    al\n"
        "        movzx   eax, al\n"
        "        test    eax, eax\n"
        "        jz      .target\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # seta + jz → "jump if NOT >", i.e., be (below or equal).
    assert "        jbe    " in out
    assert opt.stats.get("setcc_jcc_collapse", 0) == 1


def test_setcc_jcc_collapse_does_not_fire_if_eax_used_between():
    """If anything between the setCC and jz reads EAX (other than the
    movzx/test we recognize), the boolean might be needed for more
    than just the branch — skip."""
    asm = (
        "_f:\n"
        "        cmp     eax, ecx\n"
        "        setl    al\n"
        "        movzx   eax, al\n"
        "        mov     [ebp - 4], eax\n"  # stores the bool — needed!
        "        test    eax, eax\n"
        "        jz      .target\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("setcc_jcc_collapse", 0) == 0


# ── push_immediate ───────────────────────────────────────────────


def test_push_immediate_label():
    """The canonical printf-style arg push: mov eax, _str; push eax;
    call _printf — collapses to push _str; call _printf."""
    asm = (
        "_main:\n"
        "        mov     eax, _uc386_str0\n"
        "        push    eax\n"
        "        call    _printf\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "push    _uc386_str0" in out
    assert "mov     eax, _uc386_str0" not in out
    assert opt.stats.get("push_immediate", 0) == 1


def test_push_immediate_literal():
    asm = (
        "_main:\n"
        "        mov     eax, 42\n"
        "        push    eax\n"
        "        call    _f\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "push    42" in out
    assert opt.stats.get("push_immediate", 0) == 1


def test_push_immediate_chains_for_multi_arg_call():
    """`printf("%d %d", x, y)` pushes 3 args — all should collapse."""
    asm = (
        "_main:\n"
        "        mov     eax, 5\n"
        "        push    eax\n"
        "        mov     eax, 10\n"
        "        push    eax\n"
        "        mov     eax, _str\n"
        "        push    eax\n"
        "        call    _printf\n"
        "        add     esp, 12\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("push_immediate", 0) == 3


def test_push_immediate_skips_when_eax_live_after_push():
    """If the next instruction reads EAX (e.g., add eax, ecx), the
    `mov eax, X` was loading EAX for use beyond just the push. Skip."""
    asm = (
        "_f:\n"
        "        mov     eax, 5\n"
        "        push    eax\n"
        "        add     eax, ecx\n"  # reads EAX!
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("push_immediate", 0) == 0


def test_push_immediate_skips_memory_source():
    """`mov eax, [ebp - 4]` is a memory load — the rewritten push
    would need NASM mem-imm push (which doesn't exist as a single
    instruction in the same form). Skip."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp - 4]\n"
        "        push    eax\n"
        "        call    _f\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("push_immediate", 0) == 0


# ── imm_binop_collapse ───────────────────────────────────────────


def test_imm_binop_collapse_cmp():
    """`mov ecx, IMM; cmp eax, ecx` → `cmp eax, IMM`."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp - 4]\n"
        "        mov     ecx, 50\n"
        "        cmp     eax, ecx\n"
        "        jle     .end\n"
        ".end:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     ecx, 50" not in out
    assert "cmp     eax, 50" in out
    assert opt.stats.get("imm_binop_collapse", 0) == 1


def test_imm_binop_collapse_add():
    """`mov ecx, 5; add eax, ecx` → `add eax, 5`."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp - 4]\n"
        "        mov     ecx, 5\n"
        "        add     eax, ecx\n"
        "        mov     [ebp - 4], eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "add     eax, 5" in out
    assert opt.stats.get("imm_binop_collapse", 0) == 1


def test_imm_binop_collapse_label():
    """Label-as-immediate also folds: `mov ecx, _glob; cmp eax, ecx`."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp - 4]\n"
        "        mov     ecx, _glob\n"
        "        cmp     eax, ecx\n"
        "        je      .skip\n"
        ".skip:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "cmp     eax, _glob" in out
    assert opt.stats.get("imm_binop_collapse", 0) == 1


def test_imm_binop_collapse_test():
    """`mov ecx, 0xff; test eax, ecx` → `test eax, 0xff`."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp - 4]\n"
        "        mov     ecx, 0xff\n"
        "        test    eax, ecx\n"
        "        jz      .skip\n"
        ".skip:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "test    eax, 0xff" in out
    assert opt.stats.get("imm_binop_collapse", 0) == 1


def test_imm_binop_collapse_imul():
    """`mov ecx, 3; imul eax, ecx` → `imul eax, 3` (NASM accepts the
    two-operand form for register*imm)."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp - 4]\n"
        "        mov     ecx, 3\n"
        "        imul    eax, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "imul    eax, 3" in out
    assert opt.stats.get("imm_binop_collapse", 0) == 1


def test_imm_binop_collapse_does_not_fire_if_ecx_read_after():
    """If the next instr reads ECX (e.g. `mov [ecx], eax`), keep the
    `mov ecx, IMM` so ECX still holds the right value."""
    asm = (
        "_f:\n"
        "        mov     ecx, 42\n"
        "        cmp     eax, ecx\n"
        "        mov     [ecx], eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     ecx, 42" in out
    assert opt.stats.get("imm_binop_collapse", 0) == 0


def test_imm_binop_collapse_memory_source():
    """`mov ecx, [ebp - 8]; cmp eax, ecx` folds to `cmp eax, [ebp - 8]`
    — NASM accepts a memory operand on the right side of cmp/add/etc.
    Saves 3 bytes (the prior `mov ecx, [...]` is gone)."""
    asm = (
        "_f:\n"
        "        mov     ecx, [ebp - 8]\n"
        "        cmp     eax, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "cmp     eax, [ebp - 8]" in out
    assert "mov     ecx, [ebp - 8]" not in out
    assert opt.stats.get("imm_binop_collapse", 0) == 1


def test_imm_binop_collapse_register_source():
    """`mov ecx, edx; cmp eax, ecx` folds to `cmp eax, edx`. The
    `edx` source isn't ECX/CX/CL/CH so it's safe."""
    asm = (
        "_f:\n"
        "        mov     ecx, edx\n"
        "        cmp     eax, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "cmp     eax, edx" in out
    assert opt.stats.get("imm_binop_collapse", 0) == 1


def test_imm_binop_collapse_skips_self_referential_ecx_source():
    """`mov ecx, [ecx + 4]; cmp eax, ecx` shouldn't fold — the
    rewrite would become `cmp eax, [ecx + 4]` but ECX has its
    new value, not its original value."""
    asm = (
        "_f:\n"
        "        mov     ecx, [ecx + 4]\n"
        "        cmp     eax, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("imm_binop_collapse", 0) == 0


def test_imm_binop_collapse_skips_when_cl_read_after():
    """Sub-register CL reads also block the rewrite. The original
    sequence wrote IMM-low-bits to CL; post-collapse, CL has whatever
    it had before. Different state → unsafe."""
    asm = (
        "_f:\n"
        "        mov     ecx, 3\n"
        "        add     eax, ecx\n"
        "        shl     eax, cl\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("imm_binop_collapse", 0) == 0


def test_imm_binop_collapse_witness_call_overwrites_ecx():
    """A `call` overwrites caller-saved registers (EAX, ECX, EDX) per
    cdecl, so the witness is satisfied."""
    asm = (
        "_f:\n"
        "        mov     ecx, 100\n"
        "        cmp     eax, ecx\n"
        "        call    _other\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "cmp     eax, 100" in out
    assert opt.stats.get("imm_binop_collapse", 0) == 1


# ── mov_zero_to_xor ──────────────────────────────────────────────


def test_mov_zero_to_xor_simple():
    """`mov eax, 0` followed by a store + ret rewrites to `xor`.
    The mov path through the store is flag-neutral; the ret ends
    the function so flags don't matter."""
    asm = (
        "_f:\n"
        "        mov     eax, 0\n"
        "        mov     [ebp - 4], eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     eax, 0" not in out
    assert "xor     eax, eax" in out
    assert opt.stats.get("mov_zero_to_xor", 0) == 1


def test_mov_zero_to_xor_followed_by_jmp_then_label_then_ret():
    """`mov eax, 0; jmp .L; .L: ret` — cross-jmp-and-label scan
    should still be safe (codegen invariant: flags are never
    assumed to carry across labels)."""
    asm = (
        "_f:\n"
        "        mov     eax, 0\n"
        "        jmp     .epilogue\n"
        ".epilogue:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     eax, 0" not in out
    assert "xor     eax, eax" in out
    assert opt.stats.get("mov_zero_to_xor", 0) == 1


def test_mov_zero_to_xor_skips_when_jcc_follows():
    """`mov eax, 0` followed by `jcc` is unsafe — the jcc reads
    flags from a previous cmp/test, but xor would clobber them.
    Don't rewrite."""
    asm = (
        "_f:\n"
        "        cmp     ebx, ecx\n"
        "        mov     eax, 0\n"
        "        je      .L\n"
        ".L:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("mov_zero_to_xor", 0) == 0


def test_mov_zero_to_xor_skips_when_setcc_follows():
    """`mov eax, 0; setcc al; ...` reads flags. Don't rewrite."""
    asm = (
        "_f:\n"
        "        cmp     ebx, ecx\n"
        "        mov     eax, 0\n"
        "        sete    al\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("mov_zero_to_xor", 0) == 0


def test_mov_zero_to_xor_safe_when_arithmetic_follows():
    """`mov eax, 0; add eax, ebx` — the add clobbers flags before
    any read. Safe."""
    asm = (
        "_f:\n"
        "        mov     eax, 0\n"
        "        add     eax, ebx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "xor     eax, eax" in out
    assert opt.stats.get("mov_zero_to_xor", 0) == 1


def test_mov_zero_to_xor_safe_when_call_follows():
    """`mov eax, 0; call X` — call clobbers flags via callee."""
    asm = (
        "_f:\n"
        "        mov     eax, 0\n"
        "        call    _bar\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "xor     eax, eax" in out
    assert opt.stats.get("mov_zero_to_xor", 0) == 1


def test_mov_zero_to_xor_other_registers():
    """The pattern fires for any 32-bit gp reg, not just EAX."""
    asm = (
        "_f:\n"
        "        mov     ebx, 0\n"
        "        mov     [ebp - 4], ebx\n"
        "        mov     ecx, 0\n"
        "        mov     [ebp - 8], ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     ebx, 0" not in out
    assert "mov     ecx, 0" not in out
    assert "xor     ebx, ebx" in out
    assert "xor     ecx, ecx" in out
    assert opt.stats.get("mov_zero_to_xor", 0) == 2


def test_mov_zero_to_xor_skips_nonzero_immediate():
    """`mov eax, 1` doesn't rewrite (xor eax, eax produces 0, not 1)."""
    asm = (
        "_f:\n"
        "        mov     eax, 1\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     eax, 1" in out
    assert opt.stats.get("mov_zero_to_xor", 0) == 0


# ── store_load_collapse ──────────────────────────────────────────


def test_store_load_collapse_local():
    """`mov [ebp - 4], eax; mov eax, [ebp - 4]` → just the store.
    Common after function-call result is stored to a local then
    immediately re-used in an expression."""
    asm = (
        "_f:\n"
        "        call    _bar\n"
        "        mov     [ebp - 4], eax\n"
        "        mov     eax, [ebp - 4]\n"
        "        cmp     eax, 45\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # Only one occurrence of the address; the load is gone.
    assert out.count("[ebp - 4]") == 1
    assert "mov     eax, [ebp - 4]" not in out
    assert "mov     [ebp - 4], eax" in out
    assert opt.stats.get("store_load_collapse", 0) == 1


def test_store_load_collapse_global():
    """Globals work the same way: `mov [_var], eax; mov eax, [_var]`
    → just the store."""
    asm = (
        "_f:\n"
        "        mov     [_counter], eax\n"
        "        mov     eax, [_counter]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     eax, [_counter]" not in out
    assert "mov     [_counter], eax" in out
    assert opt.stats.get("store_load_collapse", 0) == 1


def test_store_load_collapse_other_registers():
    """Pattern works for any 32-bit gp register, not just EAX."""
    asm = (
        "_f:\n"
        "        mov     [ebp - 4], ebx\n"
        "        mov     ebx, [ebp - 4]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     ebx, [ebp - 4]" not in out
    assert opt.stats.get("store_load_collapse", 0) == 1


def test_store_load_collapse_skips_different_register():
    """`mov [X], eax; mov ecx, [X]` shouldn't fire — different regs.
    The load is genuine: ECX gets a copy of the just-stored value."""
    asm = (
        "_f:\n"
        "        mov     [ebp - 4], eax\n"
        "        mov     ecx, [ebp - 4]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("store_load_collapse", 0) == 0


def test_store_load_collapse_skips_different_address():
    """`mov [X], eax; mov eax, [Y]` shouldn't fire — different addrs."""
    asm = (
        "_f:\n"
        "        mov     [ebp - 4], eax\n"
        "        mov     eax, [ebp - 8]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("store_load_collapse", 0) == 0


def test_store_load_collapse_skips_with_intervening_instr():
    """Anything between the store and the load means the load might
    be reading a different value than what was just stored."""
    asm = (
        "_f:\n"
        "        mov     [ebp - 4], eax\n"
        "        call    _other\n"
        "        mov     eax, [ebp - 4]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("store_load_collapse", 0) == 0


# ── right_operand_retarget ───────────────────────────────────────


def test_right_operand_retarget_chain_2():
    """`push eax; mov eax, A; add eax, B; mov ecx, eax; pop eax` →
    `mov ecx, A; add ecx, B` — the entire push/RHS-chain/copy/pop
    scaffold collapses, since after retarget the chain only writes
    ECX and EAX is preserved naturally."""
    asm = (
        "_f:\n"
        "        push    eax\n"
        "        mov     eax, [ebp + 12]\n"
        "        add     eax, [ebp - 4]\n"
        "        mov     ecx, eax\n"
        "        pop     eax\n"
        "        cmp     eax, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     ecx, [ebp + 12]" in out
    assert "add     ecx, [ebp - 4]" in out
    assert "        mov     ecx, eax" not in out
    # The push/pop pair is gone too — entire scaffold collapsed.
    assert "push    eax" not in out
    assert "pop     eax" not in out
    assert opt.stats.get("right_operand_retarget", 0) == 1


def test_right_operand_retarget_chain_3():
    """3-instruction chain ending in movsx (which references [eax]
    in source) gets fully retargeted, including the [eax] → [ecx]
    rewrite of the source operand."""
    asm = (
        "_f:\n"
        "        push    eax\n"
        "        mov     eax, [ebp + 12]\n"
        "        add     eax, [ebp - 4]\n"
        "        movsx   eax, byte [eax]\n"
        "        mov     ecx, eax\n"
        "        pop     eax\n"
        "        cmp     eax, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "movsx   ecx, byte [ecx]" in out
    assert "mov     ecx, [ebp + 12]" in out
    assert "        mov     ecx, eax" not in out
    assert opt.stats.get("right_operand_retarget", 0) == 1


def test_right_operand_retarget_skips_when_chain_reads_ecx():
    """If a chain instruction's source references ECX, retargeting
    would self-reference. Skip."""
    asm = (
        "_f:\n"
        "        push    eax\n"
        "        mov     eax, ecx\n"     # reads ECX!
        "        mov     ecx, eax\n"
        "        pop     eax\n"
        "        cmp     eax, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("right_operand_retarget", 0) == 0


def test_right_operand_retarget_skips_when_chain_starts_rmw():
    """If the chain's first instruction is a RMW (e.g., `add eax, X`)
    rather than a fresh write, the chain depends on EAX's prior
    value. Retargeting would compute the wrong thing. Skip."""
    asm = (
        "_f:\n"
        "        push    eax\n"
        "        add     eax, [ebp - 4]\n"  # RMW — reads EAX from before push
        "        mov     ecx, eax\n"
        "        pop     eax\n"
        "        cmp     eax, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("right_operand_retarget", 0) == 0


def test_right_operand_retarget_skips_no_push_eax():
    """Without a `push eax` chain-start marker, we can't safely
    determine where the RHS chain begins. Skip."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 12]\n"
        "        mov     ecx, eax\n"
        "        pop     eax\n"
        "        cmp     eax, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("right_operand_retarget", 0) == 0


def test_right_operand_retarget_skips_esp_relative_source():
    """If the chain reads `[esp + N]`, dropping the push/pop scaffold
    that surrounds it would change which byte the load targets."""
    asm = (
        "_f:\n"
        "        push    eax\n"
        "        mov     eax, [esp + 4]\n"
        "        mov     ecx, eax\n"
        "        pop     eax\n"
        "        cmp     eax, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("right_operand_retarget", 0) == 0


def test_right_operand_retarget_lea_chain():
    """`lea eax, [...]` is a fresh EAX write — eligible chain start."""
    asm = (
        "_f:\n"
        "        push    eax\n"
        "        lea     eax, [_glob]\n"
        "        add     eax, [ebp - 4]\n"
        "        mov     ecx, eax\n"
        "        pop     eax\n"
        "        cmp     eax, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "lea     ecx, [_glob]" in out
    assert opt.stats.get("right_operand_retarget", 0) == 1


# ── cmp_zero_to_test ─────────────────────────────────────────────


def test_cmp_zero_to_test_eax():
    """`cmp eax, 0` → `test eax, eax` (1 byte saved)."""
    asm = (
        "_f:\n"
        "        cmp     eax, 0\n"
        "        je      .L\n"
        ".L:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "test    eax, eax" in out
    assert "cmp     eax, 0" not in out
    assert opt.stats.get("cmp_zero_to_test", 0) == 1


def test_cmp_zero_to_test_other_regs():
    """Pattern works for any 32-bit gp reg, not just EAX."""
    asm = (
        "_f:\n"
        "        cmp     ebx, 0\n"
        "        cmp     ecx, 0\n"
        "        cmp     esi, 0\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "test    ebx, ebx" in out
    assert "test    ecx, ecx" in out
    assert "test    esi, esi" in out
    assert opt.stats.get("cmp_zero_to_test", 0) == 3


def test_cmp_zero_to_test_skips_nonzero():
    """`cmp eax, 5` doesn't fold."""
    asm = (
        "_f:\n"
        "        cmp     eax, 5\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("cmp_zero_to_test", 0) == 0


# ── dead_mov_to_reg ──────────────────────────────────────────────


def test_dead_mov_postfix_inc():
    """The classic postfix `i++` value-discarded pattern in for-loop
    step: `mov eax, [X]; inc dword [X]; jmp .top` where the load
    is dead because .top's first instr writes EAX."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp - 4]\n"
        "        inc     dword [ebp - 4]\n"
        "        jmp     .L_top\n"
        ".L_top:\n"
        "        mov     eax, [ebp - 4]\n"
        "        cmp     eax, 10\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # The first `mov eax, [ebp - 4]` (before inc) is dropped.
    # The second one (after the label) is preserved.
    assert out.count("mov     eax, [ebp - 4]") == 1
    assert opt.stats.get("dead_mov_to_reg", 0) == 1


def test_dead_mov_skips_when_value_used():
    """If the load's value is read after, don't drop."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp - 4]\n"
        "        inc     dword [ebp - 4]\n"
        "        cmp     eax, 0\n"     # reads eax!
        "        jne     .L\n"
        ".L:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("dead_mov_to_reg", 0) == 0


def test_dead_mov_skips_self_referential_load():
    """`mov eax, [eax]` reads EAX before writing — not safe to drop."""
    asm = (
        "_f:\n"
        "        mov     eax, [eax]\n"
        "        mov     eax, 5\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("dead_mov_to_reg", 0) == 0


def test_dead_mov_call_clobbers_eax():
    """A direct `call _foo` clobbers EAX (cdecl), so a preceding
    `mov eax, X` (where X isn't used by the call) is dead."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp - 4]\n"
        "        call    _foo\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     eax, [ebp - 4]" not in out
    assert opt.stats.get("dead_mov_to_reg", 0) == 1


def test_dead_mov_skips_indirect_call_via_eax():
    """`call eax` reads EAX, so the prior `mov eax, X` is live."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp - 4]\n"
        "        call    eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("dead_mov_to_reg", 0) == 0


def test_dead_mov_eax_at_ret_is_live():
    """At a `ret`, EAX is the return value (live). Don't drop a
    prior mov to EAX — its value gets returned."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp - 4]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("dead_mov_to_reg", 0) == 0


def test_dead_mov_skips_implicit_eax_reader():
    """`cdq` reads EAX implicitly. The candidate `mov eax, X` is
    NOT dead — its value flows into cdq."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp - 4]\n"
        "        cdq\n"
        "        idiv    ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("dead_mov_to_reg", 0) == 0


def test_dead_mov_skips_div_reads_edx_eax():
    """`idiv reg` reads EDX:EAX. A `mov eax, X` before idiv is
    live — its value is the dividend's low half."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp - 4]\n"
        "        cdq\n"          # also reads eax
        "        idiv    ecx\n"  # reads edx:eax
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("dead_mov_to_reg", 0) == 0


def test_dead_mov_skips_other_regs():
    """The pattern is restricted to EAX. Non-EAX regs (ECX, EBX,
    EDX, ESI, EDI) might be implicitly used by nested-fn trampolines
    (ECX) or the LL-return ABI (EDX), so dropping their writes is
    risky. Skip."""
    asm = (
        "_f:\n"
        "        mov     ebx, [ebp - 4]\n"
        "        mov     ecx, 5\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     ebx, [ebp - 4]" in out
    assert "mov     ecx, 5" in out
    assert opt.stats.get("dead_mov_to_reg", 0) == 0


# ── prologue_to_enter ────────────────────────────────────────────


def test_prologue_to_enter_basic():
    asm = (
        "_f:\n"
        "        push    ebp\n"
        "        mov     ebp, esp\n"
        "        sub     esp, 24\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "        enter   24, 0" in out
    assert "        push    ebp" not in out
    assert "        mov     ebp, esp" not in out
    assert "        sub     esp, 24" not in out
    assert opt.stats.get("prologue_to_enter") == 1


def test_prologue_to_enter_skips_no_sub_esp():
    """Frame size 0 — no `sub esp` line — skip (enter would be larger)."""
    asm = (
        "_f:\n"
        "        push    ebp\n"
        "        mov     ebp, esp\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "enter" not in out
    assert opt.stats.get("prologue_to_enter", 0) == 0


def test_prologue_to_enter_skips_register_sub():
    """`sub esp, eax` (VLA-style) must not collapse."""
    asm = (
        "_f:\n"
        "        push    ebp\n"
        "        mov     ebp, esp\n"
        "        sub     esp, eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "enter" not in out
    assert "sub     esp, eax" in out
    assert opt.stats.get("prologue_to_enter", 0) == 0


def test_prologue_to_enter_skips_imm_too_large():
    """Frame size > 65535 — imm doesn't fit `enter`'s imm16 first
    operand. Skip."""
    asm = (
        "_f:\n"
        "        push    ebp\n"
        "        mov     ebp, esp\n"
        "        sub     esp, 70000\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "enter" not in out
    assert "sub     esp, 70000" in out
    assert opt.stats.get("prologue_to_enter", 0) == 0


def test_prologue_to_enter_at_imm16_boundary():
    """65535 fits, 65536 doesn't."""
    for imm, should_fire in [(65535, True), (65536, False)]:
        asm = (
            "_f:\n"
            "        push    ebp\n"
            "        mov     ebp, esp\n"
            f"        sub     esp, {imm}\n"
            "        ret\n"
        )
        opt = PeepholeOptimizer()
        out = opt.optimize(asm)
        if should_fire:
            assert f"enter   {imm}, 0" in out
            assert opt.stats.get("prologue_to_enter") == 1
        else:
            assert "enter" not in out
            assert opt.stats.get("prologue_to_enter", 0) == 0


def test_prologue_to_enter_hex_immediate():
    """NASM accepts `0x18` as well as `24` for the sub esp operand."""
    asm = (
        "_f:\n"
        "        push    ebp\n"
        "        mov     ebp, esp\n"
        "        sub     esp, 0x18\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # Decoded as 24 decimal, emitted as decimal in the new instruction.
    assert "        enter   24, 0" in out
    assert opt.stats.get("prologue_to_enter") == 1


def test_prologue_to_enter_multiple_functions():
    asm = (
        "_a:\n"
        "        push    ebp\n"
        "        mov     ebp, esp\n"
        "        sub     esp, 8\n"
        "        ret\n"
        "_b:\n"
        "        push    ebp\n"
        "        mov     ebp, esp\n"
        "        sub     esp, 16\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "        enter   8, 0" in out
    assert "        enter   16, 0" in out
    assert opt.stats.get("prologue_to_enter") == 2


def test_prologue_to_enter_skips_when_intervening_label():
    """If there's a label between the three lines, the pattern is not
    a true prologue — skip. (Pathological — codegen never emits this.)"""
    asm = (
        "_f:\n"
        "        push    ebp\n"
        ".weird:\n"
        "        mov     ebp, esp\n"
        "        sub     esp, 24\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "enter" not in out
    assert opt.stats.get("prologue_to_enter", 0) == 0


def test_prologue_to_enter_skips_when_static_link_save_first():
    """Lifted nested fns save ECX into a static-link slot AFTER the
    prologue, not within it. The pattern still matches (3 instr lines
    are contiguous), and the ECX save is preserved untouched."""
    asm = (
        "_inner:\n"
        "        push    ebp\n"
        "        mov     ebp, esp\n"
        "        sub     esp, 12\n"
        "        mov     [ebp - 12], ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "        enter   12, 0" in out
    assert "        mov     [ebp - 12], ecx" in out
    assert opt.stats.get("prologue_to_enter") == 1


def test_prologue_to_enter_byte_savings():
    """Verify the 3-line → 1-line collapse: line count drops by 2."""
    asm = (
        "_f:\n"
        "        push    ebp\n"
        "        mov     ebp, esp\n"
        "        sub     esp, 24\n"
        "        ret\n"
    )
    out = optimize(asm)
    # Original: 5 lines (label + 3 prologue + ret).
    # Optimized: 3 lines (label + enter + ret).
    in_lines = [l for l in asm.splitlines() if l.strip()]
    out_lines = [l for l in out.splitlines() if l.strip()]
    assert len(in_lines) - len(out_lines) == 2


# ── redundant_eax_load ───────────────────────────────────────────


def test_redundant_eax_load_basic():
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     [ebp - 4], eax\n"
        "        mov     eax, [ebp + 8]\n"
        "        imul    eax, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # Second `mov eax, [ebp + 8]` is redundant.
    occurrences = out.count("mov     eax, [ebp + 8]")
    assert occurrences == 1
    assert opt.stats.get("redundant_eax_load") == 1


def test_redundant_eax_load_skips_label_boundary():
    """Don't drop across a label — control-flow could enter from
    elsewhere."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        ".L1:\n"
        "        mov     eax, [ebp + 8]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # Both loads must survive (label could be a jump target).
    # (dead_mov_to_reg might still drop the first one — but the
    # second must remain.)
    assert "mov     eax, [ebp + 8]\n.L1:" in out or out.count(
        "mov     eax, [ebp + 8]"
    ) >= 1
    # Whatever happens, redundant_eax_load shouldn't fire.
    assert opt.stats.get("redundant_eax_load", 0) == 0


def test_redundant_eax_load_skips_call_clobber():
    """A call clobbers EAX — the second load is NOT redundant."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        call    _bar\n"
        "        mov     eax, [ebp + 8]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("redundant_eax_load", 0) == 0


def test_redundant_eax_load_skips_aliasing_write():
    """A write to the same memory invalidates the tracked value."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     [ebp + 8], ecx\n"
        "        mov     eax, [ebp + 8]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("redundant_eax_load", 0) == 0


def test_redundant_eax_load_disjoint_stack_writes_are_safe():
    """A write to a different ebp offset is provably disjoint."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     [ebp - 4], eax\n"
        "        mov     [ebp - 8], eax\n"
        "        mov     eax, [ebp + 8]\n"
        "        imul    eax, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert out.count("mov     eax, [ebp + 8]") == 1
    assert opt.stats.get("redundant_eax_load") == 1


# ── label_offset_fold ────────────────────────────────────────────


def test_label_offset_fold_basic_add():
    asm = (
        "_f:\n"
        "        mov     eax, _b\n"
        "        add     eax, 8\n"
        "        mov     eax, [eax]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "        mov     eax, _b + 8" in out
    assert "add     eax, 8" not in out
    assert opt.stats.get("label_offset_fold") == 1


def test_label_offset_fold_basic_sub():
    asm = (
        "_f:\n"
        "        mov     eax, _b\n"
        "        sub     eax, 4\n"
        "        mov     eax, [eax]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "        mov     eax, _b - 4" in out
    assert "sub     eax, 4" not in out
    assert opt.stats.get("label_offset_fold") == 1


def test_label_offset_fold_skips_numeric_source():
    """If the mov source is a numeric literal, the codegen should
    have already const-folded — don't fire."""
    asm = (
        "_f:\n"
        "        mov     eax, 42\n"
        "        add     eax, 8\n"
        "        mov     ecx, eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("label_offset_fold", 0) == 0


def test_label_offset_fold_skips_memory_source():
    asm = (
        "_f:\n"
        "        mov     eax, [_b]\n"
        "        add     eax, 8\n"
        "        mov     ecx, eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("label_offset_fold", 0) == 0


def test_label_offset_fold_skips_when_flags_read():
    """`add` sets flags; if a Jcc reads them before they're
    overwritten, dropping the add changes program behavior."""
    asm = (
        "_f:\n"
        "        mov     eax, _b\n"
        "        add     eax, 8\n"
        "        je      .L1\n"
        ".L1:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("label_offset_fold", 0) == 0


def test_label_offset_fold_safe_when_next_overwrites_flags():
    """If the next instruction overwrites flags (cmp/test/another
    arithmetic), the original add's flag effects are dead → safe to
    drop."""
    asm = (
        "_f:\n"
        "        mov     eax, _b\n"
        "        add     eax, 8\n"
        "        cmp     eax, ecx\n"
        "        je      .L1\n"
        ".L1:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "        mov     eax, _b + 8" in out
    assert opt.stats.get("label_offset_fold") == 1


def test_label_offset_fold_skips_register_source():
    """`mov reg, reg` shouldn't trigger — only label/expression
    sources warrant the fold."""
    asm = (
        "_f:\n"
        "        mov     eax, ecx\n"
        "        add     eax, 8\n"
        "        mov     edx, eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("label_offset_fold", 0) == 0


def test_label_offset_fold_handles_label_arithmetic_source():
    """`mov eax, _b + 4` followed by `add eax, 4` should fold to
    `mov eax, _b + 4 + 4` (NASM resolves at assemble time)."""
    asm = (
        "_f:\n"
        "        mov     eax, _b + 4\n"
        "        add     eax, 4\n"
        "        mov     eax, [eax]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # NASM accepts nested label-arithmetic.
    assert opt.stats.get("label_offset_fold") == 1
    assert "        mov     eax, _b + 4 + 4" in out


def test_label_offset_fold_dot_label():
    """User-declared local labels (`.foo`) are still address-takers."""
    asm = (
        "_f:\n"
        "        mov     eax, .L1_target\n"
        "        add     eax, 16\n"
        "        mov     eax, [eax]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("label_offset_fold") == 1
    assert "        mov     eax, .L1_target + 16" in out


def test_label_offset_fold_works_with_any_register():
    """Pattern fires for ECX, EDX, etc. — not just EAX."""
    for reg in ("eax", "ebx", "ecx", "edx", "esi", "edi"):
        asm = (
            "_f:\n"
            f"        mov     {reg}, _glob\n"
            f"        add     {reg}, 8\n"
            f"        mov     [esp - 4], {reg}\n"
            "        ret\n"
        )
        opt = PeepholeOptimizer()
        out = opt.optimize(asm)
        assert f"        mov     {reg}, _glob + 8" in out, f"{reg}: {out}"
        assert opt.stats.get("label_offset_fold") == 1


# ── cmp_load_collapse ────────────────────────────────────────────


def test_cmp_load_collapse_basic():
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        cmp     eax, 5\n"
        "        jne     .L1\n"
        "        mov     eax, 1\n"
        "        ret\n"
        ".L1:\n"
        "        mov     eax, [ebp + 8]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "        cmp     dword [ebp + 8], 5" in out
    assert "        mov     eax, [ebp + 8]\n        cmp" not in out
    assert opt.stats.get("cmp_load_collapse") == 1


def test_cmp_load_collapse_skips_eax_used_after():
    """If EAX is read after the cmp (and not freshly overwritten),
    the load must survive."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        cmp     eax, 5\n"
        "        jne     .L1\n"
        "        push    eax\n"  # EAX read
        "        ret\n"
        ".L1:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("cmp_load_collapse", 0) == 0


def test_cmp_load_collapse_skips_mem_mem_cmp():
    """x86 has no `cmp mem, mem` form — skip when cmp src is a memory
    reference."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        cmp     eax, [ebp - 4]\n"
        "        jne     .L1\n"
        "        mov     eax, 1\n"
        "        ret\n"
        ".L1:\n"
        "        mov     eax, 2\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("cmp_load_collapse", 0) == 0


def test_cmp_load_collapse_with_register_src():
    """`cmp [mem], reg` is a valid x86 form."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        cmp     eax, ecx\n"
        "        jne     .L1\n"
        "        mov     eax, 1\n"
        "        ret\n"
        ".L1:\n"
        "        mov     eax, 2\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "        cmp     dword [ebp + 8], ecx" in out
    assert opt.stats.get("cmp_load_collapse") == 1


def test_cmp_load_collapse_with_jcc_branching():
    """The CFG-aware liveness scan must follow BOTH branches of the
    jcc to verify EAX is dead on each."""
    # Both branches eventually overwrite EAX → dead.
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        cmp     eax, 0\n"
        "        je      .L_zero\n"
        "        mov     eax, 1\n"
        "        ret\n"
        ".L_zero:\n"
        "        xor     eax, eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "        cmp     dword [ebp + 8], 0" in out
    assert opt.stats.get("cmp_load_collapse") == 1


def test_cmp_load_collapse_one_branch_uses_eax():
    """If one branch reads EAX (e.g. uses it in a return), the
    pattern must NOT fire."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        cmp     eax, 0\n"
        "        je      .L_zero\n"
        "        ret\n"  # EAX is the return value!
        ".L_zero:\n"
        "        xor     eax, eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    # The fallthrough `ret` makes EAX live (return value).
    assert opt.stats.get("cmp_load_collapse", 0) == 0


# ── rmw_collapse ─────────────────────────────────────────────────


def test_rmw_collapse_basic_add():
    asm = (
        "_f:\n"
        "        mov     eax, [ebp - 4]\n"
        "        add     eax, 5\n"
        "        mov     [ebp - 4], eax\n"
        "        xor     eax, eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "        add     dword [ebp - 4], 5" in out
    # Original 3 instructions gone.
    assert "mov     eax, [ebp - 4]" not in out
    assert "add     eax, 5" not in out
    assert "mov     [ebp - 4], eax" not in out
    assert opt.stats.get("rmw_collapse") == 1


def test_rmw_collapse_all_ops():
    """sub, and, or, xor — all have x86 r/m32, imm forms."""
    for op in ("sub", "and", "or", "xor"):
        asm = (
            "_f:\n"
            "        mov     eax, [ebp - 4]\n"
            f"        {op}     eax, 7\n"
            "        mov     [ebp - 4], eax\n"
            "        xor     eax, eax\n"
            "        ret\n"
        )
        opt = PeepholeOptimizer()
        out = opt.optimize(asm)
        # Account for op-name padding: NASM keeps a 4-char gutter.
        # Just check the op + memory operand survives.
        assert f"{op}" in out and f"dword [ebp - 4], 7" in out, (
            f"{op}: {out}"
        )
        assert opt.stats.get("rmw_collapse") == 1


def test_rmw_collapse_skips_eax_live_after():
    """If EAX is live after the store (e.g. it's the return value),
    don't collapse — the rewrite drops the EAX update."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp - 4]\n"
        "        add     eax, 5\n"
        "        mov     [ebp - 4], eax\n"
        "        ret\n"  # EAX is the return value
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("rmw_collapse", 0) == 0


def test_rmw_collapse_with_register_source():
    """`add eax, ecx` (non-EAX register source) folds to `add [mem],
    ecx` — x86 supports the r/m32, r32 form."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp - 4]\n"
        "        add     eax, ecx\n"
        "        mov     [ebp - 4], eax\n"
        "        xor     eax, eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "        add     dword [ebp - 4], ecx" in out
    assert opt.stats.get("rmw_collapse") == 1


def test_rmw_collapse_skips_eax_source():
    """`add eax, eax` self-doubles — but if the load is dropped, EAX
    stale → wrong result. Skip this case."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp - 4]\n"
        "        add     eax, eax\n"
        "        mov     [ebp - 4], eax\n"
        "        xor     eax, eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("rmw_collapse", 0) == 0


def test_rmw_collapse_skips_memory_source():
    """`add eax, [mem2]` — x86 has no mem-mem op, can't collapse to
    `add [mem1], [mem2]`."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp - 4]\n"
        "        add     eax, [ebp - 8]\n"
        "        mov     [ebp - 4], eax\n"
        "        xor     eax, eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("rmw_collapse", 0) == 0


def test_rmw_collapse_skips_mismatched_addresses():
    """Load and store must reference the same memory."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp - 4]\n"
        "        add     eax, 5\n"
        "        mov     [ebp - 8], eax\n"  # Different address!
        "        xor     eax, eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("rmw_collapse", 0) == 0


def test_rmw_collapse_at_function_call_witness():
    """A function call that clobbers EAX makes the load-store-call
    pattern eligible for collapse."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp - 4]\n"
        "        add     eax, 1\n"
        "        mov     [ebp - 4], eax\n"
        "        call    _other\n"  # clobbers EAX
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "        add     dword [ebp - 4], 1" in out
    assert opt.stats.get("rmw_collapse") == 1


# ── fst_fstp_collapse ────────────────────────────────────────────


def test_fst_fstp_collapse_basic_qword():
    asm = (
        "_f:\n"
        "        fld     qword [eax]\n"
        "        fst     qword [ebp - 8]\n"
        "        fstp    st0\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "        fstp    qword [ebp - 8]" in out
    assert "        fst     qword" not in out
    assert "        fstp    st0" not in out
    assert opt.stats.get("fst_fstp_collapse") == 1


def test_fst_fstp_collapse_dword():
    asm = (
        "_f:\n"
        "        fst     dword [ebp - 4]\n"
        "        fstp    st0\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "        fstp    dword [ebp - 4]" in out
    assert opt.stats.get("fst_fstp_collapse") == 1


def test_fst_fstp_collapse_skips_when_not_st0():
    """`fst X; fstp st1` (popping into a different register) is not
    the simple case — leave it alone."""
    asm = (
        "_f:\n"
        "        fst     qword [ebp - 8]\n"
        "        fstp    st1\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "        fst     qword [ebp - 8]" in out
    assert opt.stats.get("fst_fstp_collapse", 0) == 0


def test_fst_fstp_collapse_skips_intervening():
    """Don't collapse when there's other code between the fst and
    fstp — that code might depend on st0."""
    asm = (
        "_f:\n"
        "        fst     qword [ebp - 8]\n"
        "        fld1\n"
        "        fstp    st0\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("fst_fstp_collapse", 0) == 0


def test_fst_fstp_collapse_st_paren_form():
    """NASM accepts both `st0` and `st(0)`. Both should match."""
    asm = (
        "_f:\n"
        "        fst     dword [eax]\n"
        "        fstp    st(0)\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "        fstp    dword [eax]" in out
    assert opt.stats.get("fst_fstp_collapse") == 1


# ── fpu_op_collapse ──────────────────────────────────────────────


def test_fpu_op_collapse_faddp():
    asm = (
        "_f:\n"
        "        fld     qword [eax]\n"
        "        faddp   st1, st0\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "        fadd    qword [eax]" in out
    assert "        fld" not in out
    assert "        faddp" not in out
    assert opt.stats.get("fpu_op_collapse") == 1


def test_fpu_op_collapse_all_ops():
    """faddp/fmulp/fsubp/fdivp/fsubrp/fdivrp all map to memory form."""
    mapping = {
        "faddp": "fadd",
        "fmulp": "fmul",
        "fsubp": "fsub",
        "fdivp": "fdiv",
        "fsubrp": "fsubr",
        "fdivrp": "fdivr",
    }
    for popf, memf in mapping.items():
        asm = (
            "_f:\n"
            "        fld     dword [ebp - 4]\n"
            f"        {popf}   st1, st0\n"
            "        ret\n"
        )
        opt = PeepholeOptimizer()
        out = opt.optimize(asm)
        # Just check the op + memory operand survives — exact spacing
        # depends on op-name length (spacer is `8 - len(op)`).
        assert (memf in out and "dword [ebp - 4]" in out), (
            f"{popf}: {out}"
        )
        assert opt.stats.get("fpu_op_collapse") == 1


def test_fpu_op_collapse_dword():
    asm = (
        "_f:\n"
        "        fld     dword [ebp - 4]\n"
        "        fmulp   st1, st0\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "        fmul    dword [ebp - 4]" in out
    assert opt.stats.get("fpu_op_collapse") == 1


def test_fpu_op_collapse_skips_intervening():
    """No collapse when there's other code between fld and the
    pop-form op."""
    asm = (
        "_f:\n"
        "        fld     qword [eax]\n"
        "        fchs\n"
        "        faddp   st1, st0\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("fpu_op_collapse", 0) == 0


def test_fpu_op_collapse_skips_non_st1_st0_form():
    """`faddp st(2), st(0)` (different operands) is not the standard
    case — leave it alone."""
    asm = (
        "_f:\n"
        "        fld     qword [eax]\n"
        "        faddp   st2, st0\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("fpu_op_collapse", 0) == 0


def test_fpu_op_collapse_bare_pop_form():
    """`faddp` with no operands defaults to `st1, st0` per Intel."""
    asm = (
        "_f:\n"
        "        fld     qword [eax]\n"
        "        faddp\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "        fadd    qword [eax]" in out
    assert opt.stats.get("fpu_op_collapse") == 1


# ── Convergence ──────────────────────────────────────────────────


def test_idempotent_on_optimal_asm():
    """Running on already-optimal asm should be a no-op."""
    asm = (
        "_f:\n"
        "        push    ebp\n"
        "        mov     ebp, esp\n"
        "        mov     eax, 42\n"
        "        pop     ebp\n"
        "        ret\n"
    )
    out = optimize(asm)
    assert out == asm


def test_preserves_trailing_newline():
    asm = "_f:\n        ret\n"
    assert optimize(asm).endswith("\n")
    asm_no_nl = "_f:\n        ret"
    assert not optimize(asm_no_nl).endswith("\n")


# ── Pipeline integration ─────────────────────────────────────────


def test_codegen_runs_peephole_by_default():
    from uc_core.lexer import Lexer
    from uc_core.parser import Parser
    from uc386.codegen import CodeGenerator

    src = "int main(void) { return 0; }"
    tokens = list(Lexer(src, "test.c").tokenize())
    unit = Parser(tokens).parse()

    gen = CodeGenerator()
    asm = gen.generate(unit)

    # Stats populated.
    assert isinstance(gen.peephole_stats, dict)
    # The codegen-emitted `mov eax, 0` for the implicit return is
    # rewritten to `xor eax, eax` (3 bytes saved). The dead one
    # after `jmp .epilogue` (if any) is dropped.
    assert gen.peephole_stats.get("mov_zero_to_xor", 0) >= 1
    # No bare `mov eax, 0` survives in main.
    assert "        mov     eax, 0" not in asm


def test_codegen_skips_peephole_when_disabled():
    from uc_core.lexer import Lexer
    from uc_core.parser import Parser
    from uc386.codegen import CodeGenerator

    src = "int main(void) { return 0; }"
    tokens = list(Lexer(src, "test.c").tokenize())
    unit = Parser(tokens).parse()

    gen = CodeGenerator(peephole=False)
    asm = gen.generate(unit)

    # The dead `xor eax, eax / .epilogue:` pattern survives unoptimized.
    assert "        xor     eax, eax\n.epilogue:" in asm
    assert gen.peephole_stats == {}
