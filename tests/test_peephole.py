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
    """`mov eax, 0; mov [...], eax; mov eax, 1` initially collapses
    via imm_store_collapse to `mov dword [...], 0; mov eax, 1`. Then
    zero_init_collapse rewrites the dword-zero store to a shorter
    `xor eax, eax; mov [...], eax` form (saving 2 more bytes).

    End-to-end output:
        xor eax, eax
        mov [ebp - 4], eax
        mov eax, 1
        mov [ebp - 8], eax
    """
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
    assert "xor     eax, eax" in out
    assert "mov     [ebp - 4], eax" in out
    assert opt.stats.get("imm_store_collapse", 0) >= 1
    assert opt.stats.get("zero_init_collapse", 0) >= 1


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


def test_imm_store_collapse_cfg_aware_across_label():
    """CFG-aware fallback fires when EAX is dead across a label
    boundary (e.g. loop initialization where the loop top doesn't
    read EAX).

    Pre: `mov eax, IMM; mov [m], eax; .label: cmp [m], X; ...`
    Post: `mov dword [m], IMM; .label: cmp [m], X; ...`
    """
    asm = (
        "_f:\n"
        "        mov     eax, -5\n"
        "        mov     [ebp - 8], eax\n"
        ".L_top:\n"
        "        cmp     dword [ebp - 8], 200\n"
        "        jge     .L_end\n"
        "        mov     eax, [ebp - 8]\n"  # overwrites EAX (dead across label)
        "        jmp     .L_top\n"
        ".L_end:\n"
        "        mov     eax, 0\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     dword [ebp - 8], -5" in out
    assert opt.stats.get("imm_store_collapse") == 1


def test_imm_store_collapse_cfg_aware_skips_when_eax_live():
    """CFG-aware fallback must respect liveness — if EAX is read
    after the label boundary, the rewrite is unsafe."""
    asm = (
        "_f:\n"
        "        mov     eax, 99\n"
        "        mov     [ebp - 4], eax\n"
        ".L_top:\n"
        "        push    eax\n"  # EAX read here!
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
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
    # label_offset_fold produces `mov eax, _b + 8`, then
    # label_load_collapse fuses with the deref to `mov eax, [_b + 8]`.
    assert "        mov     eax, [_b + 8]" in out
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
    # label_offset_fold + label_load_collapse compose.
    assert "        mov     eax, [_b - 4]" in out
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
    # NASM accepts nested label-arithmetic. label_load_collapse
    # then fuses with the deref.
    assert opt.stats.get("label_offset_fold") == 1
    assert "        mov     eax, [_b + 4 + 4]" in out


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
    # label_load_collapse fuses with the deref.
    assert "        mov     eax, [.L1_target + 16]" in out


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
    pattern eligible for collapse. After rmw_collapse fires, the
    follow-up add_one_to_inc pass converts the resulting
    `add dword [mem], 1` → `inc dword [mem]` for an additional
    1-byte saving."""
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
    # rmw_collapse fired (load-add-store → memory-RMW)...
    assert opt.stats.get("rmw_collapse") == 1
    # ...and add_one_to_inc then folded the +1 into inc.
    assert "        inc     dword [ebp - 4]" in out
    assert "        add     dword [ebp - 4], 1" not in out


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


# ── add_one_to_inc ───────────────────────────────────────────────


def test_add_one_to_inc_basic():
    asm = (
        "_f:\n"
        "        add     eax, 1\n"
        "        mov     [ebp - 4], eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "        inc     eax" in out
    assert "add     eax, 1" not in out
    assert opt.stats.get("add_one_to_inc") == 1


def test_sub_one_to_dec():
    asm = (
        "_f:\n"
        "        sub     ecx, 1\n"
        "        cmp     ecx, eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "        dec     ecx" in out
    assert opt.stats.get("add_one_to_inc") == 1


def test_add_one_to_inc_skips_when_carry_read():
    """`jc` reads CF — `inc` doesn't set CF, so the rewrite changes
    behavior. Must skip."""
    asm = (
        "_f:\n"
        "        add     eax, 1\n"
        "        jc      .L1\n"
        "        ret\n"
        ".L1:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("add_one_to_inc", 0) == 0


def test_add_one_to_inc_safe_after_cmp():
    """`cmp` overwrites all flags including CF — `add`'s CF is dead."""
    asm = (
        "_f:\n"
        "        add     ebx, 1\n"
        "        cmp     ebx, eax\n"
        "        je      .L1\n"
        ".L1:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "        inc     ebx" in out
    assert opt.stats.get("add_one_to_inc") == 1


def test_add_one_to_inc_skips_imm_other_than_1():
    asm = (
        "_f:\n"
        "        add     eax, 2\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("add_one_to_inc", 0) == 0


def test_add_one_to_inc_memory_dest():
    """`add dword [mem], 1` → `inc dword [mem]` saves 1 byte
    (4 bytes → 3 bytes for ebp-relative addressing)."""
    asm = (
        "_f:\n"
        "        add     dword [ebp - 4], 1\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "inc     dword [ebp - 4]" in out
    assert "add     dword [ebp - 4], 1" not in out
    assert opt.stats.get("add_one_to_inc", 0) == 1


def test_add_one_to_inc_memory_byte():
    """Byte-form memory inc/dec also saves 1 byte."""
    asm = (
        "_f:\n"
        "        add     byte [eax], 1\n"
        "        sub     byte [eax], 1\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "inc     byte [eax]" in out
    assert "dec     byte [eax]" in out
    assert opt.stats.get("add_one_to_inc", 0) == 2


def test_add_one_to_inc_memory_unsized_skips():
    """Without a NASM size keyword we can't safely emit `inc [mem]`
    (NASM rejects it), so skip."""
    asm = (
        "_f:\n"
        "        add     [ebp - 4], 1\n"  # malformed but defensive
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("add_one_to_inc", 0) == 0


def test_add_one_to_inc_memory_skips_when_cf_live():
    """Memory inc/dec must also skip when CF is live (jc, jb, etc.)."""
    asm = (
        "_f:\n"
        "        add     dword [ebp - 4], 1\n"
        "        jc      .overflow\n"
        "        ret\n"
        ".overflow:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("add_one_to_inc", 0) == 0


def test_add_one_to_inc_all_registers():
    """Pattern fires for any of the 8 32-bit GP regs."""
    for reg in ("eax", "ebx", "ecx", "edx", "esi", "edi"):
        asm = (
            "_f:\n"
            f"        add     {reg}, 1\n"
            f"        cmp     {reg}, eax\n"
            "        je      .L1\n"
            ".L1:\n"
            "        ret\n"
        )
        opt = PeepholeOptimizer()
        out = opt.optimize(asm)
        assert f"        inc     {reg}" in out, f"{reg}: {out}"
        assert opt.stats.get("add_one_to_inc") == 1


# ── redundant_test_collapse ──────────────────────────────────────


def test_redundant_test_collapse_after_and():
    asm = (
        "_f:\n"
        "        and     eax, 1\n"
        "        test    eax, eax\n"
        "        jz      .L1\n"
        ".L1:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "        test    eax, eax" not in out
    assert "        and     eax, 1" in out
    assert opt.stats.get("redundant_test_collapse") == 1


def test_redundant_test_collapse_after_logical():
    """and/or/xor clear OF and CF (same as test reg, reg) and set
    ZF/SF/PF based on result — the test eax, eax after them is
    fully redundant for any subsequent Jcc."""
    for op_setup in (
        "        and     eax, 0xF\n",
        "        or      eax, 0xFF\n",
        "        xor     eax, ebx\n",
    ):
        asm = (
            "_f:\n"
            f"{op_setup}"
            "        test    eax, eax\n"
            "        jz      .L1\n"
            ".L1:\n"
            "        ret\n"
        )
        opt = PeepholeOptimizer()
        out = opt.optimize(asm)
        assert "test    eax, eax" not in out, f"{op_setup}: {out}"
        assert opt.stats.get("redundant_test_collapse") == 1


def test_redundant_test_collapse_skips_unsafe_arithmetic():
    """add/sub/inc/dec/neg/shl/shr/sar set OF and CF based on
    overflow or shifted-out bits, which differs from test's
    'always clear OF/CF'. Dropping `test reg, reg` after these
    would change behavior for jg/jl/ja/jb/jo/etc.

    Specifically caught by torture's signed-comparison-after-sub
    tests (20000403-1, bf-sign-2): `sub eax, ecx; test eax, eax;
    jg L` with overflow has SF != OF after sub (so jg not-taken)
    but SF == OF after sub+test (so jg may be taken). Different.
    """
    for op_setup in (
        "        add     eax, 5\n",
        "        sub     eax, 3\n",
        "        inc     eax\n",
        "        dec     eax\n",
        "        neg     eax\n",
        "        shl     eax, 2\n",
        "        shr     eax, 1\n",
        "        sar     eax, 4\n",
    ):
        asm = (
            "_f:\n"
            f"{op_setup}"
            "        test    eax, eax\n"
            "        jg      .L1\n"
            ".L1:\n"
            "        ret\n"
        )
        opt = PeepholeOptimizer()
        opt.optimize(asm)
        assert opt.stats.get("redundant_test_collapse", 0) == 0, (
            f"{op_setup}: must not collapse"
        )


def test_redundant_test_collapse_skips_intervening_op():
    """Don't drop test if there's an instruction between the
    flag-setter and the test."""
    asm = (
        "_f:\n"
        "        and     eax, 1\n"
        "        push    eax\n"
        "        test    eax, eax\n"
        "        jz      .L1\n"
        ".L1:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("redundant_test_collapse", 0) == 0


def test_redundant_test_collapse_skips_mismatched_register():
    """`add eax, X; test ebx, ebx` — different reg, not redundant."""
    asm = (
        "_f:\n"
        "        add     eax, 5\n"
        "        test    ebx, ebx\n"
        "        jz      .L1\n"
        ".L1:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("redundant_test_collapse", 0) == 0


def test_redundant_test_collapse_skips_non_arithmetic():
    """`mov eax, X; test eax, eax` — mov DOES NOT set flags. Test
    is needed."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp - 4]\n"
        "        test    eax, eax\n"
        "        jz      .L1\n"
        ".L1:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("redundant_test_collapse", 0) == 0


def test_redundant_test_collapse_block_boundary():
    """A label between flag-setter and test is a basic block
    boundary — control could enter from elsewhere with different
    flag state."""
    asm = (
        "_f:\n"
        "        and     eax, 1\n"
        ".L_target:\n"
        "        test    eax, eax\n"
        "        jz      .L1\n"
        ".L1:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("redundant_test_collapse", 0) == 0


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


# ── narrowing_load_test_collapse ─────────────────────────────────


def test_narrowing_load_test_collapse_movsx_byte():
    """`movsx eax, byte [SRC]; test eax, eax` → `cmp byte [SRC], 0`.
    EAX must be dead after the test (here: both branches overwrite
    EAX before any read)."""
    asm = (
        "_f:\n"
        "        movsx   eax, byte [esi]\n"
        "        test    eax, eax\n"
        "        jz      .end\n"
        "        mov     eax, 1\n"  # overwrites EAX
        "        ret\n"
        ".end:\n"
        "        xor     eax, eax\n"  # also overwrites EAX
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "cmp     byte [esi], 0" in out
    assert "movsx" not in out
    assert opt.stats.get("narrowing_load_test_collapse") == 1


def test_narrowing_load_test_collapse_movzx_byte():
    """`movzx eax, byte [SRC]` is also handled."""
    asm = (
        "_f:\n"
        "        movzx   eax, byte [ebp - 4]\n"
        "        test    eax, eax\n"
        "        jnz     .yes\n"
        "        mov     eax, [ebp - 8]\n"  # overwrites EAX
        "        ret\n"
        ".yes:\n"
        "        mov     eax, [ebp - 12]\n"  # overwrites EAX
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "cmp     byte [ebp - 4], 0" in out
    assert opt.stats.get("narrowing_load_test_collapse") == 1


def test_narrowing_load_test_collapse_word():
    """Word-form variant: `movsx eax, word [...]; test eax, eax`."""
    asm = (
        "_f:\n"
        "        movsx   eax, word [edi]\n"
        "        test    eax, eax\n"
        "        jz      .end\n"
        "        mov     eax, 5\n"
        "        ret\n"
        ".end:\n"
        "        xor     eax, eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "cmp     word [edi], 0" in out
    assert opt.stats.get("narrowing_load_test_collapse") == 1


def test_narrowing_load_test_collapse_skips_when_eax_live():
    """If EAX is read after the test, the load can't be dropped."""
    asm = (
        "_f:\n"
        "        movsx   eax, byte [esi]\n"
        "        test    eax, eax\n"
        "        mov     [ebp - 4], eax\n"  # uses EAX
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("narrowing_load_test_collapse", 0) == 0


def test_narrowing_load_test_collapse_skips_non_eax_dest():
    """Restricted to EAX dest for now."""
    asm = (
        "_f:\n"
        "        movsx   ecx, byte [esi]\n"
        "        test    ecx, ecx\n"
        "        jz      .end\n"
        "        ret\n"
        ".end:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("narrowing_load_test_collapse", 0) == 0


def test_narrowing_load_test_collapse_skips_full_dword_load():
    """Plain ``mov eax, [mem]; test eax, eax`` is handled by
    cmp_load_collapse (becomes ``cmp dword [mem], 0``); this pass
    should not double-fire on it."""
    asm = (
        "_f:\n"
        "        mov     eax, [esi]\n"
        "        test    eax, eax\n"
        "        jz      .end\n"
        "        mov     eax, 1\n"
        "        ret\n"
        ".end:\n"
        "        xor     eax, eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # cmp_load_collapse handles this — the result is dword cmp, not byte.
    assert "cmp     dword [esi], 0" in out
    assert opt.stats.get("narrowing_load_test_collapse", 0) == 0
    assert opt.stats.get("cmp_load_collapse", 0) == 1


# ── index_load_collapse ──────────────────────────────────────────


def test_index_load_collapse_scale_4():
    """`shl ecx, 2; add eax, ecx; mov eax, [eax]` collapses to
    `mov eax, [eax + ecx*4]` using x86's SIB byte."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     ecx, [ebp - 8]\n"
        "        shl     ecx, 2\n"
        "        add     eax, ecx\n"
        "        mov     eax, [eax]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     eax, [eax + ecx*4]" in out
    assert "shl     ecx, 2" not in out
    assert opt.stats.get("index_load_collapse") == 1


def test_index_load_collapse_scale_2():
    """Scale 2 (for short arrays): `shl reg, 1`."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     ecx, [ebp - 8]\n"
        "        shl     ecx, 1\n"
        "        add     eax, ecx\n"
        "        mov     eax, [eax]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     eax, [eax + ecx*2]" in out
    assert opt.stats.get("index_load_collapse") == 1


def test_index_load_collapse_scale_8():
    """Scale 8 (for long-long arrays / pointer arrays): `shl 3`."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     ecx, [ebp - 8]\n"
        "        shl     ecx, 3\n"
        "        add     eax, ecx\n"
        "        mov     eax, [eax]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     eax, [eax + ecx*8]" in out
    assert opt.stats.get("index_load_collapse") == 1


def test_index_load_collapse_skips_when_idx_live():
    """If IDX is read after the load, we can't drop the shl."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     ecx, [ebp - 8]\n"
        "        shl     ecx, 2\n"
        "        add     eax, ecx\n"
        "        mov     eax, [eax]\n"
        "        mov     [ebp - 4], ecx\n"  # reads ECX!
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("index_load_collapse", 0) == 0


def test_index_load_collapse_skips_when_base_live_distinct():
    """If DST != BASE and BASE is live after the load, we can't drop
    the add."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     ecx, [ebp - 8]\n"
        "        shl     ecx, 2\n"
        "        add     eax, ecx\n"
        "        mov     edx, [eax]\n"  # DST=edx, BASE=eax — distinct
        "        mov     [ebp - 4], eax\n"  # reads EAX
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("index_load_collapse", 0) == 0


def test_index_load_collapse_distinct_dst_when_base_dead():
    """When DST != BASE but BASE is dead after the load, the rewrite
    is safe."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     ecx, [ebp - 8]\n"
        "        shl     ecx, 2\n"
        "        add     eax, ecx\n"
        "        mov     edx, [eax]\n"  # DST=edx, BASE=eax dead after
        "        mov     eax, edx\n"  # overwrites eax (witness)
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     edx, [eax + ecx*4]" in out
    assert opt.stats.get("index_load_collapse") == 1


def test_index_load_collapse_skips_invalid_scale():
    """Only scales 2/4/8 — not scale 16 or scale 1."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     ecx, [ebp - 8]\n"
        "        shl     ecx, 4\n"  # scale 16, not supported
        "        add     eax, ecx\n"
        "        mov     eax, [eax]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("index_load_collapse", 0) == 0


def test_index_load_collapse_with_positive_displacement():
    """Struct member access via array index: ``arr[i].y`` lowers
    as ``shl ecx, 3; add eax, ecx; mov eax, [eax + 4]`` and
    collapses to ``mov eax, [eax + ecx*8 + 4]``.
    """
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     ecx, [ebp - 8]\n"
        "        shl     ecx, 3\n"
        "        add     eax, ecx\n"
        "        mov     eax, [eax + 4]\n"  # struct member offset
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     eax, [eax + ecx*8 + 4]" in out
    assert "shl     ecx, 3" not in out
    assert "add     eax, ecx" not in out
    assert opt.stats.get("index_load_collapse") == 1


def test_index_load_collapse_with_negative_displacement():
    """Less common but possible — ``[BASE - DISP]`` form."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     ecx, [ebp - 8]\n"
        "        shl     ecx, 2\n"
        "        add     eax, ecx\n"
        "        mov     eax, [eax - 8]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     eax, [eax + ecx*4 - 8]" in out
    assert opt.stats.get("index_load_collapse") == 1


def test_index_load_collapse_with_size_prefix_and_disp():
    """Sized memory operand with displacement preserves the size
    prefix in the rewrite."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     ecx, [ebp - 8]\n"
        "        shl     ecx, 2\n"
        "        add     eax, ecx\n"
        "        movsx   eax, byte [eax + 4]\n"  # sized + disp
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # Note: this matcher only fires for the `mov` op, not movsx,
    # so it should NOT collapse (movsx is not in our match set).
    assert opt.stats.get("index_load_collapse", 0) == 0


# ── compound_assign_collapse ─────────────────────────────────────


def test_compound_assign_collapse_basic():
    """The codegen's compound-assign frame collapses to in-place RMW.

    Pattern:
        push    dword [m]
        ... chain ending with value in eax ...
        mov     ecx, eax
        pop     eax
        add     eax, ecx
        mov     [m], eax

    becomes:
        ... chain ...
        add     [m], eax
    """
    asm = (
        "_f:\n"
        "        push    dword [ebp - 4]\n"
        "        mov     eax, [ebp + 8]\n"  # chain: load rhs
        "        mov     ecx, eax\n"
        "        pop     eax\n"
        "        add     eax, ecx\n"
        "        mov     [ebp - 4], eax\n"
        "        mov     eax, 0\n"  # ECX dead witness
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "add     [ebp - 4], eax" in out
    assert "push    dword [ebp - 4]" not in out
    assert "pop     eax" not in out
    assert opt.stats.get("compound_assign_collapse") == 1


def test_compound_assign_collapse_subtract():
    """Sub variant — `s -= rhs`."""
    asm = (
        "_f:\n"
        "        push    dword [ebp - 4]\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     ecx, eax\n"
        "        pop     eax\n"
        "        sub     eax, ecx\n"
        "        mov     [ebp - 4], eax\n"
        "        mov     eax, 0\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "sub     [ebp - 4], eax" in out
    assert opt.stats.get("compound_assign_collapse") == 1


def test_compound_assign_collapse_skips_when_chain_modifies_m():
    """If the chain modifies [m], we can't collapse — the saved
    value would be stale."""
    asm = (
        "_f:\n"
        "        push    dword [ebp - 4]\n"
        "        mov     dword [ebp - 4], 99\n"  # modifies [m]!
        "        mov     eax, [ebp + 8]\n"
        "        mov     ecx, eax\n"
        "        pop     eax\n"
        "        add     eax, ecx\n"
        "        mov     [ebp - 4], eax\n"
        "        mov     eax, 0\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("compound_assign_collapse", 0) == 0


def test_compound_assign_collapse_skips_when_chain_has_call():
    """A `call` in the chain clobbers ECX (cdecl)."""
    asm = (
        "_f:\n"
        "        push    dword [ebp - 4]\n"
        "        call    _g\n"  # clobbers ecx
        "        mov     ecx, eax\n"
        "        pop     eax\n"
        "        add     eax, ecx\n"
        "        mov     [ebp - 4], eax\n"
        "        mov     eax, 0\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("compound_assign_collapse", 0) == 0


def test_compound_assign_collapse_skips_when_ecx_live():
    """If ECX is live after the store (read by subsequent code),
    we can't drop the `mov ecx, eax` step."""
    asm = (
        "_f:\n"
        "        push    dword [ebp - 4]\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     ecx, eax\n"
        "        pop     eax\n"
        "        add     eax, ecx\n"
        "        mov     [ebp - 4], eax\n"
        "        mov     [ebp - 8], ecx\n"  # reads ECX!
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("compound_assign_collapse", 0) == 0


def test_compound_assign_collapse_global_destination():
    """Global memory (`[_var]`) also works."""
    asm = (
        "_f:\n"
        "        push    dword [_glob]\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     ecx, eax\n"
        "        pop     eax\n"
        "        add     eax, ecx\n"
        "        mov     [_glob], eax\n"
        "        mov     eax, 0\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "add     [_glob], eax" in out
    assert opt.stats.get("compound_assign_collapse") == 1


def test_compound_assign_collapse_short_tail_commutative():
    """The codegen's commutative-OP shortcut: ``pop ecx; OP eax,
    ecx; mov [m], eax`` — only 3-line tail, no save-restore.

    For commutative OPs (add/and/or/xor), ``rhs OP lhs == lhs OP
    rhs``, so the codegen pops lhs into ECX and ops directly.
    """
    asm = (
        "_f:\n"
        "        push    dword [ebp - 4]\n"
        "        mov     eax, [ebp + 8]\n"  # chain: load rhs
        "        pop     ecx\n"  # pop lhs into ecx (3-line tail)
        "        add     eax, ecx\n"
        "        mov     [ebp - 4], eax\n"
        "        mov     eax, 0\n"  # ECX dead witness
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "add     [ebp - 4], eax" in out
    assert "push    dword [ebp - 4]" not in out
    assert "pop     ecx" not in out
    assert opt.stats.get("compound_assign_collapse") == 1


def test_compound_assign_collapse_short_tail_xor():
    """``xor`` is commutative — short tail works."""
    asm = (
        "_f:\n"
        "        push    dword [ebp - 4]\n"
        "        mov     eax, [ebp + 8]\n"
        "        pop     ecx\n"
        "        xor     eax, ecx\n"
        "        mov     [ebp - 4], eax\n"
        "        xor     eax, eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "xor     [ebp - 4], eax" in out
    assert opt.stats.get("compound_assign_collapse") == 1


def test_compound_assign_collapse_short_tail_rejects_sub():
    """``sub`` is NOT commutative — short tail must NOT match.

    With ``pop ecx; sub eax, ecx; mov [m], eax``, the result is
    ``rhs - lhs``, but the C semantics is ``lhs - rhs``. The
    codegen never emits this short form for sub, but defend
    against false matches anyway.
    """
    asm = (
        "_f:\n"
        "        push    dword [ebp - 4]\n"
        "        mov     eax, [ebp + 8]\n"
        "        pop     ecx\n"
        "        sub     eax, ecx\n"
        "        mov     [ebp - 4], eax\n"
        "        mov     eax, 0\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("compound_assign_collapse", 0) == 0


def test_compound_assign_collapse_short_tail_balanced_inner_pushpop():
    """Chain may contain balanced inner push/pop pairs (e.g.
    nested array indexing). Walk-back tracks stack depth so they
    don't trip the bail-on-stack-manipulation check.
    """
    asm = (
        "_f:\n"
        "        push    dword [ebp - 4]\n"  # outer push (lhs)
        "        push    _g\n"  # inner push (indexing base)
        "        mov     eax, [ebp - 8]\n"
        "        shl     eax, 2\n"
        "        pop     ecx\n"  # inner pop (matches inner push)
        "        add     eax, ecx\n"
        "        mov     eax, [eax]\n"  # deref → eax = g[i]
        "        pop     ecx\n"  # outer pop (matches outer push)
        "        add     eax, ecx\n"
        "        mov     [ebp - 4], eax\n"
        "        mov     eax, 0\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # The compound-assign frame collapses to in-place add.
    assert "add     [ebp - 4], eax" in out
    assert opt.stats.get("compound_assign_collapse") == 1


def test_compound_assign_collapse_long_tail_balanced_inner_pushpop():
    """Same balanced-inner-pushpop fix applies to the canonical
    4-line tail too.
    """
    asm = (
        "_f:\n"
        "        push    dword [ebp - 4]\n"
        "        push    _g\n"  # inner push
        "        mov     eax, [ebp - 8]\n"
        "        shl     eax, 2\n"
        "        pop     ecx\n"  # inner pop
        "        add     eax, ecx\n"
        "        mov     eax, [eax]\n"
        "        mov     ecx, eax\n"  # 4-line canonical tail begins
        "        pop     eax\n"
        "        sub     eax, ecx\n"
        "        mov     [ebp - 4], eax\n"
        "        mov     eax, 0\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "sub     [ebp - 4], eax" in out
    assert opt.stats.get("compound_assign_collapse") == 1


# ── redundant_xor_zero ───────────────────────────────────────────


def test_redundant_xor_zero_basic():
    """`xor eax, eax; mov [m], eax; xor eax, eax` — drop the second
    xor, eax already 0."""
    asm = (
        "_f:\n"
        "        xor     eax, eax\n"
        "        mov     [ebp - 4], eax\n"
        "        xor     eax, eax\n"
        "        mov     [ebp - 8], eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # Only one `xor eax, eax` should remain.
    assert out.count("xor     eax, eax") == 1
    assert opt.stats.get("redundant_xor_zero") == 1


def test_redundant_xor_zero_preserves_first():
    """The first xor stays — it's the actual zero-setter."""
    asm = (
        "_f:\n"
        "        xor     eax, eax\n"
        "        mov     [ebp - 4], eax\n"
        "        xor     eax, eax\n"  # redundant
        "        mov     [ebp - 8], eax\n"
        "        xor     eax, eax\n"  # also redundant
        "        mov     [ebp - 12], eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert out.count("xor     eax, eax") == 1
    assert opt.stats.get("redundant_xor_zero") == 2


def test_redundant_xor_zero_invalidates_after_eax_write():
    """A write to EAX (e.g., `mov eax, X`) invalidates the 'zero'
    state. The next `xor eax, eax` is then NOT redundant.

    We keep both intermediate instructions live by having them
    used downstream so dead_mov_to_reg doesn't drop them."""
    asm = (
        "_f:\n"
        "        xor     eax, eax\n"
        "        mov     [ebp - 4], eax\n"  # uses eax (live)
        "        mov     eax, [ebp + 8]\n"  # writes eax with a non-zero value
        "        mov     [ebp - 8], eax\n"  # uses eax (live)
        "        xor     eax, eax\n"  # NOT redundant
        "        mov     [ebp - 12], eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("redundant_xor_zero", 0) == 0


def test_redundant_xor_zero_invalidates_at_label():
    """Labels are control-flow merge points — multiple incoming
    paths could leave EAX in different states."""
    asm = (
        "_f:\n"
        "        xor     eax, eax\n"
        ".target:\n"
        "        xor     eax, eax\n"  # NOT redundant — could be entry from another path
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("redundant_xor_zero", 0) == 0


def test_redundant_xor_zero_invalidates_after_call():
    """Calls clobber EAX (return value)."""
    asm = (
        "_f:\n"
        "        xor     eax, eax\n"
        "        call    _g\n"
        "        xor     eax, eax\n"  # NOT redundant
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("redundant_xor_zero", 0) == 0


def test_redundant_xor_zero_per_register():
    """Tracking is per-register: zero state for ECX is independent
    of EAX."""
    asm = (
        "_f:\n"
        "        xor     eax, eax\n"
        "        xor     ecx, ecx\n"
        "        mov     [ebp - 4], eax\n"
        "        mov     [ebp - 8], ecx\n"
        "        xor     eax, eax\n"  # redundant
        "        xor     ecx, ecx\n"  # redundant
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("redundant_xor_zero") == 2


# ── zero_init_collapse ───────────────────────────────────────────


def test_zero_init_collapse_dword_chain():
    """Adjacent `mov dword [m], 0` stores collapse to a single
    `xor eax, eax` followed by `mov [m], eax` per store."""
    asm = (
        "_main:\n"
        "        mov     dword [ebp - 4], 0\n"
        "        mov     dword [ebp - 8], 0\n"
        "        mov     dword [ebp - 12], 0\n"
        "        mov     eax, [ebp - 4]\n"  # witness: overwrites EAX
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "xor     eax, eax" in out
    assert "mov     [ebp - 4], eax" in out
    assert "mov     [ebp - 8], eax" in out
    assert "mov     [ebp - 12], eax" in out
    assert "mov     dword [ebp" not in out
    assert opt.stats.get("zero_init_collapse") == 3


def test_zero_init_collapse_single_store():
    """Even a single `mov dword [m], 0` is rewritten — the upfront
    `xor eax, eax` (2 bytes) plus `mov [m], eax` (3 bytes) totals
    5 bytes, vs the original 7 bytes. Saves 2 bytes."""
    asm = (
        "_f:\n"
        "        mov     dword [ebp - 4], 0\n"
        "        mov     eax, 1\n"  # witness
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "xor     eax, eax" in out
    assert "mov     [ebp - 4], eax" in out
    assert opt.stats.get("zero_init_collapse") == 1


def test_zero_init_collapse_byte_form():
    """Byte zero-stores use AL: `mov byte [m], 0` → `mov [m], al`
    after `xor eax, eax`."""
    asm = (
        "_f:\n"
        "        mov     byte [ebp - 1], 0\n"
        "        mov     byte [ebp - 2], 0\n"
        "        mov     eax, 1\n"  # witness
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "xor     eax, eax" in out
    assert "mov     [ebp - 1], al" in out
    assert "mov     [ebp - 2], al" in out
    assert opt.stats.get("zero_init_collapse") == 2


def test_zero_init_collapse_skips_when_eax_live():
    """If EAX is live after the chain, we can't clobber it via xor."""
    asm = (
        "_f:\n"
        "        mov     dword [ebp - 4], 0\n"
        "        ret\n"  # ret reads EAX (return value)
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("zero_init_collapse", 0) == 0


def test_zero_init_collapse_skips_when_flag_reader_after():
    """If the next instr reads flags (e.g. jc/jnc), the xor's flag
    side effect would change the branch outcome."""
    asm = (
        "_f:\n"
        "        mov     dword [ebp - 4], 0\n"
        "        jc      .skip\n"  # flag-reader before any clobberer
        "        mov     eax, 1\n"
        "        ret\n"
        ".skip:\n"
        "        mov     eax, 2\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("zero_init_collapse", 0) == 0


def test_zero_init_collapse_stops_at_non_zero_store():
    """A non-zero store breaks the chain — only consecutive zero
    stores collapse."""
    asm = (
        "_f:\n"
        "        mov     dword [ebp - 4], 0\n"
        "        mov     dword [ebp - 8], 5\n"  # not zero
        "        mov     dword [ebp - 12], 0\n"
        "        mov     eax, 1\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # First chain: just [ebp - 4], single store.
    assert "xor     eax, eax" in out
    assert "mov     [ebp - 4], eax" in out
    # The non-zero store is preserved.
    assert "mov     dword [ebp - 8], 5" in out
    # Stats: 1 + 1 = 2 (one for first chain, one for last chain).
    assert opt.stats.get("zero_init_collapse") == 2


# ── jcc_jmp_inversion ────────────────────────────────────────────


def test_jcc_jmp_inversion_basic():
    """`jle L1; jmp L2; L1:` → `jg L2; L1:`."""
    asm = (
        "_f:\n"
        "        cmp     eax, ebx\n"
        "        jle     .L1\n"
        "        jmp     .L2\n"
        ".L1:\n"
        "        mov     eax, ebx\n"
        ".L2:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "jg      .L2" in out
    assert "jmp     .L2" not in out
    assert opt.stats.get("jcc_jmp_inversion") == 1


def test_jcc_jmp_inversion_je_to_jne():
    """`je L1; jmp L2; L1:` → `jne L2; L1:`."""
    asm = (
        "_f:\n"
        "        cmp     eax, 0\n"
        "        je      .L1\n"
        "        jmp     .L2\n"
        ".L1:\n"
        "        mov     eax, 1\n"
        ".L2:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "jne     .L2" in out
    assert opt.stats.get("jcc_jmp_inversion") == 1


def test_jcc_jmp_inversion_jb_to_jae():
    """Unsigned: `jb L1; jmp L2; L1:` → `jae L2; L1:`."""
    asm = (
        "_f:\n"
        "        cmp     eax, ebx\n"
        "        jb      .L1\n"
        "        jmp     .L2\n"
        ".L1:\n"
        "        mov     eax, 1\n"
        ".L2:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "jae     .L2" in out
    assert opt.stats.get("jcc_jmp_inversion") == 1


def test_jcc_jmp_inversion_skips_when_label_not_target():
    """The label after jmp must match the jcc's target."""
    asm = (
        "_f:\n"
        "        cmp     eax, ebx\n"
        "        jle     .L1\n"
        "        jmp     .L2\n"
        ".OTHER:\n"  # not L1 — pattern shouldn't match
        "        mov     eax, ebx\n"
        ".L2:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("jcc_jmp_inversion", 0) == 0


def test_jcc_jmp_inversion_tolerates_blanks_between():
    """Blanks/comments between jcc, jmp, and label are OK."""
    asm = (
        "_f:\n"
        "        cmp     eax, ebx\n"
        "        jle     .L1\n"
        "        ; comment\n"
        "        jmp     .L2\n"
        "        ; another comment\n"
        ".L1:\n"
        "        mov     eax, ebx\n"
        ".L2:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "jg      .L2" in out
    assert opt.stats.get("jcc_jmp_inversion") == 1


def test_jcc_jmp_inversion_skips_jmp_jmp():
    """Two consecutive jmp's aren't a jcc-jmp pattern."""
    asm = (
        "_f:\n"
        "        jmp     .L1\n"
        "        jmp     .L2\n"
        ".L1:\n"
        "        ret\n"
        ".L2:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("jcc_jmp_inversion", 0) == 0


# ── redundant_eax_load: read-only ops + jcc fall-through ─────────


def test_redundant_eax_load_through_cmp():
    """`cmp eax, X` reads EAX but doesn't write — eax_mem stays
    valid across cmp."""
    asm = (
        "_max:\n"
        "        mov     eax, [ebp + 8]\n"
        "        cmp     eax, [ebp + 12]\n"
        "        mov     eax, [ebp + 8]\n"  # redundant
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # Only one `mov eax, [ebp + 8]` should remain — the second is
    # dropped (eax already holds [ebp + 8] after the cmp).
    assert out.count("mov     eax, [ebp + 8]") == 1
    assert opt.stats.get("redundant_eax_load") == 1


def test_redundant_eax_load_through_test():
    """`test eax, eax` is read-only on EAX — eax_mem stays valid.

    Note: `mov eax, [mem]; test eax, eax` adjacent is consumed by
    `cmp_load_collapse` first. We add an unrelated instruction
    between to force this case to flow through redundant_eax_load."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        add     ebx, 1\n"  # blocks cmp_load_collapse
        "        test    eax, eax\n"
        "        mov     eax, [ebp + 8]\n"  # redundant
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert out.count("mov     eax, [ebp + 8]") == 1
    assert opt.stats.get("redundant_eax_load") == 1


def test_redundant_eax_load_through_push_eax():
    """`push eax` reads EAX but doesn't write — eax_mem stays valid.

    Note: a tight ``mov eax, [mem]; push eax`` pair would be eaten by
    ``push_memory``. To test only the redundant-load behavior across
    a push, we put another use of EAX between the load and the push,
    and an OUTSIDE redundant load after the push."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        add     eax, eax\n"  # uses eax — push_memory can't fire
        "        push    eax\n"
        "        mov     eax, [ebp + 8]\n"  # not redundant (eax was modified)
        "        pop     edx\n"  # different reg avoids push_memory tail
        "        ret\n"
    )
    # push_memory only fires when src is an exact mem operand of mov.
    # The above is *literally* not a push_memory candidate. The
    # `mov eax, [ebp + 8]` after the push IS NOT redundant (eax got
    # modified by `add eax, eax`). This is a placeholder showing the
    # tracker correctly invalidates after EAX-modifying ops.
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    # Tracker reset after `add eax, eax`, so the second mov isn't
    # redundant.
    assert opt.stats.get("redundant_eax_load", 0) == 0


def test_redundant_eax_load_through_jcc_fallthrough():
    """Conditional jumps (jcc) preserve EAX state on the fallthrough
    path. `mov eax, M; cmp eax, X; jcc L; mov eax, M` — the second
    load is redundant in the fallthrough path."""
    asm = (
        "_max:\n"
        "        mov     eax, [ebp + 8]\n"
        "        cmp     eax, [ebp + 12]\n"
        "        jle     .L1\n"
        "        mov     eax, [ebp + 8]\n"  # redundant on fallthrough
        "        jmp     .L2\n"
        ".L1:\n"
        "        mov     eax, [ebp + 12]\n"
        ".L2:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # Two `mov eax, [ebp + 8]` originally; one is dropped (redundant).
    assert out.count("mov     eax, [ebp + 8]") == 1
    assert opt.stats.get("redundant_eax_load") == 1


def test_redundant_eax_load_invalidates_after_label():
    """Labels are merge points — multiple incoming paths could put
    different values in EAX. After a label, the tracker must reset.

    To prove that, we keep the first mov live and avoid passes that
    would consume it (cmp_load_collapse, dead_mov_to_reg)."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        push    eax\n"  # uses eax (keeps the load alive)
        ".target:\n"  # label invalidates tracking
        "        mov     eax, [ebp + 8]\n"  # NOT redundant
        "        pop     edx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    # Tracker invalidates at .target:, so the second mov is NOT
    # treated as redundant.
    assert opt.stats.get("redundant_eax_load", 0) == 0


def test_redundant_eax_load_invalidates_after_call():
    """`call` clobbers EAX (return value)."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        push    eax\n"  # uses eax (keeps the load alive)
        "        call    _g\n"
        "        add     esp, 4\n"
        "        mov     eax, [ebp + 8]\n"  # NOT redundant — call wrote eax
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # The first mov+push pair gets push_memory'd to `push dword [...]`,
    # so the FIRST mov is consumed — but the SECOND mov is preserved.
    assert out.count("mov     eax, [ebp + 8]") == 1
    assert opt.stats.get("redundant_eax_load", 0) == 0


def test_redundant_eax_load_invalidates_after_unconditional_jmp():
    """An unconditional `jmp` also invalidates — the next text
    instruction is unreachable except via a label, where we'd reset
    again. Conservative: reset at jmp. (jcc keeps tracking — that's
    the previous test.)"""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        jmp     .else\n"
        "        mov     eax, [ebp + 8]\n"  # unreachable as fallthrough
        ".else:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    # The unreachable mov gets dropped by `dead_after_terminator`,
    # not by redundant_eax_load.
    assert opt.stats.get("redundant_eax_load", 0) == 0


# ── push_memory ──────────────────────────────────────────────────


def test_push_memory_basic():
    """`mov eax, [mem]; push eax` → `push dword [mem]`."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp - 4]\n"
        "        push    eax\n"
        "        call    _g\n"  # clobbers eax (witness)
        "        add     esp, 4\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "push    dword [ebp - 4]" in out
    assert "mov     eax, [ebp - 4]" not in out
    assert opt.stats.get("push_memory") == 1


def test_push_memory_label_addressed():
    """Memory operand can be a label-addressed global."""
    asm = (
        "_f:\n"
        "        mov     eax, [_g]\n"
        "        push    eax\n"
        "        call    _h\n"
        "        add     esp, 4\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "push    dword [_g]" in out
    assert opt.stats.get("push_memory") == 1


def test_push_memory_skips_when_eax_live_after_push():
    """If EAX is read after the push (e.g. used as another arg or
    return value), the load can't be dropped."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp - 4]\n"
        "        push    eax\n"
        "        push    eax\n"  # eax is used again
        "        call    _g\n"
        "        add     esp, 8\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("push_memory", 0) == 0


def test_push_memory_skips_esp_relative():
    """ESP-relative memory operands are skipped — push decrements
    ESP, so mov-then-push of [esp + N] would shift between the two
    operations."""
    asm = (
        "_f:\n"
        "        mov     eax, [esp + 8]\n"
        "        push    eax\n"
        "        call    _g\n"
        "        add     esp, 4\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("push_memory", 0) == 0


def test_push_memory_works_for_other_regs():
    """Pattern fires for any of the supported regs."""
    for reg in ("ebx", "ecx", "edx", "esi", "edi", "ebp"):
        asm = (
            "_f:\n"
            f"        mov     {reg}, [ebp - 4]\n"
            f"        push    {reg}\n"
            f"        mov     {reg}, 0\n"  # witness: overwrites reg
            "        ret\n"
        )
        opt = PeepholeOptimizer()
        out = opt.optimize(asm)
        assert "push    dword [ebp - 4]" in out, f"failed for {reg}"
        assert opt.stats.get("push_memory") == 1


def test_push_memory_runs_after_right_operand_retarget():
    """The inner save-restore pattern
    `push eax / chain / mov ecx, eax / pop eax` is a
    right_operand_retarget candidate. push_memory runs AFTER that
    pass so it doesn't disrupt it.

    Here the inner mov+push is `mov eax, [ebp + 8]; push eax`
    followed by a mov-eax chain. right_operand_retarget should
    retarget the chain to ECX and remove the push/pop framing
    entirely. push_memory should NOT fire on that mov+push."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        push    eax\n"
        "        mov     eax, [ebp - 8]\n"
        "        shl     eax, 2\n"
        "        mov     ecx, eax\n"
        "        pop     eax\n"
        "        add     eax, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # right_operand_retarget collapsed the push/pop frame.
    assert "push    eax" not in out
    assert "pop     eax" not in out
    # push_memory didn't fire (the mov+push was consumed by retarget).
    assert opt.stats.get("push_memory", 0) == 0


def test_narrowing_load_test_collapse_skips_signed_jcc():
    """Regression: torture's doloop-1 uses
    ``unsigned char z; --z > 0`` which expands to
    ``movzx eax, byte; setg; ...; test eax, eax; jnz``. The setcc-jcc
    collapse turns this into ``movzx eax, byte; cmp eax, 0; jg`` —
    wait, actually after several other passes the chain becomes
    ``movzx eax, byte; test eax, eax; jg``. If we'd rewrite that to
    ``cmp byte, 0; jg``, the byte's signed comparison gives the wrong
    answer for z=255 (treats 0xFF as -1, jg false; original treats
    0x000000FF as 255, jg true).

    The pass restricts itself to jz/jnz/je/jne (only ZF read) so the
    rewrite never applies in the signed-Jcc case."""
    asm = (
        "_f:\n"
        "        movzx   eax, byte [ebp - 4]\n"
        "        test    eax, eax\n"
        "        jg      .top\n"
        "        mov     eax, 0\n"
        "        ret\n"
        ".top:\n"
        "        mov     eax, 1\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("narrowing_load_test_collapse", 0) == 0


def test_narrowing_load_test_collapse_skips_unsigned_jcc():
    """Same restriction for unsigned Jcc — `ja`/`jb` read CF, which
    differs between byte cmp and dword cmp/test."""
    asm = (
        "_f:\n"
        "        movzx   eax, byte [esi]\n"
        "        test    eax, eax\n"
        "        ja      .top\n"
        "        mov     eax, 0\n"
        "        ret\n"
        ".top:\n"
        "        mov     eax, 1\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("narrowing_load_test_collapse", 0) == 0


def test_narrowing_load_test_collapse_strlen_pattern():
    """The classic `while (*p)` byte-pointer loop ends up tighter:
    no movsx, just cmp byte through the pointer."""
    asm = (
        "_strlen:\n"
        "        enter   4, 0\n"
        "        xor     eax, eax\n"
        "        mov     [ebp - 4], eax\n"
        ".top:\n"
        "        mov     eax, [ebp + 8]\n"
        "        movsx   eax, byte [eax]\n"
        "        test    eax, eax\n"
        "        jz      .end\n"
        "        inc     dword [ebp + 8]\n"
        "        inc     dword [ebp - 4]\n"
        "        jmp     .top\n"
        ".end:\n"
        "        mov     eax, [ebp - 4]\n"
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # The narrowing-load collapse fires once for the byte-deref.
    assert opt.stats.get("narrowing_load_test_collapse") == 1
    # The pointer is now read as `cmp byte [eax], 0` directly.
    assert "cmp     byte [eax], 0" in out
    assert "movsx" not in out


# ── disp_load_collapse ───────────────────────────────────────────


def test_disp_load_collapse_basic():
    """`add reg, N; mov reg, [reg]` → `mov reg, [reg + N]`."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        add     eax, 4\n"
        "        mov     eax, [eax]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     eax, [eax + 4]" in out
    assert "add     eax, 4" not in out
    assert opt.stats.get("disp_load_collapse") == 1


def test_disp_load_collapse_distinct_dst():
    """`add base, N; mov dst, [base]` works when base is dead after."""
    asm = (
        "_f:\n"
        "        mov     ecx, [ebp + 8]\n"
        "        add     ecx, 12\n"
        "        mov     eax, [ecx]\n"
        "        ret\n"  # ecx dead after the mov
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     eax, [ecx + 12]" in out
    assert opt.stats.get("disp_load_collapse") == 1


def test_disp_load_collapse_skips_when_base_live_distinct():
    """If base is live after and dst != base, the add can't be dropped."""
    asm = (
        "_f:\n"
        "        mov     ecx, [ebp + 8]\n"
        "        add     ecx, 12\n"
        "        mov     eax, [ecx]\n"
        "        mov     [ebp - 4], ecx\n"  # ecx still needed
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("disp_load_collapse", 0) == 0


def test_disp_load_collapse_negative_disp():
    """Negative DISP collapses with `-` sign in the operand."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        add     eax, -8\n"
        "        mov     eax, [eax]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     eax, [eax - 8]" in out
    assert opt.stats.get("disp_load_collapse") == 1


def test_disp_load_collapse_rejects_existing_offset():
    """If the load already has `[reg + N]`, can't add more displacement."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        add     eax, 4\n"
        "        mov     eax, [eax + 8]\n"  # already has offset
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("disp_load_collapse", 0) == 0


def test_disp_load_collapse_rejects_non_literal_disp():
    """If DISP isn't a numeric literal (e.g. a label), skip."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        add     eax, _some_label\n"
        "        mov     eax, [eax]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("disp_load_collapse", 0) == 0


def test_disp_load_collapse_struct_member_pattern():
    """The canonical `p->y` pattern: load p, add offset, deref."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        add     eax, 4\n"
        "        mov     eax, [eax]\n"
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # Three instr lines collapsed into two:
    assert "add     eax" not in out
    assert "mov     eax, [eax + 4]" in out
    assert opt.stats.get("disp_load_collapse") == 1


# ── push_disp_collapse ───────────────────────────────────────────


def test_push_disp_collapse_basic():
    """`add reg, N; push dword [reg]` → `push dword [reg + N]`."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        add     eax, 8\n"
        "        push    dword [eax]\n"
        "        call    _consumer\n"
        "        add     esp, 4\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "push    dword [eax + 8]" in out
    assert "add     eax, 8" not in out
    assert opt.stats.get("push_disp_collapse") == 1


def test_push_disp_collapse_skips_when_reg_live():
    """If REG is read after the push, can't drop the add."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        add     eax, 8\n"
        "        push    dword [eax]\n"
        "        mov     [ebp - 4], eax\n"  # eax still needed
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("push_disp_collapse", 0) == 0


def test_push_disp_collapse_negative_disp():
    """Negative DISP collapses with `-` sign."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        add     eax, -16\n"
        "        push    dword [eax]\n"
        "        call    _consumer\n"
        "        add     esp, 4\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "push    dword [eax - 16]" in out
    assert opt.stats.get("push_disp_collapse") == 1


# ── push_index_collapse ──────────────────────────────────────────


def test_push_index_collapse_scale_4():
    """`shl idx, 2; add base, idx; push dword [base]` →
    `push dword [base + idx*4]`."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     ecx, [ebp - 8]\n"
        "        shl     ecx, 2\n"
        "        add     eax, ecx\n"
        "        push    dword [eax]\n"
        "        call    _consumer\n"
        "        add     esp, 4\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "push    dword [eax + ecx*4]" in out
    assert "shl     ecx" not in out
    assert "add     eax, ecx" not in out
    assert opt.stats.get("push_index_collapse") == 1


def test_push_index_collapse_skips_when_base_live():
    """If BASE is read after the push, can't drop the add."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     ecx, [ebp - 8]\n"
        "        shl     ecx, 2\n"
        "        add     eax, ecx\n"
        "        push    dword [eax]\n"
        "        mov     [ebp - 4], eax\n"  # eax still needed
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("push_index_collapse", 0) == 0


def test_push_index_collapse_skips_when_idx_live():
    """If IDX is read after the push, can't drop the shl."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     ecx, [ebp - 8]\n"
        "        shl     ecx, 2\n"
        "        add     eax, ecx\n"
        "        push    dword [eax]\n"
        "        mov     [ebp - 4], ecx\n"  # ecx still needed
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("push_index_collapse", 0) == 0


def test_push_index_collapse_scale_8():
    """Scale 8 (long-long arrays)."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     ecx, [ebp - 8]\n"
        "        shl     ecx, 3\n"
        "        add     eax, ecx\n"
        "        push    dword [eax]\n"
        "        call    _consumer\n"
        "        add     esp, 4\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "push    dword [eax + ecx*8]" in out
    assert opt.stats.get("push_index_collapse") == 1


# ── redundant_ecx_load ───────────────────────────────────────────


def test_redundant_ecx_load_basic():
    """`mov ecx, [m]; <push that doesn't touch ecx>; mov ecx, [m]`
    drops the second load."""
    asm = (
        "_f:\n"
        "        mov     ecx, [ebp - 8]\n"
        "        push    dword [eax + ecx*4]\n"  # reads ecx, doesn't write
        "        mov     ecx, [ebp - 8]\n"  # redundant
        "        inc     ecx\n"
        "        mov     eax, [eax + ecx*4]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # Only one `mov ecx, [ebp - 8]` should remain.
    assert out.count("mov     ecx, [ebp - 8]") == 1
    assert opt.stats.get("redundant_ecx_load") == 1


def test_redundant_ecx_load_invalidates_after_ecx_write():
    """If ECX is written between two loads, the second isn't redundant."""
    asm = (
        "_f:\n"
        "        mov     ecx, [ebp - 8]\n"
        "        mov     ecx, eax\n"  # ECX overwritten
        "        mov     ecx, [ebp - 8]\n"  # NOT redundant (ECX changed)
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("redundant_ecx_load", 0) == 0


def test_redundant_ecx_load_invalidates_after_call():
    """A call clobbers ECX (cdecl scratch); reload is needed."""
    asm = (
        "_f:\n"
        "        mov     ecx, [ebp - 8]\n"
        "        call    _other\n"  # clobbers ECX
        "        mov     ecx, [ebp - 8]\n"  # NOT redundant
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("redundant_ecx_load", 0) == 0


def test_redundant_ecx_load_invalidates_after_cl_write():
    """A write to CL (sub-register) invalidates ECX tracking."""
    asm = (
        "_f:\n"
        "        mov     ecx, [ebp - 8]\n"
        "        mov     cl, 0\n"  # CL overwritten
        "        mov     ecx, [ebp - 8]\n"  # NOT redundant
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("redundant_ecx_load", 0) == 0


def test_redundant_ecx_load_invalidates_after_shl_cl():
    """`shl reg, cl` reads CL — but doesn't write ECX. So a subsequent
    `mov ecx, [m]` IS redundant if [m] was already loaded into ecx."""
    asm = (
        "_f:\n"
        "        mov     ecx, [ebp - 8]\n"
        "        shl     eax, cl\n"  # reads CL, doesn't write
        "        mov     ecx, [ebp - 8]\n"  # actually redundant
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    # `shl eax, cl` references the ecx family in operands, so tracker
    # invalidates conservatively. The pass treats "any reference" as
    # potentially clobbering, even though shl reg, cl only reads.
    # Conservative is correct; we won't drop the second mov.
    assert opt.stats.get("redundant_ecx_load", 0) == 0


def test_redundant_ecx_load_through_jcc_fallthrough():
    """A jcc's fallthrough preserves ECX tracking."""
    asm = (
        "_f:\n"
        "        mov     ecx, [ebp - 8]\n"
        "        cmp     eax, 5\n"
        "        jl      .else\n"
        "        mov     ecx, [ebp - 8]\n"  # redundant on fallthrough
        "        ret\n"
        ".else:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("redundant_ecx_load") == 1


# ── self_mov_elimination ─────────────────────────────────────────


def test_self_mov_elimination_basic():
    """`mov reg, reg` (dst == src) is a no-op — drop it."""
    asm = (
        "_f:\n"
        "        mov     ecx, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     ecx, ecx" not in out
    assert opt.stats.get("self_mov_elimination") == 1


def test_self_mov_elimination_after_pop():
    """The common pattern: `pop ecx; add esp, 4; mov ecx, ecx`."""
    asm = (
        "_f:\n"
        "        push    eax\n"
        "        push    eax\n"
        "        call    _other\n"
        "        pop     ecx\n"
        "        add     esp, 4\n"
        "        mov     ecx, ecx\n"
        "        add     eax, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     ecx, ecx" not in out
    assert opt.stats.get("self_mov_elimination") == 1


def test_self_mov_elimination_eax():
    """`mov eax, eax` also dropped."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     eax, eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     eax, eax" not in out
    assert opt.stats.get("self_mov_elimination") == 1


def test_self_mov_elimination_keeps_distinct_regs():
    """`mov eax, ecx` (dst != src) is NOT dropped."""
    asm = (
        "_f:\n"
        "        mov     eax, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     eax, ecx" in out
    assert opt.stats.get("self_mov_elimination", 0) == 0


def test_self_mov_elimination_keeps_mov_with_memory():
    """`mov [eax], eax` (memory dest) is NOT dropped."""
    asm = (
        "_f:\n"
        "        mov     [eax], eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     [eax], eax" in out
    assert opt.stats.get("self_mov_elimination", 0) == 0


# ── transfer_pop_collapse ────────────────────────────────────────


def test_transfer_pop_collapse_add():
    """`mov ecx, eax; pop eax; add eax, ecx` → `pop ecx; add eax, ecx`.
    Add is commutative.

    Use a chain that touches ECX so right_operand_retarget can't fire
    (it requires all chain ops to be pure writes to EAX).
    """
    asm = (
        "_f:\n"
        "        push    eax\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     ecx, [ebp - 8]\n"
        "        mov     eax, [eax + ecx*4]\n"
        "        mov     ecx, eax\n"
        "        pop     eax\n"
        "        add     eax, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     ecx, eax" not in out
    assert "pop     ecx" in out
    assert "add     eax, ecx" in out
    assert opt.stats.get("transfer_pop_collapse") == 1


def test_transfer_pop_collapse_imul():
    """imul (two-operand) is also commutative."""
    asm = (
        "_f:\n"
        "        push    eax\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     ecx, [ebp - 8]\n"
        "        mov     eax, [eax + ecx*4]\n"
        "        mov     ecx, eax\n"
        "        pop     eax\n"
        "        imul    eax, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "pop     ecx" in out
    assert "imul    eax, ecx" in out
    assert opt.stats.get("transfer_pop_collapse") == 1


def test_transfer_pop_collapse_skips_sub():
    """sub is NOT commutative — must not collapse."""
    asm = (
        "_f:\n"
        "        push    eax\n"
        "        mov     eax, [ebp - 4]\n"
        "        mov     ecx, eax\n"
        "        pop     eax\n"
        "        sub     eax, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("transfer_pop_collapse", 0) == 0


def test_transfer_pop_collapse_skips_cmp():
    """cmp's operand order matters for signed comparisons — skip."""
    asm = (
        "_f:\n"
        "        push    eax\n"
        "        mov     eax, [ebp - 4]\n"
        "        mov     ecx, eax\n"
        "        pop     eax\n"
        "        cmp     eax, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("transfer_pop_collapse", 0) == 0


def test_transfer_pop_collapse_skips_idiv():
    """idiv reads EDX:EAX and is NOT commutative."""
    asm = (
        "_f:\n"
        "        push    eax\n"
        "        mov     eax, [ebp - 4]\n"
        "        mov     ecx, eax\n"
        "        pop     eax\n"
        "        idiv    ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("transfer_pop_collapse", 0) == 0


def test_transfer_pop_collapse_xor():
    """xor is commutative."""
    asm = (
        "_f:\n"
        "        push    eax\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     ecx, [ebp - 8]\n"
        "        mov     eax, [eax + ecx*4]\n"
        "        mov     ecx, eax\n"
        "        pop     eax\n"
        "        xor     eax, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "pop     ecx" in out
    assert "xor     eax, ecx" in out
    assert opt.stats.get("transfer_pop_collapse") == 1


# ── label_load_collapse ──────────────────────────────────────────


def test_label_load_collapse_basic():
    """`mov eax, _label; mov ecx, [eax]` → `mov ecx, [_label]`
    when EAX is dead after."""
    asm = (
        "_f:\n"
        "        mov     eax, _glob\n"
        "        mov     ecx, [eax]\n"
        "        mov     eax, ecx\n"  # eax overwritten (dead before)
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     ecx, [_glob]" in out
    assert opt.stats.get("label_load_collapse") == 1


def test_label_load_collapse_same_reg():
    """`mov eax, _label; mov eax, [eax]` → `mov eax, [_label]`
    (no liveness check needed since the second mov overwrites)."""
    asm = (
        "_f:\n"
        "        mov     eax, _glob\n"
        "        mov     eax, [eax]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     eax, [_glob]" in out
    assert opt.stats.get("label_load_collapse") == 1


def test_label_load_collapse_label_arithmetic():
    """The label can be a label-arithmetic expression."""
    asm = (
        "_f:\n"
        "        mov     eax, _b + 8\n"
        "        mov     ecx, [eax]\n"
        "        mov     eax, ecx\n"  # eax overwritten
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     ecx, [_b + 8]" in out
    assert opt.stats.get("label_load_collapse") == 1


def test_label_load_collapse_sib():
    """SIB form: `mov eax, _label; mov ecx, [eax + ecx*4]` →
    `mov ecx, [_label + ecx*4]`."""
    asm = (
        "_f:\n"
        "        mov     eax, _arr\n"
        "        mov     ecx, [eax + ecx*4]\n"
        "        mov     eax, ecx\n"  # eax overwritten
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # The new mov uses ecx as both dest and idx — that's OK in
    # x86 (the read of ecx happens before the write).
    assert "mov     ecx, [_arr + ecx*4]" in out
    assert opt.stats.get("label_load_collapse") == 1


def test_label_load_collapse_skips_when_eax_live():
    """If EAX is read after the deref, can't drop the address load."""
    asm = (
        "_f:\n"
        "        mov     eax, _glob\n"
        "        mov     ecx, [eax]\n"
        "        mov     edx, eax\n"  # eax live after
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("label_load_collapse", 0) == 0


def test_label_load_collapse_skips_numeric_imm():
    """If the source is a numeric immediate, don't fold (NASM
    can't dereference an arbitrary integer in disp32 form without
    losing semantics — the numeric value is the address but more
    typically a misaddressed pointer)."""
    asm = (
        "_f:\n"
        "        mov     eax, 42\n"
        "        mov     ecx, [eax]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("label_load_collapse", 0) == 0


def test_label_load_collapse_skips_existing_offset():
    """Source has [eax + N] (not just [eax]) — skip."""
    asm = (
        "_f:\n"
        "        mov     eax, _glob\n"
        "        mov     ecx, [eax + 4]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    # disp form doesn't match — the source must be `[eax]` or
    # `[eax + idx*scale]`, not `[eax + N]`.
    assert opt.stats.get("label_load_collapse", 0) == 0


# ── value_forward_to_reg ─────────────────────────────────────────


def test_value_forward_to_reg_label():
    """`mov eax, _label; mov ebx, eax` → `mov ebx, _label`
    when EAX is dead after."""
    asm = (
        "_f:\n"
        "        mov     eax, _glob\n"
        "        mov     ebx, eax\n"
        "        mov     eax, ecx\n"  # eax overwritten, dead
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     ebx, _glob" in out
    assert "mov     eax, _glob" not in out
    assert opt.stats.get("value_forward_to_reg") == 1


def test_value_forward_to_reg_label_arithmetic():
    """`mov eax, _b + 8; mov ebx, eax` → `mov ebx, _b + 8`."""
    asm = (
        "_f:\n"
        "        mov     eax, _b + 8\n"
        "        mov     ebx, eax\n"
        "        mov     eax, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     ebx, _b + 8" in out
    assert opt.stats.get("value_forward_to_reg") == 1


def test_value_forward_to_reg_immediate():
    """Numeric immediate also forwards."""
    asm = (
        "_f:\n"
        "        mov     eax, 42\n"
        "        mov     ebx, eax\n"
        "        mov     eax, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     ebx, 42" in out
    assert opt.stats.get("value_forward_to_reg") == 1


def test_value_forward_to_reg_skips_when_eax_live():
    """If EAX is read after the transfer, can't drop."""
    asm = (
        "_f:\n"
        "        mov     eax, _glob\n"
        "        mov     ebx, eax\n"
        "        push    eax\n"  # eax still needed
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("value_forward_to_reg", 0) == 0


def test_value_forward_to_reg_skips_memory_source():
    """Memory sources aren't handled by this pass — skip to avoid
    double-firing with other passes."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp - 4]\n"
        "        mov     ebx, eax\n"
        "        mov     eax, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("value_forward_to_reg", 0) == 0


def test_value_forward_to_reg_skips_self_mov():
    """`mov eax, X; mov eax, eax` — the second is a self-mov, not
    a transfer."""
    asm = (
        "_f:\n"
        "        mov     eax, 42\n"
        "        mov     eax, eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # value_forward shouldn't fire — self_mov_elimination drops
    # the second mov instead.
    assert opt.stats.get("value_forward_to_reg", 0) == 0
    # And the self-mov is gone.
    assert "mov     eax, eax" not in out


# ── byte_stores_to_dword ─────────────────────────────────────────


def test_byte_stores_to_dword_basic():
    """4 consecutive byte-imm stores at consecutive offsets pack
    into a single dword-imm store."""
    asm = (
        "_f:\n"
        "        mov     byte [ebp - 32], 104\n"
        "        mov     byte [ebp - 31], 101\n"
        "        mov     byte [ebp - 30], 108\n"
        "        mov     byte [ebp - 29], 108\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # 0x6c6c6568 = 'l'<<24 | 'l'<<16 | 'e'<<8 | 'h' = 1819043176
    assert "mov     dword [ebp - 32], 1819043176" in out
    assert "mov     byte [ebp - 32], 104" not in out
    assert opt.stats.get("byte_stores_to_dword") == 1


def test_byte_stores_to_dword_eight_consecutive():
    """8 consecutive byte stores → 2 dword stores (2 fires)."""
    asm = (
        "_f:\n"
        "        mov     byte [ebp - 8], 1\n"
        "        mov     byte [ebp - 7], 2\n"
        "        mov     byte [ebp - 6], 3\n"
        "        mov     byte [ebp - 5], 4\n"
        "        mov     byte [ebp - 4], 5\n"
        "        mov     byte [ebp - 3], 6\n"
        "        mov     byte [ebp - 2], 7\n"
        "        mov     byte [ebp - 1], 8\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("byte_stores_to_dword") == 2


def test_byte_stores_to_dword_skips_non_consecutive():
    """If offsets aren't consecutive, can't pack."""
    asm = (
        "_f:\n"
        "        mov     byte [ebp - 32], 104\n"
        "        mov     byte [ebp - 31], 101\n"
        "        mov     byte [ebp - 29], 108\n"  # gap
        "        mov     byte [ebp - 28], 108\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("byte_stores_to_dword", 0) == 0


def test_byte_stores_to_dword_skips_three_stores():
    """3 stores aren't enough to pack."""
    asm = (
        "_f:\n"
        "        mov     byte [ebp - 32], 104\n"
        "        mov     byte [ebp - 31], 101\n"
        "        mov     byte [ebp - 30], 108\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("byte_stores_to_dword", 0) == 0


def test_byte_stores_to_dword_positive_offset():
    """Positive offsets (params) also work."""
    asm = (
        "_f:\n"
        "        mov     byte [ebp + 8], 1\n"
        "        mov     byte [ebp + 9], 2\n"
        "        mov     byte [ebp + 10], 3\n"
        "        mov     byte [ebp + 11], 4\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # 0x04030201 = 67305985
    assert "mov     dword [ebp + 8], 67305985" in out
    assert opt.stats.get("byte_stores_to_dword") == 1


# ── pop_index_push_collapse ──────────────────────────────────────


def test_pop_index_push_collapse_basic():
    """`shl idx, 2; pop base; add idx, base; push dword [idx]` →
    `pop base; push dword [base + idx*4]`."""
    asm = (
        "_f:\n"
        "        push    eax\n"  # save base
        "        mov     eax, 3\n"  # load index
        "        shl     eax, 2\n"
        "        pop     ecx\n"
        "        add     eax, ecx\n"
        "        push    dword [eax]\n"
        "        mov     eax, ecx\n"  # eax dead before this
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "push    dword [ecx + eax*4]" in out
    assert "shl     eax, 2" not in out
    assert "add     eax, ecx" not in out
    assert opt.stats.get("pop_index_push_collapse") == 1


def test_pop_index_push_collapse_skips_when_idx_live():
    """If IDX is read after the push, can't drop."""
    asm = (
        "_f:\n"
        "        push    eax\n"
        "        mov     eax, 3\n"
        "        shl     eax, 2\n"
        "        pop     ecx\n"
        "        add     eax, ecx\n"
        "        push    dword [eax]\n"
        "        mov     [ebp - 4], eax\n"  # eax live (the address)
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("pop_index_push_collapse", 0) == 0


def test_pop_index_push_collapse_skips_invalid_scale():
    """Only scales 2/4/8."""
    asm = (
        "_f:\n"
        "        mov     eax, 3\n"
        "        shl     eax, 4\n"  # scale 16, not supported
        "        pop     ecx\n"
        "        add     eax, ecx\n"
        "        push    dword [eax]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("pop_index_push_collapse", 0) == 0


def test_pop_index_push_collapse_scale_8():
    """Scale 8 (long-long arrays)."""
    asm = (
        "_f:\n"
        "        mov     eax, 3\n"
        "        shl     eax, 3\n"  # scale 8
        "        pop     ecx\n"
        "        add     eax, ecx\n"
        "        push    dword [eax]\n"
        "        mov     eax, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "push    dword [ecx + eax*8]" in out
    assert opt.stats.get("pop_index_push_collapse") == 1


# ── pop_index_load_collapse ──────────────────────────────────────


def test_pop_index_load_collapse_basic():
    """`shl idx, 2; pop base; add idx, base; mov dst, [idx]` →
    `pop base; mov dst, [base + idx*4]`. The DST is also IDX (eax)
    in the typical loop body — load target overwrites, so IDX-dead
    check doesn't apply."""
    asm = (
        "_f:\n"
        "        push    eax\n"
        "        mov     eax, 3\n"
        "        shl     eax, 2\n"
        "        pop     ecx\n"
        "        add     eax, ecx\n"
        "        mov     eax, [eax]\n"  # DST = IDX = eax
        "        mov     eax, 0\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     eax, [ecx + eax*4]" in out
    assert "shl     eax, 2" not in out
    assert "add     eax, ecx" not in out
    assert opt.stats.get("pop_index_load_collapse") == 1


def test_pop_index_load_collapse_dst_different_from_idx():
    """DST != IDX — IDX-dead check applies and must succeed."""
    asm = (
        "_f:\n"
        "        push    eax\n"
        "        mov     eax, 3\n"
        "        shl     eax, 2\n"
        "        pop     ecx\n"
        "        add     eax, ecx\n"
        "        mov     edx, [eax]\n"  # DST = edx
        "        xor     eax, eax\n"  # eax dead witness
        "        mov     eax, edx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     edx, [ecx + eax*4]" in out
    assert opt.stats.get("pop_index_load_collapse") == 1


def test_pop_index_load_collapse_skips_invalid_scale():
    """Only scales 2/4/8."""
    asm = (
        "_f:\n"
        "        mov     eax, 3\n"
        "        shl     eax, 4\n"  # scale 16
        "        pop     ecx\n"
        "        add     eax, ecx\n"
        "        mov     eax, [eax]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("pop_index_load_collapse", 0) == 0


def test_pop_index_load_collapse_scale_8():
    """Scale 8 (long-long arrays)."""
    asm = (
        "_f:\n"
        "        mov     eax, 3\n"
        "        shl     eax, 3\n"  # scale 8
        "        pop     ecx\n"
        "        add     eax, ecx\n"
        "        mov     eax, [eax]\n"
        "        mov     eax, 0\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     eax, [ecx + eax*8]" in out
    assert opt.stats.get("pop_index_load_collapse") == 1


def test_pop_index_load_collapse_skips_when_idx_live_and_dst_diff():
    """DST != IDX, and IDX is read after — can't drop."""
    asm = (
        "_f:\n"
        "        push    eax\n"
        "        mov     eax, 3\n"
        "        shl     eax, 2\n"
        "        pop     ecx\n"
        "        add     eax, ecx\n"
        "        mov     edx, [eax]\n"
        "        mov     [ebp - 4], eax\n"  # eax live (the address)
        "        mov     eax, edx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("pop_index_load_collapse", 0) == 0


# ── push_pop_to_mov ──────────────────────────────────────────────


def test_push_pop_to_mov_label():
    """`push _label; chain; pop ecx` → `chain; mov ecx, _label`.
    Drops the push, replaces pop with mov.

    A subsequent imm_binop_collapse pass may further collapse
    `mov ecx, _label; add eax, ecx` to `add eax, _label`. Test
    asserts only the structural drop of push/pop."""
    asm = (
        "_f:\n"
        "        push    _g\n"
        "        mov     eax, [ebp - 4]\n"
        "        pop     ecx\n"
        "        mov     [_glob], ecx\n"  # consume ecx so cascade doesn't apply
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "push    _g" not in out
    assert "pop     ecx" not in out
    assert "mov     ecx, _g" in out
    assert opt.stats.get("push_pop_to_mov") == 1


def test_push_pop_to_mov_immediate():
    """Numeric immediate also folds."""
    asm = (
        "_f:\n"
        "        push    42\n"
        "        mov     eax, [ebp - 4]\n"
        "        pop     edx\n"
        "        mov     [_glob], edx\n"  # consume edx
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     edx, 42" in out
    assert opt.stats.get("push_pop_to_mov") == 1


def test_push_pop_to_mov_skips_register():
    """`push reg` (not imm/label) — don't rewrite (would lose
    the original reg's value)."""
    asm = (
        "_f:\n"
        "        push    eax\n"
        "        mov     eax, 5\n"
        "        pop     ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("push_pop_to_mov", 0) == 0


def test_push_pop_to_mov_skips_with_call():
    """Call inside the chain — fence (call could disturb stack)."""
    asm = (
        "_f:\n"
        "        push    _g\n"
        "        call    _helper\n"
        "        pop     ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("push_pop_to_mov", 0) == 0


def test_push_pop_to_mov_skips_with_esp_access():
    """Chain accesses [esp + N] — push offsets it; can't drop."""
    asm = (
        "_f:\n"
        "        push    100\n"
        "        mov     eax, [esp + 4]\n"  # reaches PAST our push
        "        pop     ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("push_pop_to_mov", 0) == 0


def test_push_pop_to_mov_skips_with_jump():
    """Chain has a jump — control flow may bypass the pop."""
    asm = (
        "_f:\n"
        "        push    _g\n"
        "        jmp     .L_target\n"
        "        pop     ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("push_pop_to_mov", 0) == 0


def test_push_pop_to_mov_balanced_inner_pushpop():
    """Chain has balanced inner push/pop — depth tracking matches
    the OUTER pop, not the inner one."""
    asm = (
        "_f:\n"
        "        push    _g\n"  # outer push
        "        push    eax\n"  # inner push
        "        mov     eax, 5\n"
        "        pop     edx\n"  # inner pop (matches inner push)
        "        pop     ecx\n"  # outer pop (matches outer push) — replaced
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # ecx should hold _g; the inner pair is preserved.
    assert "mov     ecx, _g" in out
    # The OUTER push must be dropped.
    assert out.count("push    _g") == 0
    assert opt.stats.get("push_pop_to_mov") == 1


# ── label_push_collapse ──────────────────────────────────────────


def test_label_push_collapse_basic():
    """`mov eax, _label; push dword [eax]` → `push dword [_label]`."""
    asm = (
        "_f:\n"
        "        mov     eax, _glob\n"
        "        push    dword [eax]\n"
        "        call    _consumer\n"
        "        add     esp, 4\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "push    dword [_glob]" in out
    assert opt.stats.get("label_push_collapse") == 1


def test_label_push_collapse_label_arithmetic():
    """`mov eax, _b + 4; push dword [eax]` → `push dword [_b + 4]`."""
    asm = (
        "_f:\n"
        "        mov     eax, _b + 4\n"
        "        push    dword [eax]\n"
        "        call    _consumer\n"
        "        add     esp, 4\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "push    dword [_b + 4]" in out
    assert opt.stats.get("label_push_collapse") == 1


def test_label_push_collapse_skips_when_eax_live():
    """Can't drop the address load if EAX is read after."""
    asm = (
        "_f:\n"
        "        mov     eax, _glob\n"
        "        push    dword [eax]\n"
        "        mov     [ebp - 4], eax\n"  # eax live
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("label_push_collapse", 0) == 0


def test_label_push_collapse_skips_numeric_imm():
    """Numeric source isn't a label."""
    asm = (
        "_f:\n"
        "        mov     eax, 42\n"
        "        push    dword [eax]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("label_push_collapse", 0) == 0


# ── label_store_collapse ─────────────────────────────────────────


def test_label_store_collapse_dword_imm():
    """`mov eax, _label; mov dword [eax], IMM` → `mov dword [_label], IMM`."""
    asm = (
        "_f:\n"
        "        mov     eax, _glob\n"
        "        mov     dword [eax], 42\n"
        "        xor     eax, eax\n"  # eax dead before this
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     dword [_glob], 42" in out
    assert opt.stats.get("label_store_collapse") == 1


def test_label_store_collapse_label_arithmetic():
    """`mov ecx, _label + N; mov dword [ecx], IMM`."""
    asm = (
        "_f:\n"
        "        mov     ecx, _b + 4\n"
        "        mov     dword [ecx], 100\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     dword [_b + 4], 100" in out
    assert opt.stats.get("label_store_collapse") == 1


def test_label_store_collapse_word_byte():
    """Also works for word and byte stores."""
    asm = (
        "_f:\n"
        "        mov     eax, _glob\n"
        "        mov     word [eax], 0x1234\n"
        "        mov     ecx, _other\n"
        "        mov     byte [ecx], 1\n"
        "        xor     eax, eax\n"  # eax dead before this
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     word [_glob], 0x1234" in out
    assert "mov     byte [_other], 1" in out
    assert opt.stats.get("label_store_collapse") == 2


def test_label_store_collapse_skips_when_reg_live():
    """If REG is read after the store, can't drop."""
    asm = (
        "_f:\n"
        "        mov     eax, _glob\n"
        "        mov     dword [eax], 42\n"
        "        push    eax\n"  # eax live
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("label_store_collapse", 0) == 0


def test_label_store_collapse_register_source():
    """Source can be a different register."""
    asm = (
        "_f:\n"
        "        mov     eax, _glob\n"
        "        mov     dword [eax], ecx\n"
        "        xor     eax, eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     dword [_glob], ecx" in out
    assert opt.stats.get("label_store_collapse") == 1


def test_label_store_collapse_skips_when_src_uses_reg():
    """If SRC references REG, can't fold."""
    asm = (
        "_f:\n"
        "        mov     eax, _glob\n"
        "        mov     dword [eax], eax\n"  # SRC uses EAX
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("label_store_collapse", 0) == 0


# ── lea_load_collapse ────────────────────────────────────────────


def test_lea_load_collapse_basic():
    """`lea eax, [ebp - 12]; mov eax, [eax]` →
    `mov eax, [ebp - 12]`."""
    asm = (
        "_f:\n"
        "        lea     eax, [ebp - 12]\n"
        "        mov     eax, [eax]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     eax, [ebp - 12]" in out
    assert "lea" not in out
    assert opt.stats.get("lea_load_collapse") == 1


def test_lea_load_collapse_with_offset():
    """`lea eax, [ebp - 12]; mov eax, [eax + 4]` →
    `mov eax, [ebp - 8]`."""
    asm = (
        "_f:\n"
        "        lea     eax, [ebp - 12]\n"
        "        mov     eax, [eax + 4]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     eax, [ebp - 8]" in out
    assert "lea" not in out
    assert opt.stats.get("lea_load_collapse") == 1


def test_lea_load_collapse_distinct_regs():
    """`lea ecx, [ebp - 12]; mov eax, [ecx + 4]` works when ecx
    is dead after."""
    asm = (
        "_f:\n"
        "        lea     ecx, [ebp - 12]\n"
        "        mov     eax, [ecx + 4]\n"
        "        xor     ecx, ecx\n"  # ecx overwritten
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     eax, [ebp - 8]" in out
    assert opt.stats.get("lea_load_collapse") == 1


def test_lea_load_collapse_skips_when_reg_live():
    """If REG is read after, can't drop the lea."""
    asm = (
        "_f:\n"
        "        lea     ecx, [ebp - 12]\n"
        "        mov     eax, [ecx]\n"
        "        mov     [ebp - 4], ecx\n"  # ecx still needed
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("lea_load_collapse", 0) == 0


def test_lea_load_collapse_negative_combined_offset():
    """`lea eax, [ebp + 4]; mov eax, [eax - 12]` →
    `mov eax, [ebp - 8]`. Combined offset: 4 + (-12) = -8."""
    asm = (
        "_f:\n"
        "        lea     eax, [ebp + 4]\n"
        "        mov     eax, [eax - 12]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     eax, [ebp - 8]" in out
    assert opt.stats.get("lea_load_collapse") == 1


def test_lea_load_collapse_size_keyword():
    """Size keyword on the load is preserved."""
    asm = (
        "_f:\n"
        "        lea     eax, [ebp - 12]\n"
        "        mov     eax, byte [eax + 1]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     eax, byte [ebp - 11]" in out
    assert opt.stats.get("lea_load_collapse") == 1


# ── lea_offset_fold ──────────────────────────────────────────────


def test_lea_offset_fold_add():
    """`lea reg, [ebp - N]; add reg, M` → `lea reg, [ebp - (N - M)]`."""
    asm = (
        "_f:\n"
        "        lea     eax, [ebp - 12]\n"
        "        add     eax, 4\n"
        "        mov     [ebp - 4], eax\n"  # flags dead by here
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "lea     eax, [ebp - 8]" in out
    assert "add     eax, 4" not in out
    assert opt.stats.get("lea_offset_fold") == 1


def test_lea_offset_fold_skips_when_flags_read():
    """If flags are read before being clobbered, can't drop add."""
    asm = (
        "_f:\n"
        "        lea     eax, [ebp - 12]\n"
        "        add     eax, 4\n"
        "        je      .L1\n"  # reads flags
        ".L1:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("lea_offset_fold", 0) == 0


# ── lea_forward_to_reg ───────────────────────────────────────────


def test_lea_forward_to_reg_basic():
    """`lea eax, [ebp - 12]; mov ecx, eax` → `lea ecx, [ebp - 12]`."""
    asm = (
        "_f:\n"
        "        lea     eax, [ebp - 12]\n"
        "        mov     ecx, eax\n"
        "        mov     eax, edx\n"  # eax dead before
        "        mov     [ecx], 0\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "lea     ecx, [ebp - 12]" in out
    assert opt.stats.get("lea_forward_to_reg") == 1


def test_lea_forward_to_reg_skips_when_eax_live():
    asm = (
        "_f:\n"
        "        lea     eax, [ebp - 12]\n"
        "        mov     ecx, eax\n"
        "        push    eax\n"  # eax live
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("lea_forward_to_reg", 0) == 0


# ── lea_store_collapse ───────────────────────────────────────────


def test_lea_store_collapse_basic():
    """`lea reg, [ebp - N]; mov dword [reg], V` → `mov dword [ebp - N], V`."""
    asm = (
        "_f:\n"
        "        lea     ecx, [ebp - 12]\n"
        "        mov     dword [ecx], 100\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     dword [ebp - 12], 100" in out
    assert "lea" not in out
    assert opt.stats.get("lea_store_collapse") == 1


def test_lea_store_collapse_with_offset():
    """`lea reg, [ebp - 12]; mov dword [reg + 4], V` → `mov dword [ebp - 8], V`."""
    asm = (
        "_f:\n"
        "        lea     ecx, [ebp - 12]\n"
        "        mov     dword [ecx + 4], 200\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     dword [ebp - 8], 200" in out
    assert opt.stats.get("lea_store_collapse") == 1


def test_lea_store_collapse_skips_when_src_uses_reg():
    """SRC referencing REG can't be folded."""
    asm = (
        "_f:\n"
        "        lea     ecx, [ebp - 12]\n"
        "        mov     dword [ecx], ecx\n"  # SRC uses ECX
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("lea_store_collapse", 0) == 0


# ── dead_stack_store ─────────────────────────────────────────────


def test_dead_stack_store_basic():
    """`mov [ebp - 12], 1; mov [ebp - 12], 100` drops the first."""
    asm = (
        "_f:\n"
        "        mov     dword [ebp - 12], 1\n"
        "        mov     dword [ebp - 12], 100\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     dword [ebp - 12], 1\n" not in out
    assert "mov     dword [ebp - 12], 100" in out
    assert opt.stats.get("dead_stack_store") == 1


def test_dead_stack_store_with_unrelated_stores():
    """Dead store survives across unrelated stores to other slots."""
    asm = (
        "_f:\n"
        "        mov     dword [ebp - 12], 1\n"
        "        mov     dword [ebp - 8], 2\n"
        "        mov     dword [ebp - 12], 100\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     dword [ebp - 12], 1\n" not in out
    assert "mov     dword [ebp - 8], 2" in out
    assert "mov     dword [ebp - 12], 100" in out
    assert opt.stats.get("dead_stack_store") == 1


def test_dead_stack_store_skips_when_read():
    """If [ebp - 12] is read between the two stores, don't drop."""
    asm = (
        "_f:\n"
        "        mov     dword [ebp - 12], 1\n"
        "        mov     eax, [ebp - 12]\n"  # reads it
        "        mov     dword [ebp - 12], 100\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("dead_stack_store", 0) == 0


def test_dead_stack_store_skips_across_call():
    """Calls clobber stack semantics — bail conservatively."""
    asm = (
        "_f:\n"
        "        mov     dword [ebp - 12], 1\n"
        "        call    _other\n"
        "        mov     dword [ebp - 12], 100\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("dead_stack_store", 0) == 0


def test_dead_stack_store_skips_across_jump():
    """Jumps mark control-flow boundaries — bail."""
    asm = (
        "_f:\n"
        "        mov     dword [ebp - 12], 1\n"
        "        jmp     .L1\n"
        ".L1:\n"
        "        mov     dword [ebp - 12], 100\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("dead_stack_store", 0) == 0


def test_dead_stack_store_skips_across_indirect_write():
    """Indirect memory write through register might alias — bail."""
    asm = (
        "_f:\n"
        "        mov     dword [ebp - 12], 1\n"
        "        mov     dword [eax], 99\n"  # indirect write, might alias
        "        mov     dword [ebp - 12], 100\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("dead_stack_store", 0) == 0


def test_dead_stack_store_skips_across_lea_of_same_offset():
    """LEA producing the same address means a register might be used
    for indirect read later — bail."""
    asm = (
        "_f:\n"
        "        mov     dword [ebp - 12], 1\n"
        "        lea     eax, [ebp - 12]\n"
        "        mov     dword [ebp - 12], 100\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("dead_stack_store", 0) == 0
