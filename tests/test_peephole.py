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
    # `.unused:` is now also dropped — it has 0 references and is
    # preceded by a terminator, so it's unreachable.
    assert ".unused:" not in out


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
        "        nop\n"  # extra non-droppable line so .far survives the cascade
    )
    out = optimize(asm)
    # `.far:` survives because it's referenced by the jmp. `.near:`
    # is unreferenced and preceded by jmp, so it's dropped (and its
    # following `ret` becomes dead, extending the dead zone). The
    # `jmp .far` itself doesn't get redirected — `.far:` is still
    # present after `.near:` is removed.
    assert ".far:" in out
    assert ".near:" not in out


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


def test_dead_after_call_abort():
    """`call _abort` is noreturn — code after is unreachable.
    abort/exit/longjmp are recognized as terminators."""
    asm = (
        "_f:\n"
        "        call    _abort\n"
        "        xor     eax, eax\n"
        "        mov     ebx, 1\n"
        ".epilogue:\n"
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # The xor and mov after call _abort are dropped.
    assert "xor     eax, eax" not in out
    assert "mov     ebx, 1" not in out
    # The unreferenced `.epilogue:` is also dropped (preceded by
    # noreturn terminator, no refs), and its trailing leave/ret
    # become dead and get dropped by the cascade.
    assert ".epilogue:" not in out
    assert "leave" not in out
    assert opt.stats.get("dead_after_terminator", 0) >= 2


def test_dead_after_call_exit():
    """`call _exit` is noreturn."""
    asm = (
        "_f:\n"
        "        push    0\n"
        "        call    _exit\n"
        "        add     esp, 4\n"
        "        xor     eax, eax\n"
        ".epilogue:\n"
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "add     esp, 4" not in out
    assert "xor     eax, eax" not in out
    # `.epilogue:` is unreferenced and preceded by a noreturn call —
    # dropped, and the cascade drops trailing leave/ret.
    assert ".epilogue:" not in out


def test_dead_after_call_builtin_unreachable():
    """`call ___builtin_unreachable` is noreturn."""
    asm = (
        "_f:\n"
        "        call    ___builtin_unreachable\n"
        "        mov     eax, 1\n"
        ".end:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     eax, 1" not in out
    assert opt.stats.get("dead_after_terminator", 0) >= 1


def test_dead_after_call_returning_fn():
    """A regular call (not in noreturn set) is NOT a terminator."""
    asm = (
        "_f:\n"
        "        call    _foo\n"
        "        mov     eax, 1\n"
        ".end:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # The mov must NOT be dropped — _foo might return.
    assert "mov     eax, 1" in out


def test_unreferenced_label_removal_basic():
    """Drop a label that's preceded by terminator and not referenced."""
    asm = (
        "_f:\n"
        "        jmp     .target\n"
        ".orphan:\n"
        "        mov     eax, 1\n"
        ".target:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # .orphan is preceded by `jmp .target` (terminator) and unreferenced.
    # After dropping it, dead_after_terminator extends past `mov eax, 1`.
    assert ".orphan:" not in out
    assert opt.stats.get("unreferenced_label_removal", 0) >= 1


def test_unreferenced_label_removal_skips_referenced():
    """Don't drop a label that has a reference."""
    asm = (
        "_f:\n"
        "        jmp     .target\n"
        ".target:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # .target is referenced by jmp — keep.
    # (Although jmp_to_next_label may merge it; that's a different pass.)
    assert opt.stats.get("unreferenced_label_removal", 0) == 0


def test_unreferenced_label_removal_skips_with_fallthrough():
    """Don't drop a label preceded by a non-terminator (fallthrough)."""
    asm = (
        "_f:\n"
        "        mov     eax, 1\n"
        ".label:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # .label has fallthrough from `mov eax, 1` — don't drop.
    assert ".label:" in out
    assert opt.stats.get("unreferenced_label_removal", 0) == 0


def test_unreferenced_label_removal_drops_dead_epilogue():
    """Drop unreferenced `.epilogue:` even when followed by `leave;
    ret` — when preceded by a noreturn call, the entire epilogue is
    unreachable. The cascade then drops leave/ret too."""
    asm = (
        "_f:\n"
        "        call    _abort\n"
        ".epilogue:\n"
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # `.epilogue:` is dropped, and dead_after_terminator extends past
    # to drop the trailing leave; ret.
    assert ".epilogue:" not in out
    assert "leave" not in out
    assert "ret" not in out


def test_unreferenced_label_scoped_per_function():
    """Local labels with the same name in different functions are
    scoped per-function. `.epilogue:` referenced in `_f` (via early
    return) but not in `_g` (which uses noreturn call) — only `_g`'s
    `.epilogue:` is dropped."""
    asm = (
        "_f:\n"
        "        test    eax, eax\n"
        "        jz      .epilogue\n"
        "        mov     eax, 1\n"
        ".epilogue:\n"
        "        leave\n"
        "        ret\n"
        "_g:\n"
        "        call    _abort\n"
        ".epilogue:\n"
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # `_f`'s `.epilogue:` is REFERENCED via `jz .epilogue` — preserve.
    # `_g`'s `.epilogue:` has 0 refs in _g's scope and is preceded by
    # noreturn — drop, and cascade drops trailing leave/ret.
    # Verify there's exactly one `.epilogue:` left (the one in _f).
    assert out.count(".epilogue:") == 1
    # The leave; ret in `_f` survives (still referenced via jz).
    assert "leave" in out


def test_unreferenced_label_removal_drops_chained_after_jmp():
    """Unreferenced label after `jmp .other` is dropped. The
    cascade then drops dead instructions following."""
    asm = (
        "_f:\n"
        "        jmp     .other\n"
        ".dead:\n"
        "        ret\n"
        ".other:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # `.dead:` is unreferenced and preceded by jmp — dropped.
    assert ".dead:" not in out
    # `.other:` is referenced — preserved.
    assert ".other:" in out


def test_unreferenced_label_removal_global_label_preserved():
    """Global labels (no leading `.`) are not considered for removal."""
    asm = (
        "_f:\n"
        "        jmp     .end\n"
        "_unreferenced:\n"
        "        ret\n"
        ".end:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # _unreferenced is global — preserve.
    assert "_unreferenced:" in out


def test_unreferenced_label_removal_switch_end():
    """The canonical case: a switch's `.L1_switch_end:` after a
    default-body that always returns. The label is unreferenced."""
    asm = (
        "_f:\n"
        "        jmp     .L_default\n"
        ".L_default:\n"
        "        mov     eax, -1\n"
        "        jmp     .epilogue\n"
        ".L1_switch_end:\n"
        "        xor     eax, eax\n"
        ".epilogue:\n"
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # .L1_switch_end is preceded by `jmp .epilogue` (terminator) and
    # unreferenced. Drop it.
    assert ".L1_switch_end:" not in out
    # The dead `xor eax, eax` between the dropped label and .epilogue
    # gets cleaned by dead_after_terminator.
    assert opt.stats.get("unreferenced_label_removal", 0) >= 1


# ── store_chain_retarget ─────────────────────────────────────────


def test_store_chain_retarget_two_instr_chain():
    """`push eax; mov eax, X; add eax, Y; pop ecx; mov [ecx], eax`
    (multi-instr chain) → `mov ecx, X; add ecx, Y; mov [eax], ecx`.

    The chain rewrites to ECX, the final store uses EAX (preserved
    address). Drops push + pop = 2 bytes.

    Use a multi-instr address computation (not just `mov eax, _g`,
    which push_immediate would consume first via the cascade
    push_immediate → push_pop_to_mov)."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp - 8]\n"  # eax = address (from local)
        "        shl     eax, 2\n"
        "        add     eax, _arr\n"
        "        push    eax\n"
        "        mov     eax, [ebp - 4]\n"
        "        add     eax, [ebp - 4]\n"
        "        pop     ecx\n"
        "        mov     [ecx], eax\n"
        "        xor     eax, eax\n"  # EAX dead after store
        "        xor     ecx, ecx\n"  # ECX dead after store
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # push/pop are gone.
    assert "push    eax" not in out
    assert "pop     ecx" not in out
    # Chain retargeted to ECX. The second instruction is then further
    # collapsed by `same_memory_operand_reuse` to `add ecx, ecx` since
    # ECX already holds [ebp - 4].
    assert "mov     ecx, [ebp - 4]" in out
    assert "add     ecx, ecx" in out
    # Store swapped to use EAX as address.
    assert "mov     [eax], ecx" in out
    assert opt.stats.get("store_chain_retarget", 0) == 1


def test_store_chain_retarget_skips_short_chain():
    """Single-instr chain (push eax; mov eax, src; pop ecx; mov
    [ecx], eax) is handled by the existing store_collapse pass.
    Both passes can fire; either is acceptable."""
    asm = (
        "_f:\n"
        "        mov     eax, _g\n"
        "        push    eax\n"
        "        mov     eax, 100\n"
        "        pop     ecx\n"
        "        mov     [ecx], eax\n"
        "        xor     eax, eax\n"
        "        xor     ecx, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # Either store_collapse or store_chain_retarget fires.
    fired = (
        opt.stats.get("store_collapse", 0)
        + opt.stats.get("store_chain_retarget", 0)
    )
    assert fired >= 1


def test_store_chain_retarget_skips_eax_live_after():
    """If EAX is live after the store, the rewrite would change EAX's
    post-state from chain result to address. Bail."""
    asm = (
        "_f:\n"
        "        mov     eax, _g\n"
        "        push    eax\n"
        "        mov     eax, [ebp - 4]\n"
        "        add     eax, [ebp - 4]\n"
        "        pop     ecx\n"
        "        mov     [ecx], eax\n"
        "        ret\n"  # EAX is the return value!
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    # Neither pattern may fire.
    assert opt.stats.get("store_chain_retarget", 0) == 0


def test_store_chain_retarget_skips_ecx_live_after():
    """If ECX is live after the store, the rewrite leaves ECX with
    the chain result instead of the address. Bail."""
    asm = (
        "_f:\n"
        "        mov     eax, _g\n"
        "        push    eax\n"
        "        mov     eax, [ebp - 4]\n"
        "        add     eax, [ebp - 4]\n"
        "        pop     ecx\n"
        "        mov     [ecx], eax\n"
        "        mov     eax, ecx\n"  # ECX is read after the store
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("store_chain_retarget", 0) == 0


def test_store_chain_retarget_skips_chain_reads_ecx():
    """If a chain instr reads ECX, the chain is not retargetable —
    after retarget the chain would read its own running value. Bail."""
    asm = (
        "_f:\n"
        "        mov     eax, _g\n"
        "        push    eax\n"
        "        mov     eax, [ebp - 4]\n"
        "        add     eax, ecx\n"  # reads ECX (would self-reference)
        "        pop     ecx\n"
        "        mov     [ecx], eax\n"
        "        xor     eax, eax\n"
        "        xor     ecx, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("store_chain_retarget", 0) == 0


def test_store_chain_retarget_skips_non_fresh_first():
    """Chain's first instr must be a fresh-write to EAX. If it reads
    EAX in src (RMW), the chain depends on EAX's prior value (the
    address) — bail."""
    asm = (
        "_f:\n"
        "        mov     eax, _g\n"
        "        push    eax\n"
        "        add     eax, 4\n"  # NOT a fresh write — reads EAX
        "        pop     ecx\n"
        "        mov     [ecx], eax\n"
        "        xor     eax, eax\n"
        "        xor     ecx, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("store_chain_retarget", 0) == 0


# ── shl_add_label_to_lea ─────────────────────────────────────────


def test_shl_add_label_to_lea_basic():
    """`shl eax, 2; add eax, _g` → `lea eax, [_g + eax*4]`.
    Saves 1 instruction, 1 byte."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp - 4]\n"
        "        shl     eax, 2\n"
        "        add     eax, _g\n"
        "        push    eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "lea     eax, [_g + eax*4]" in out
    assert "shl     eax, 2" not in out
    assert "add     eax, _g" not in out
    assert opt.stats.get("shl_add_label_to_lea") == 1


def test_shl_add_label_to_lea_scale_2():
    """N=1 produces SCALE=2."""
    asm = (
        "_f:\n"
        "        shl     ecx, 1\n"
        "        add     ecx, _arr\n"
        "        push    ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "lea     ecx, [_arr + ecx*2]" in out


def test_shl_add_label_to_lea_scale_8():
    """N=3 produces SCALE=8."""
    asm = (
        "_f:\n"
        "        shl     edx, 3\n"
        "        add     edx, _arr\n"
        "        push    edx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "lea     edx, [_arr + edx*8]" in out


def test_shl_add_label_to_lea_skips_non_label():
    """add with imm32 source (not a label) — skip; label_offset_fold
    handles different patterns. `shl + add IMM` doesn't fold via this
    pass."""
    asm = (
        "_f:\n"
        "        shl     eax, 2\n"
        "        add     eax, 100\n"
        "        push    eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("shl_add_label_to_lea", 0) == 0


def test_shl_add_label_to_lea_skips_diff_reg():
    """Different reg in shl vs add — skip."""
    asm = (
        "_f:\n"
        "        shl     eax, 2\n"
        "        add     ecx, _g\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("shl_add_label_to_lea", 0) == 0


def test_shl_add_label_to_lea_skips_when_flags_read():
    """If flags after the add are read (e.g., by jcc), don't rewrite —
    lea doesn't set flags."""
    asm = (
        "_f:\n"
        "        shl     eax, 2\n"
        "        add     eax, _g\n"
        "        jz      .L\n"  # reads ZF
        "        ret\n"
        ".L:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("shl_add_label_to_lea", 0) == 0


def test_store_chain_retarget_chain_with_eax_in_src_ok():
    """Chain instrs after the first may reference [eax] — these get
    retargeted to [ecx] in the rewrite, preserving semantics. Use
    a multi-instr address load to avoid push_immediate."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp - 8]\n"
        "        shl     eax, 2\n"
        "        add     eax, _arr\n"
        "        push    eax\n"
        "        mov     eax, [ebp - 4]\n"
        "        add     eax, [eax + 4]\n"  # reads [eax]; retargets to [ecx]
        "        pop     ecx\n"
        "        mov     [ecx], eax\n"
        "        xor     eax, eax\n"
        "        xor     ecx, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # Chain retargeted: EAX → ECX in dest AND in src [eax + 4] → [ecx + 4].
    assert "add     ecx, [ecx + 4]" in out
    assert opt.stats.get("store_chain_retarget", 0) == 1


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
    scaffold changes which byte the [esp + N] reads.

    Use jl (reads SF/OF) instead of bare ret so transfer_pop_cmp_collapse
    can't fire either; we want to confirm the ESP-relative source
    blocks all collapses on this pattern."""
    asm = (
        "_f:\n"
        "        push    eax\n"
        "        mov     eax, [esp + 4]\n"
        "        mov     ecx, eax\n"
        "        pop     eax\n"
        "        cmp     eax, ecx\n"
        "        jl      .L\n"
        "        ret\n"
        ".L:\n"
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
    """Store-through-pointer with literal value drops push/pop pair.

    Cascades with value_forward_to_reg: the `mov eax, [ebp - 4];
    mov ecx, eax` chain (where store_collapse inserted the transfer)
    folds further to `mov ecx, [ebp - 4]` since EAX is overwritten
    by the next `mov eax, 42`.
    """
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
    # After the cascade, the address loads directly into ECX.
    assert "mov     ecx, [ebp - 4]" in out
    # The value-load and store survive.
    assert "mov     eax, 42" in out
    assert "mov     [ecx], eax" in out
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
    """CFG-aware fallback fires across a label boundary when EAX is
    dead via every CFG path. The pre-loop init `int i = -5;` lowers
    as `mov eax, -5; mov [m], eax; .L_top: ...`. The label boundary
    blocks the immediate-witness fast path; the CFG-aware fallback
    sees that every path through the loop body and the loop exit
    overwrites EAX before reading it, so the rewrite is safe.

    Re-enabled after fixing `redundant_eax_load` to invalidate its
    cached reg_mem at any label that's the target of a backward
    jump (loop top). The earlier strcmp-1/strncmp-1 regression was
    rooted in that downstream pass — once the back-edge invalidation
    landed, this CFG fallback became safe to re-enable.
    """
    asm = (
        "_f:\n"
        "        mov     eax, -5\n"
        "        mov     [ebp - 8], eax\n"
        ".L_top:\n"
        "        cmp     dword [ebp - 8], 200\n"
        "        jge     .L_end\n"
        "        mov     eax, [ebp - 8]\n"
        "        jmp     .L_top\n"
        ".L_end:\n"
        "        mov     eax, 0\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("imm_store_collapse", 0) == 1
    assert "mov     dword [ebp - 8], -5" in out
    # The two pre-rewrite lines should be gone.
    assert "mov     eax, -5\n" not in out
    assert "        mov     [ebp - 8], eax\n" not in out


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


def test_mov_zero_to_xor_recognizes_hex_zero():
    """`mov edx, 0x00000000` is the same as `mov edx, 0`. The codegen's
    long-long path emits the hex form for zero high-halves; we should
    recognize and rewrite it the same way."""
    asm = (
        "_f:\n"
        "        mov     edx, 0x00000000\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "xor     edx, edx" in out
    assert opt.stats.get("mov_zero_to_xor", 0) == 1


def test_mov_zero_to_xor_recognizes_short_hex_zero():
    """`mov eax, 0x0` should also recognize."""
    asm = (
        "_f:\n"
        "        mov     eax, 0x0\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "xor     eax, eax" in out
    assert opt.stats.get("mov_zero_to_xor", 0) == 1


def test_mov_zero_to_xor_recognizes_uppercase_hex_zero():
    """`mov ebx, 0X00` (uppercase X) should also recognize."""
    asm = (
        "_f:\n"
        "        mov     ebx, 0X00\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "xor     ebx, ebx" in out
    assert opt.stats.get("mov_zero_to_xor", 0) == 1


def test_mov_zero_to_xor_skips_nonzero_hex_immediate():
    """`mov eax, 0x00000001` is NOT zero — must NOT rewrite."""
    asm = (
        "_f:\n"
        "        mov     eax, 0x00000001\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     eax, 0x00000001" in out
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
    """`mov eax, [eax]` reads EAX as an address — the memory deref
    could have observable effects (page faults, MMIO). Skip even
    when EAX is dead after."""
    asm = (
        "_f:\n"
        "        mov     eax, [eax]\n"
        "        mov     eax, 5\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("dead_mov_to_reg", 0) == 0


def test_dead_mov_drops_movsx_self_register():
    """`movsx eax, al` reads AL (sub-reg of EAX) and writes EAX —
    no memory access, no observable side effect. Safe to drop when
    EAX is dead. Common in postfix C99 assignment-expression value
    materialization (`dst[i] = src[i];` lowers to `mov [ecx], al;
    movsx eax, al` where the second produces the assignment-
    expression's int-promoted value, but it's unused).
    """
    asm = (
        "_f:\n"
        "        movsx   eax, byte [esi]\n"
        "        mov     byte [ecx], al\n"
        "        movsx   eax, al\n"  # redundant — value unused
        "        mov     eax, 5\n"  # overwrite — confirms EAX dead
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # The redundant movsx is gone.
    assert out.count("movsx") == 1


def test_dead_mov_drops_movzx_self_register():
    """Same as above but movzx — also safe to drop."""
    asm = (
        "_f:\n"
        "        movzx   eax, byte [esi]\n"
        "        mov     byte [ecx], al\n"
        "        movzx   eax, al\n"
        "        mov     eax, 5\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert out.count("movzx") == 1


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
    are contiguous), and the ECX save is preserved untouched.

    The trailing `mov eax, [ebp - 12]` simulates a nested-fn goto
    site that reads the static link, keeping the slot live so
    `dead_unused_slot_stores` doesn't fire."""
    asm = (
        "_inner:\n"
        "        push    ebp\n"
        "        mov     ebp, esp\n"
        "        sub     esp, 12\n"
        "        mov     [ebp - 12], ecx\n"
        "        mov     eax, [ebp - 12]\n"
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


def test_redundant_eax_load_after_store_to_unknown():
    """When reg_mem was None (no prior load tracked), a `mov [m], REG`
    establishes that REG equals [m] for any subsequent `mov REG, m`."""
    asm = (
        "_f:\n"
        ".L_target:\n"
        # No prior load — eax's value is unknown coming in (e.g.,
        # from a merge of multiple branches).
        "        mov     [ebp - 4], eax\n"
        "        cmp     eax, ecx\n"  # read-only on eax, preserves
        "        jle     .L_else\n"
        "        mov     eax, [ebp - 4]\n"  # redundant after store
        "        ret\n"
        ".L_else:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # The redundant load should be detected and dropped.
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


def test_cmp_load_collapse_chain_strchr_pattern():
    """The strchr-style pattern: ``mov eax, c; chain that builds *s
    in ECX; cmp eax, ecx; jne``. EAX is loaded once, used at the end.
    Chain of 2 instructions doesn't read or write EAX. Both branches
    overwrite EAX before ret, so EAX is dead after the cmp."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 12]\n"
        "        mov     ecx, [ebp + 8]\n"
        "        movsx   ecx, byte [ecx]\n"
        "        cmp     eax, ecx\n"
        "        jne     .L4_endif\n"
        "        mov     eax, 1\n"
        "        ret\n"
        ".L4_endif:\n"
        "        xor     eax, eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     eax, [ebp + 12]" not in out
    assert "cmp     dword [ebp + 12], ecx" in out
    assert opt.stats.get("cmp_load_collapse") == 1


def test_cmp_load_collapse_chain_skips_eax_in_chain():
    """If the chain references EAX (read or write), bail."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 12]\n"
        "        mov     ecx, eax\n"  # reads EAX!
        "        cmp     eax, ecx\n"
        "        jne     .L\n"
        "        ret\n"
        ".L:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("cmp_load_collapse", 0) == 0


def test_cmp_load_collapse_skips_chain_writes_to_sib_idx_reg():
    """Regression: the source mem operand `[eax + ecx*4]` references
    ECX in its addressing. A chain `pop ecx` writes to ECX, which the
    collapsed cmp would re-read. The collapsed `cmp [eax + ecx*4], X`
    would read from a different address. Must bail.

    Reproduces a real bug from gcc-c-torture 20000422-1 where the
    pattern fired and produced incorrect address arithmetic."""
    asm = (
        "_f:\n"
        "        mov     eax, [eax + ecx*4]\n"
        "        pop     ecx\n"
        "        cmp     eax, ecx\n"
        "        je      .L\n"
        "        ret\n"
        ".L:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("cmp_load_collapse", 0) == 0


def test_cmp_load_collapse_chain_skips_call():
    """A call clobbers EAX per cdecl — bail."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 12]\n"
        "        call    _foo\n"
        "        cmp     eax, 0\n"
        "        jne     .L\n"
        "        ret\n"
        ".L:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("cmp_load_collapse", 0) == 0


def test_cmp_load_collapse_chain_skips_mem_alias():
    """If the chain stores to memory that may alias [m], bail."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 12]\n"
        "        mov     [ecx], 99\n"  # may alias [ebp+12]
        "        cmp     eax, 0\n"
        "        jne     .L\n"
        "        ret\n"
        ".L:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("cmp_load_collapse", 0) == 0


def test_cmp_load_collapse_chain_allows_disjoint_store():
    """A store to a definitively-disjoint ebp-relative slot is OK."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 12]\n"
        "        mov     dword [ebp - 4], 99\n"  # disjoint!
        "        cmp     eax, 0\n"
        "        jne     .L\n"
        "        mov     eax, 1\n"
        "        ret\n"
        ".L:\n"
        "        xor     eax, eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "cmp     dword [ebp + 12], 0" in out
    assert opt.stats.get("cmp_load_collapse") == 1


def test_cmp_load_collapse_chain_skips_label():
    """A label inside the chain is a basic-block boundary — bail."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 12]\n"
        ".L_mid:\n"
        "        cmp     eax, 0\n"
        "        jne     .L\n"
        "        ret\n"
        ".L:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("cmp_load_collapse", 0) == 0


def test_cmp_load_collapse_chain_skips_inc_aliasing_mem():
    """`inc dword [m]` is a 1-operand RMW that writes [m]. Bail on
    the alias check. Regression: was missed because `_operands_split`
    returns None for 1-operand instructions; the alias check was
    skipped entirely. Caused 20010915-1 / c-testsuite 00207 etc.
    where the codegen emits `mov eax, [_check]; inc dword [_check];
    cmp eax, 1` for `check++ > 1` (post-increment compare on OLD
    value).
    """
    asm = (
        "_s:\n"
        "        mov     eax, [_check]\n"
        "        inc     dword [_check]\n"
        "        cmp     eax, 1\n"
        "        jg      .L_or\n"
        "        mov     eax, 0\n"
        "        ret\n"
        ".L_or:\n"
        "        mov     eax, 1\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    # Must NOT collapse — inc aliases [_check] and changes its value
    # before the cmp would read it.
    assert opt.stats.get("cmp_load_collapse", 0) == 0


def test_cmp_load_collapse_chain_skips_dec_aliasing_mem():
    """Same as inc — dec is the symmetric write."""
    asm = (
        "_s:\n"
        "        mov     eax, [_check]\n"
        "        dec     dword [_check]\n"
        "        cmp     eax, 1\n"
        "        jg      .L_or\n"
        "        mov     eax, 0\n"
        "        ret\n"
        ".L_or:\n"
        "        mov     eax, 1\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("cmp_load_collapse", 0) == 0


def test_cmp_load_collapse_chain_inc_disjoint_mem_ok():
    """An `inc dword [m1]` where m1 differs from [m] in `mov eax, [m]`
    is OK — disjoint memory."""
    asm = (
        "_f:\n"
        "        mov     eax, [_check]\n"
        "        inc     dword [_other]\n"
        "        cmp     eax, 1\n"
        "        jg      .L_or\n"
        "        mov     eax, 0\n"
        "        ret\n"
        ".L_or:\n"
        "        mov     eax, 1\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    # Both labels — `_check` and `_other` — are non-ebp-relative,
    # so `_mem_disjoint` returns False (= "may alias") for safety.
    # Pass conservatively bails. Test documents this.
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


# ── rmw_intermediate_collapse ─────────────────────────────────────


def test_rmw_intermediate_collapse_basic():
    """`mov eax, [m]; <intermediate>; add eax, REG; mov [m], eax` →
    `<intermediate>; add [m], REG`. The intermediate sets up REG
    via memory loads but doesn't touch EAX or [m]."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp - 4]\n"
        "        mov     ecx, [ebp + 8]\n"  # intermediate (sets ecx)
        "        mov     ecx, [ecx + 4]\n"  # intermediate (deref ecx)
        "        add     eax, ecx\n"
        "        mov     [ebp - 4], eax\n"
        "        xor     eax, eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # The original load+add+store is gone:
    assert "mov     eax, [ebp - 4]" not in out
    assert "add     eax, ecx" not in out
    # Replaced with memory-RMW:
    assert "add     dword [ebp - 4], ecx" in out
    assert opt.stats.get("rmw_intermediate_collapse") == 1


def test_rmw_intermediate_collapse_skips_when_intermediate_writes_eax():
    """If an intermediate writes EAX, can't drop the original load."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp - 4]\n"
        "        mov     eax, [ebp + 8]\n"  # writes EAX!
        "        add     eax, ecx\n"
        "        mov     [ebp - 4], eax\n"
        "        xor     eax, eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("rmw_intermediate_collapse", 0) == 0


def test_rmw_intermediate_collapse_skips_when_intermediate_reads_flags():
    """If an intermediate reads flags, can't fold."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp - 4]\n"
        "        je      .L_skip\n"  # reads flags
        "        add     eax, ecx\n"
        "        mov     [ebp - 4], eax\n"
        ".L_skip:\n"
        "        xor     eax, eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("rmw_intermediate_collapse", 0) == 0


def test_rmw_intermediate_collapse_skips_when_intermediate_modifies_mem():
    """If intermediate modifies the memory location, can't fold."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp - 4]\n"
        "        mov     [ebp - 4], 99\n"  # modifies [ebp - 4]
        "        add     eax, ecx\n"
        "        mov     [ebp - 4], eax\n"
        "        xor     eax, eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("rmw_intermediate_collapse", 0) == 0


def test_rmw_intermediate_collapse_all_ops():
    """sub/and/or/xor all work. Use the linked-list-shape intermediate
    (deref through ecx) which prevents imm_binop_collapse from folding
    into a single memory operand."""
    for op in ("sub", "and", "or", "xor"):
        asm = (
            "_f:\n"
            "        mov     eax, [ebp - 4]\n"
            "        mov     ecx, [ebp + 8]\n"
            "        mov     ecx, [ecx + 4]\n"  # deref blocks
                                                # imm_binop_collapse
            f"        {op}     eax, ecx\n"
            "        mov     [ebp - 4], eax\n"
            "        xor     eax, eax\n"
            "        ret\n"
        )
        opt = PeepholeOptimizer()
        out = opt.optimize(asm)
        # NASM emits varying whitespace based on op length; just
        # check op + memory-RMW form exists.
        assert f"{op}" in out and "dword [ebp - 4], ecx" in out, (
            f"{op}: {out}"
        )
        assert opt.stats.get("rmw_intermediate_collapse") == 1


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
    """movsx with sized memory operand and displacement collapses
    via SIB form, preserving op and size prefix."""
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
    assert "movsx   eax, byte [eax + ecx*4 + 4]" in out
    assert opt.stats.get("index_load_collapse") == 1


def test_index_load_collapse_movzx_word():
    """movzx for word loads also folds via SIB."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     ecx, [ebp - 8]\n"
        "        shl     ecx, 1\n"
        "        add     eax, ecx\n"
        "        movzx   eax, word [eax]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "movzx   eax, word [eax + ecx*2]" in out
    assert opt.stats.get("index_load_collapse") == 1


def test_sib_const_index_fold_movsx():
    """sib_const_index_fold preserves movsx (and movzx) op when
    folding the const index."""
    asm = (
        "_f:\n"
        "        mov     ecx, 3\n"
        "        movsx   eax, word [eax + ecx*2]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "movsx   eax, word [eax + 6]" in out
    assert opt.stats.get("sib_const_index_fold") == 1


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


def test_add_esp_to_pop_4():
    """`add esp, 4` after a call collapses to `pop ecx` (saves 2
    bytes — 3-byte add → 1-byte pop). ECX is caller-saved so dead
    after the call by convention."""
    asm = (
        "_f:\n"
        "        push    5\n"
        "        call    _foo\n"
        "        add     esp, 4\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "add     esp, 4" not in out
    assert "pop     ecx" in out
    assert opt.stats.get("add_esp_to_pop") == 1


def test_add_esp_to_pop_8():
    """`add esp, 8` collapses to two consecutive `pop ecx` (saves 1
    byte — 3-byte add → 2 × 1-byte pops)."""
    asm = (
        "_f:\n"
        "        push    5\n"
        "        push    6\n"
        "        call    _foo\n"
        "        add     esp, 8\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "add     esp, 8" not in out
    assert out.count("pop     ecx") == 2
    assert opt.stats.get("add_esp_to_pop") == 1


def test_add_esp_to_pop_skips_when_ecx_live():
    """If ECX is read after the add esp, the rewrite would clobber its
    value. Bail."""
    asm = (
        "_f:\n"
        "        push    5\n"
        "        call    _foo\n"
        "        add     esp, 4\n"
        "        mov     edx, ecx\n"  # ECX read here
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "add     esp, 4" in out
    assert opt.stats.get("add_esp_to_pop", 0) == 0


def test_add_esp_to_pop_skips_large_imm():
    """`add esp, 16` is shorter than 4 pops; don't rewrite."""
    asm = (
        "_f:\n"
        "        call    _foo\n"
        "        add     esp, 16\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "add     esp, 16" in out
    assert opt.stats.get("add_esp_to_pop", 0) == 0


def test_add_esp_to_pop_when_next_is_pop_ecx():
    """`add esp, 4; pop ecx` collapses even without a preceding
    call. The next `pop ecx` overwrites ECX, so the value the new
    pop loads is irrelevant."""
    asm = (
        "_f:\n"
        "        fild    dword [esp]\n"
        "        add     esp, 4\n"
        "        pop     ecx\n"
        "        fstp    dword [ecx]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    out_lines = out.split("\n")
    has_add = any(
        line.strip() == "add     esp, 4" for line in out_lines
    )
    assert not has_add
    # Both pop ecx instructions should be present (one from
    # the conversion + the original).
    assert out.count("pop     ecx") == 2
    assert opt.stats.get("add_esp_to_pop") == 1


def test_narrow_store_reload_collapse_word():
    """`mov word [m], ax; movsx eax, word [m]` collapses to
    `mov word [m], ax; movsx eax, ax`. Saves 1 byte."""
    asm = (
        "_f:\n"
        "        mov     word [ebp - 4], ax\n"
        "        movsx   eax, word [ebp - 4]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     word [ebp - 4], ax" in out
    assert "movsx   eax, ax" in out
    assert opt.stats.get("narrow_store_reload_collapse") == 1


def test_narrow_store_reload_collapse_byte():
    """Byte form: `mov byte [m], al; movzx eax, byte [m]` →
    `mov byte [m], al; movzx eax, al`."""
    asm = (
        "_f:\n"
        "        mov     byte [ebp - 1], al\n"
        "        movzx   eax, byte [ebp - 1]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     byte [ebp - 1], al" in out
    assert "movzx   eax, al" in out
    assert opt.stats.get("narrow_store_reload_collapse") == 1


def test_narrow_store_reload_collapse_movzx():
    """movzx (zero-extend) version also collapses."""
    asm = (
        "_f:\n"
        "        mov     word [ebp - 4], cx\n"
        "        movzx   ebx, word [ebp - 4]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "movzx   ebx, cx" in out
    assert opt.stats.get("narrow_store_reload_collapse") == 1


def test_narrow_store_reload_collapse_skips_size_mismatch():
    """If the load and store sizes don't match, don't collapse."""
    asm = (
        "_f:\n"
        "        mov     byte [ebp - 4], al\n"
        "        movsx   eax, word [ebp - 4]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("narrow_store_reload_collapse", 0) == 0


def test_narrow_store_reload_collapse_skips_different_addr():
    """Different memory addresses don't collapse."""
    asm = (
        "_f:\n"
        "        mov     word [ebp - 4], ax\n"
        "        movsx   eax, word [ebp - 8]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("narrow_store_reload_collapse", 0) == 0


def test_dual_zero_init_consolidate_basic():
    """Pattern `xor eax, eax; xor edx, edx; mov [m1], eax; mov [m2], edx`
    consolidates to `xor eax, eax; mov [m1], eax; mov [m2], eax`. The
    `xor edx, edx` is dropped because EDX is dead after the chain
    (overwritten by `mov edx, X` later)."""
    asm = (
        "_f:\n"
        "        xor     eax, eax\n"
        "        xor     edx, edx\n"
        "        mov     [ebp - 8], eax\n"
        "        mov     [ebp - 4], edx\n"
        "        mov     edx, 1\n"  # overwrites edx (dead before)
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "xor     edx, edx" not in out
    assert "mov     [ebp - 4], eax" in out
    assert opt.stats.get("dual_zero_init_consolidate") == 1


def test_dual_zero_init_consolidate_chain_of_stores():
    """Multiple LL inits in a row: 6 stores from 2 xor regs collapse
    to 6 stores from 1 xor reg."""
    asm = (
        "_f:\n"
        "        xor     eax, eax\n"
        "        xor     edx, edx\n"
        "        mov     [ebp - 8], eax\n"
        "        mov     [ebp - 4], edx\n"
        "        mov     [ebp - 16], eax\n"
        "        mov     [ebp - 12], edx\n"
        "        mov     [ebp - 24], eax\n"
        "        mov     [ebp - 20], edx\n"
        "        mov     edx, 1\n"  # witness
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "xor     edx, edx" not in out
    # All 3 EDX stores rewritten to use EAX.
    assert out.count("eax\n") >= 6


def test_dual_zero_init_consolidate_skips_when_edx_live_at_ret():
    """If EDX is live at ret (might be LL return value), don't drop the
    xor."""
    asm = (
        "_f:\n"
        "        xor     eax, eax\n"
        "        xor     edx, edx\n"
        "        mov     [ebp - 8], eax\n"
        "        mov     [ebp - 4], edx\n"
        "        ret\n"  # ret reads EDX (LL return value possible)
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "xor     edx, edx" in out
    assert opt.stats.get("dual_zero_init_consolidate", 0) == 0


def test_dual_zero_init_consolidate_skips_when_no_edx_store():
    """If the chain has only EAX stores, no consolidation needed."""
    asm = (
        "_f:\n"
        "        xor     eax, eax\n"
        "        xor     edx, edx\n"
        "        mov     [ebp - 8], eax\n"
        "        mov     edx, 1\n"  # not a store of edx
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("dual_zero_init_consolidate", 0) == 0


def test_dual_zero_init_consolidate_skips_when_chain_breaks():
    """If a non-store instruction appears between the xors and the
    desired stores, the chain breaks. Bail."""
    asm = (
        "_f:\n"
        "        xor     eax, eax\n"
        "        xor     edx, edx\n"
        "        add     eax, 5\n"  # breaks the chain
        "        mov     [ebp - 8], eax\n"
        "        mov     [ebp - 4], edx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("dual_zero_init_consolidate", 0) == 0


def test_zero_init_collapse_recognizes_hex_zero():
    """`mov dword [m], 0x00000000` is the same as `mov dword [m], 0` —
    the codegen sometimes emits the hex form. Both should fire
    zero_init_collapse the same way."""
    asm = (
        "_f:\n"
        "        mov     dword [ebp - 8], 0x00000000\n"
        "        mov     dword [ebp - 4], 0x00000000\n"
        "        mov     eax, 1\n"  # witness
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "xor     eax, eax" in out
    assert "mov     [ebp - 8], eax" in out
    assert "mov     [ebp - 4], eax" in out
    assert opt.stats.get("zero_init_collapse") == 2


def test_zero_init_collapse_skips_nonzero_hex():
    """`mov dword [m], 0x00000001` is NOT zero — must NOT rewrite."""
    asm = (
        "_f:\n"
        "        mov     dword [ebp - 4], 0x00000001\n"
        "        mov     eax, 1\n"  # witness
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     dword [ebp - 4], 0x00000001" in out
    assert opt.stats.get("zero_init_collapse", 0) == 0


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

    With cmp_load_collapse extended to walk chains, the first
    ``mov eax, [m]`` is consumed into the test (becomes ``cmp [m],
    0``) before redundant_eax_load runs. The end state is the same:
    the redundant second load is dropped (count == 1 in output).
    We verify the deduplication regardless of which pass got credit.
    """
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        add     ebx, 1\n"
        "        test    eax, eax\n"
        "        mov     eax, [ebp + 8]\n"  # redundant
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # At most one occurrence of `mov eax, [ebp + 8]` survives.
    assert out.count("mov     eax, [ebp + 8]") <= 1
    # Some dedup pass fired.
    assert (
        opt.stats.get("redundant_eax_load", 0)
        + opt.stats.get("cmp_load_collapse", 0)
    ) >= 1


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


# ── disp_store_collapse ───────────────────────────────────────────


def test_disp_store_collapse_basic():
    """`add reg, N; mov [reg], src` → `mov [reg + N], src` when reg is
    dead after the store. Saves 2 bytes (drops the add)."""
    asm = (
        "_f:\n"
        "        mov     ecx, [ebp + 8]\n"
        "        add     ecx, 4\n"
        "        mov     [ecx], eax\n"
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "add     ecx" not in out
    assert "mov     [ecx + 4], eax" in out
    assert opt.stats.get("disp_store_collapse") == 1


def test_disp_store_collapse_with_size_prefix():
    """`add reg, N; mov dword [reg], imm` → `mov dword [reg + N], imm`."""
    asm = (
        "_f:\n"
        "        mov     ecx, [ebp + 8]\n"
        "        add     ecx, 4\n"
        "        mov     dword [ecx], 42\n"
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "add     ecx" not in out
    assert "mov     dword [ecx + 4], 42" in out
    assert opt.stats.get("disp_store_collapse") == 1


def test_disp_store_collapse_negative_disp():
    """Negative DISP collapses with `-` sign."""
    asm = (
        "_f:\n"
        "        mov     ecx, [ebp + 8]\n"
        "        add     ecx, -8\n"
        "        mov     [ecx], eax\n"
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     [ecx - 8], eax" in out
    assert opt.stats.get("disp_store_collapse") == 1


def test_disp_store_collapse_skips_when_reg_live():
    """If REG is read after the store, can't drop the add."""
    asm = (
        "_f:\n"
        "        mov     ecx, [ebp + 8]\n"
        "        add     ecx, 4\n"
        "        mov     [ecx], eax\n"
        "        mov     edx, ecx\n"  # ecx live here
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("disp_store_collapse", 0) == 0


def test_disp_store_collapse_skips_when_src_uses_base():
    """If SRC references REG, the rewrite would change which value
    is read at the store time."""
    asm = (
        "_f:\n"
        "        mov     eax, 7\n"
        "        add     eax, 4\n"
        "        mov     [eax], eax\n"  # SRC == address reg
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("disp_store_collapse", 0) == 0


def test_disp_store_collapse_rejects_non_literal_disp():
    """Non-numeric add operand (e.g., label) doesn't collapse."""
    asm = (
        "_f:\n"
        "        mov     ecx, [ebp + 8]\n"
        "        add     ecx, _some_label\n"
        "        mov     [ecx], eax\n"
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("disp_store_collapse", 0) == 0


def test_disp_store_collapse_rejects_existing_offset():
    """If the store already uses `[reg + N]`, no further fold."""
    asm = (
        "_f:\n"
        "        mov     ecx, [ebp + 8]\n"
        "        add     ecx, 4\n"
        "        mov     [ecx + 4], eax\n"
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("disp_store_collapse", 0) == 0


def test_disp_store_collapse_intermediate_load():
    """Intermediate non-touching load is allowed between add and store.

    Common shape: codegen emits `add reg, offset` then `mov src_reg,
    [val_slot]` then `mov [reg], src_reg`. The intermediate load
    doesn't reference REG, so disp_store_collapse can still fire
    across it."""
    asm = (
        "_f:\n"
        "        mov     ecx, [ebp + 8]\n"
        "        add     ecx, 4\n"
        "        mov     eax, [ebp + 16]\n"  # doesn't touch ecx
        "        mov     [ecx], eax\n"
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "add     ecx" not in out
    assert "mov     [ecx + 4], eax" in out
    assert opt.stats.get("disp_store_collapse") == 1


def test_disp_store_collapse_skips_when_intermediate_reads_reg():
    """If an intermediate instruction reads REG, can't fold."""
    asm = (
        "_f:\n"
        "        mov     ecx, [ebp + 8]\n"
        "        add     ecx, 4\n"
        "        mov     edx, ecx\n"  # reads ecx
        "        mov     [ecx], eax\n"
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("disp_store_collapse", 0) == 0


def test_disp_store_collapse_skips_when_intermediate_reads_flags():
    """If an intermediate instruction reads flags, can't drop the add
    (the add's flags would be observed)."""
    asm = (
        "_f:\n"
        "        mov     ecx, [ebp + 8]\n"
        "        add     ecx, 4\n"
        "        je      .L_skip\n"  # reads flags
        "        mov     [ecx], eax\n"
        ".L_skip:\n"
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("disp_store_collapse", 0) == 0


# ── dead_store_before_push ─────────────────────────────────────────


def test_dead_store_before_push_basic():
    """`mov [m], REG; push REG; ...; ret` where [m] is never read.
    The store is dead — only the push consumes the value."""
    asm = (
        "_f:\n"
        "        mov     eax, 42\n"
        "        mov     [ebp - 12], eax\n"  # dead store
        "        push    eax\n"
        "        push    _str\n"
        "        call    _printf\n"
        "        add     esp, 8\n"
        "        xor     eax, eax\n"
        ".epilogue:\n"
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    out_lines = out.split("\n")
    has_store = any(
        "[ebp - 12]" in line for line in out_lines
    )
    assert not has_store
    assert opt.stats.get("dead_store_before_push") == 1


def test_dead_store_before_push_skips_when_slot_read_after():
    """If [m] is read later in the function, the store is alive."""
    asm = (
        "_f:\n"
        "        mov     eax, 42\n"
        "        mov     [ebp - 12], eax\n"
        "        push    eax\n"
        "        push    dword [ebp - 12]\n"  # reads [ebp - 12]
        "        call    _foo\n"
        "        add     esp, 8\n"
        ".epilogue:\n"
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     [ebp - 12], eax" in out
    assert opt.stats.get("dead_store_before_push", 0) == 0


def test_dead_store_before_push_skips_when_next_is_push_mem():
    """`mov [m], R; push dword [m]` — the push reads [m], so the
    store is alive."""
    asm = (
        "_f:\n"
        "        mov     edx, 42\n"
        "        mov     [ebp - 4], edx\n"
        "        push    dword [ebp - 4]\n"  # reads [ebp - 4]
        "        call    _foo\n"
        "        pop     ecx\n"
        ".epilogue:\n"
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # Store should remain.
    assert "mov     [ebp - 4], edx" in out
    assert opt.stats.get("dead_store_before_push", 0) == 0


def test_dead_store_before_push_skips_when_lea_taken():
    """If any function uses `lea` on an ebp-offset slot, the address
    might escape and we can't safely eliminate stores."""
    asm = (
        "_f:\n"
        "        lea     eax, [ebp - 8]\n"  # address-take
        "        mov     [ebp - 12], eax\n"
        "        push    eax\n"
        "        call    _foo\n"
        "        pop     ecx\n"
        ".epilogue:\n"
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("dead_store_before_push", 0) == 0


# ── redundant_mem_load_via_xfer ────────────────────────────────────


def test_redundant_mem_load_via_xfer_basic():
    """`mov [m], R1; mov R2, R1; ...; mov R2, [m]` drops the final
    load (R2 still holds [m]'s value via the xfer)."""
    asm = (
        "_f:\n"
        "        mov     [ebp - 4], eax\n"
        "        mov     ecx, eax\n"
        "        mov     eax, 5\n"  # doesn't touch ECX or [ebp - 4]
        "        mov     ecx, [ebp - 4]\n"  # redundant
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("redundant_mem_load_via_xfer") == 1
    # The mov ecx, [ebp - 4] should be gone (only the original
    # mov ecx, eax remains for ECX).
    out_lines = out.split("\n")
    assert not any(
        line.strip() == "mov     ecx, [ebp - 4]" for line in out_lines
    )


def test_redundant_mem_load_via_xfer_skips_when_r2_clobbered():
    """If R2 is written between the xfer and the candidate load,
    the load isn't redundant."""
    asm = (
        "_f:\n"
        "        mov     [ebp - 4], eax\n"
        "        mov     ecx, eax\n"
        "        mov     ecx, 99\n"  # ECX clobbered
        "        mov     ecx, [ebp - 4]\n"  # NOT redundant
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("redundant_mem_load_via_xfer", 0) == 0


def test_redundant_mem_load_via_xfer_skips_when_mem_clobbered():
    """If [m] is written between the xfer and the load, the load
    isn't redundant."""
    asm = (
        "_f:\n"
        "        mov     [ebp - 4], eax\n"
        "        mov     ecx, eax\n"
        "        mov     dword [ebp - 4], 42\n"  # [m] clobbered
        "        mov     ecx, [ebp - 4]\n"  # NOT redundant
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("redundant_mem_load_via_xfer", 0) == 0


def test_redundant_mem_load_via_xfer_skips_when_addr_taken():
    """If [m]'s address is taken via lea, register-base derefs
    could alias it. Conservatively skip."""
    asm = (
        "_f:\n"
        "        lea     edx, [ebp - 4]\n"  # address-take
        "        mov     [ebp - 4], eax\n"
        "        mov     ecx, eax\n"
        "        mov     [edx], 99\n"  # alias write through edx
        "        mov     ecx, [ebp - 4]\n"  # NOT redundant
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("redundant_mem_load_via_xfer", 0) == 0


# ── dup_load_chain_to_copy ─────────────────────────────────────────


def test_dup_load_chain_to_copy_basic():
    """`mov R1, [m]; mov R1, [R1]; mov R2, [m]; mov R2, [R2]` →
    `mov R1, [m]; mov R1, [R1]; mov R2, R1`. Saves 4 bytes.

    Cascade note: subsequent passes collapse the resulting
    `mov ecx, eax; imul eax, ecx` to `imul eax, eax` for an extra
    2 bytes. The test asserts the cascaded state."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     eax, [eax]\n"
        "        mov     ecx, [ebp + 8]\n"
        "        mov     ecx, [ecx]\n"
        "        imul    eax, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # The duplicate load chain is gone:
    assert "mov     ecx, [ebp + 8]" not in out
    # Pass fired:
    assert opt.stats.get("dup_load_chain_to_copy") == 1
    # Downstream cascade: imul eax, eax (squaring).
    assert "imul    eax, eax" in out


def test_dup_load_chain_to_copy_with_offset():
    """`mov R1, [m]; mov R1, [R1+N]; mov R2, [m]; mov R2, [R2+N]`
    folds when offsets match."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     eax, [eax + 4]\n"
        "        mov     ecx, [ebp + 8]\n"
        "        mov     ecx, [ecx + 4]\n"
        "        imul    eax, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("dup_load_chain_to_copy") == 1


def test_dup_load_chain_to_copy_different_offset_fires():
    """Mismatched offsets — different members of the same struct.
    First pass drops the redundant `mov R2, [m]` and inserts a
    `mov R2, R1` BEFORE B; then `copy_save_deref_collapse` cascades
    to drop that newly-inserted register copy too. Combined: 5
    bytes saved.

    Original:                       After both passes:
        mov eax, [ebp + 8]              mov ecx, [ebp + 8]
        mov eax, [eax + 4]              mov eax, [ecx + 4]
        mov ecx, [ebp + 8]              mov ecx, [ecx + 8]
        mov ecx, [ecx + 8]
    """
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     eax, [eax + 4]\n"
        "        mov     ecx, [ebp + 8]\n"
        "        mov     ecx, [ecx + 8]\n"  # different offset
        "        add     eax, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # Final form: load to ECX directly, then derefs through it.
    assert "mov     ecx, [ebp + 8]" in out
    assert "mov     eax, [ecx + 4]" in out
    assert "mov     ecx, [ecx + 8]" in out
    # The intermediate register copy is gone.
    assert "mov     ecx, eax" not in out
    # The duplicate load is gone — only one `mov ecx, [ebp + 8]`.
    assert out.count("mov     ecx, [ebp + 8]") == 1
    # Both passes fire.
    assert opt.stats.get("dup_load_chain_to_copy", 0) == 0
    assert opt.stats.get("dup_load_chain_to_copy_diff_off") == 1
    assert opt.stats.get("copy_save_deref_collapse", 0) >= 1


def test_dup_load_chain_to_copy_skips_intermediate_instr():
    """If an intermediate instruction is between the chains, the
    pass doesn't fire (conservative)."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     eax, [eax]\n"
        "        nop\n"  # intermediate
        "        mov     ecx, [ebp + 8]\n"
        "        mov     ecx, [ecx]\n"
        "        imul    eax, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("dup_load_chain_to_copy", 0) == 0


# ── copy_save_deref_collapse ──────────────────────────────────────


def test_copy_save_deref_collapse_basic():
    """The codegen's "save pointer before deref" idiom for s->a + s->b:
        mov R1, [SRC]
        mov R2, R1
        mov R1, [R1]
        mov R2, [R2 + 4]
    →
        mov R2, [SRC]
        mov R1, [R2]
        mov R2, [R2 + 4]
    Saves 2 bytes (drops the `mov R2, R1` register copy).
    """
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     ecx, eax\n"
        "        mov     eax, [eax]\n"
        "        mov     ecx, [ecx + 4]\n"
        "        add     eax, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("copy_save_deref_collapse") == 1
    # The register-copy line is gone.
    assert "mov     ecx, eax" not in out
    # The rewrite now loads the source directly to ECX.
    assert "mov     ecx, [ebp + 8]" in out
    # The first deref (offset 0) targets EAX via ECX.
    assert "mov     eax, [ecx]" in out
    # The second deref unchanged.
    assert "mov     ecx, [ecx + 4]" in out


def test_copy_save_deref_collapse_both_with_offset():
    """Both derefs have offsets — covers `s->b + s->c` style code."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     ecx, eax\n"
        "        mov     eax, [eax + 4]\n"
        "        mov     ecx, [ecx + 8]\n"
        "        add     eax, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("copy_save_deref_collapse") == 1
    assert "mov     ecx, eax" not in out
    assert "mov     ecx, [ebp + 8]" in out
    assert "mov     eax, [ecx + 4]" in out
    assert "mov     ecx, [ecx + 8]" in out


def test_copy_save_deref_collapse_with_label_source():
    """Source can be a global label memory operand."""
    asm = (
        "_f:\n"
        "        mov     eax, [_g]\n"
        "        mov     ecx, eax\n"
        "        mov     eax, [eax]\n"
        "        mov     ecx, [ecx + 4]\n"
        "        add     eax, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("copy_save_deref_collapse") == 1
    assert "mov     ecx, [_g]" in out


def test_copy_save_deref_collapse_negative_offset():
    """Negative offsets are handled (uncommon but possible)."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     ecx, eax\n"
        "        mov     eax, [eax - 4]\n"
        "        mov     ecx, [ecx + 8]\n"
        "        add     eax, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("copy_save_deref_collapse") == 1
    assert "mov     eax, [ecx - 4]" in out


def test_copy_save_deref_collapse_skips_intermediate():
    """If an intermediate instruction breaks the 4-line pattern,
    the pass doesn't fire."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     ecx, eax\n"
        "        nop\n"
        "        mov     eax, [eax]\n"
        "        mov     ecx, [ecx + 4]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("copy_save_deref_collapse", 0) == 0


def test_copy_save_deref_collapse_skips_when_b_not_register_copy():
    """B must be a register-to-register copy (not a memory load)."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     ecx, [ebp + 12]\n"  # not a copy of eax
        "        mov     eax, [eax]\n"
        "        mov     ecx, [ecx + 4]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("copy_save_deref_collapse", 0) == 0


def test_copy_save_deref_collapse_skips_when_c_not_self_deref():
    """C must deref through R1 (the loaded register), not through
    a different register or memory."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     ecx, eax\n"
        "        mov     eax, [edx]\n"  # derefs edx, not eax
        "        mov     ecx, [ecx + 4]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("copy_save_deref_collapse", 0) == 0


def test_copy_save_deref_collapse_skips_when_r1_eq_r2():
    """R1 must differ from R2 (no self-mov)."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     eax, eax\n"  # self-mov, not copy to a different reg
        "        mov     eax, [eax]\n"
        "        mov     eax, [eax + 4]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("copy_save_deref_collapse", 0) == 0


# ── dup_load_with_intermediate ────────────────────────────────────


def test_dup_load_with_intermediate_basic():
    """Linked-list traversal: load ptr, deref, accumulate, RELOAD
    ptr, deref, advance. The reload is redundant when we route the
    first load to a different register that survives the deref.

    Original:                       After:
        mov ecx, [ebp + 8]              mov eax, [ebp + 8]
        mov ecx, [ecx]                  mov ecx, [eax]
        add [ebp - 4], ecx              add [ebp - 4], ecx
        mov eax, [ebp + 8]              mov eax, [eax + 4]
        mov eax, [eax + 4]              mov [ebp + 8], eax
        mov [ebp + 8], eax
    """
    asm = (
        "_f:\n"
        "        mov     ecx, [ebp + 8]\n"
        "        mov     ecx, [ecx]\n"
        "        add     dword [ebp - 4], ecx\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     eax, [eax + 4]\n"
        "        mov     [ebp + 8], eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("dup_load_with_intermediate") == 1
    # Only ONE `mov reg, [ebp + 8]` should remain (the load).
    # The redundant second load is gone.
    out_lines = [
        ln.strip() for ln in out.split("\n")
        if ln.strip().startswith("mov     eax, [ebp + 8]")
        or ln.strip().startswith("mov     ecx, [ebp + 8]")
    ]
    assert len(out_lines) == 1


def test_dup_load_with_intermediate_skips_when_m_modified():
    """If intermediate code stores to [m], the rewrite is unsafe."""
    asm = (
        "_f:\n"
        "        mov     ecx, [ebp + 8]\n"
        "        mov     ecx, [ecx]\n"
        "        mov     [ebp + 8], 42\n"  # modifies [m]!
        "        mov     eax, [ebp + 8]\n"
        "        mov     eax, [eax + 4]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("dup_load_with_intermediate", 0) == 0


def test_dup_load_with_intermediate_skips_when_r2_read():
    """If intermediate code reads R2, the rewrite breaks because in
    the rewrite R2 holds [m]'s value, not its pre-A value."""
    asm = (
        "_f:\n"
        "        mov     ecx, [ebp + 8]\n"
        "        mov     ecx, [ecx]\n"
        "        mov     [ebp - 4], eax\n"  # reads R2 (eax)
        "        mov     eax, [ebp + 8]\n"
        "        mov     eax, [eax + 4]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("dup_load_with_intermediate", 0) == 0


def test_dup_load_with_intermediate_skips_when_r2_written():
    """If intermediate code writes R2, the rewrite breaks because
    by E, R2 doesn't hold [m] anymore."""
    asm = (
        "_f:\n"
        "        mov     ecx, [ebp + 8]\n"
        "        mov     ecx, [ecx]\n"
        "        xor     eax, eax\n"  # writes R2 (eax)
        "        mov     eax, [ebp + 8]\n"
        "        mov     eax, [eax + 4]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("dup_load_with_intermediate", 0) == 0


def test_dup_load_with_intermediate_skips_when_call_in_between():
    """Function calls invalidate the cross-block assumption (callee
    might modify [m])."""
    asm = (
        "_f:\n"
        "        mov     ecx, [ebp + 8]\n"
        "        mov     ecx, [ecx]\n"
        "        call    _helper\n"  # could modify anything
        "        mov     eax, [ebp + 8]\n"
        "        mov     eax, [eax + 4]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("dup_load_with_intermediate", 0) == 0


def test_dup_load_with_intermediate_skips_when_label_in_between():
    """Labels create CFG entry points; the rewrite would change
    semantics for any predecessor that jumps to the label."""
    asm = (
        "_f:\n"
        "        mov     ecx, [ebp + 8]\n"
        "        mov     ecx, [ecx]\n"
        ".L_mid:\n"  # external entry point
        "        mov     eax, [ebp + 8]\n"
        "        mov     eax, [eax + 4]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("dup_load_with_intermediate", 0) == 0


def test_dup_load_with_intermediate_with_offset_in_b():
    """B can have any offset (not just zero)."""
    asm = (
        "_f:\n"
        "        mov     ecx, [ebp + 8]\n"
        "        mov     ecx, [ecx + 12]\n"
        "        add     dword [ebp - 4], ecx\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     eax, [eax + 16]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("dup_load_with_intermediate") == 1
    # First deref now uses EAX as base.
    assert "mov     ecx, [eax + 12]" in out


def test_dup_load_with_intermediate_skips_when_addr_taken():
    """If [m]'s address is taken via lea, then a register-base
    write in intermediate could alias [m]."""
    asm = (
        "_f:\n"
        "        lea     edx, [ebp + 8]\n"  # address-take
        "        mov     ecx, [ebp + 8]\n"
        "        mov     ecx, [ecx]\n"
        "        mov     [edx], 99\n"  # could alias [ebp + 8]!
        "        mov     eax, [ebp + 8]\n"
        "        mov     eax, [eax + 4]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("dup_load_with_intermediate", 0) == 0


# ── uncollapse_cmp_when_reload ────────────────────────────────────


def test_uncollapse_cmp_when_reload_zero():
    """`cmp dword [m], 0; jcc L; mov reg, [m]` →
    `mov reg, [m]; test reg, reg; jcc L`. Saves 2 bytes for the
    zero-comparison case (shorter `test` form)."""
    asm = (
        "_f:\n"
        "        cmp     dword [ebp + 8], 0\n"
        "        jz      .L_end\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     ecx, [eax]\n"
        "        ret\n"
        ".L_end:\n"
        "        xor     eax, eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("uncollapse_cmp_when_reload") == 1
    # The cmp dword [m], 0 is gone; replaced with mov + test.
    assert "cmp     dword [ebp + 8], 0" not in out
    assert "mov     eax, [ebp + 8]" in out
    assert "test    eax, eax" in out
    # The redundant reload is gone.
    assert out.count("mov     eax, [ebp + 8]") == 1


def test_uncollapse_cmp_when_reload_nonzero():
    """For non-zero CONST, we use cmp reg, CONST."""
    asm = (
        "_f:\n"
        "        cmp     dword [ebp + 8], 42\n"
        "        je      .L_end\n"
        "        mov     eax, [ebp + 8]\n"
        "        ret\n"
        ".L_end:\n"
        "        xor     eax, eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("uncollapse_cmp_when_reload") == 1
    assert "cmp     dword [ebp + 8], 42" not in out
    assert "mov     eax, [ebp + 8]" in out
    assert "cmp     eax, 42" in out


def test_uncollapse_cmp_when_reload_register_operand():
    """`cmp dword [m], reg2; jcc L; mov reg, [m]` works too."""
    asm = (
        "_f:\n"
        "        mov     edx, 5\n"
        "        cmp     dword [ebp + 8], edx\n"
        "        jl      .L_end\n"
        "        mov     eax, [ebp + 8]\n"
        "        ret\n"
        ".L_end:\n"
        "        xor     eax, eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("uncollapse_cmp_when_reload") == 1
    assert "cmp     eax, edx" in out


def test_uncollapse_cmp_when_reload_skips_when_reg_live_at_target():
    """If the jcc target reads `reg` (its pre-A value), the rewrite
    breaks because in the rewrite reg now holds [m]'s value."""
    asm = (
        "_f:\n"
        "        cmp     dword [ebp + 8], 0\n"
        "        jz      .L_end\n"
        "        mov     eax, [ebp + 8]\n"
        "        ret\n"
        ".L_end:\n"
        "        add     ecx, eax\n"  # reads EAX!
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("uncollapse_cmp_when_reload", 0) == 0


def test_uncollapse_cmp_when_reload_skips_when_mem_differs():
    """If A's mem operand differs from C's, no fold."""
    asm = (
        "_f:\n"
        "        cmp     dword [ebp + 8], 0\n"
        "        jz      .L_end\n"
        "        mov     eax, [ebp + 12]\n"  # different slot
        "        ret\n"
        ".L_end:\n"
        "        xor     eax, eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("uncollapse_cmp_when_reload", 0) == 0


def test_uncollapse_cmp_when_reload_skips_byte_cmp():
    """A `byte` cmp would compare only the low 8 bits. A subsequent
    `mov reg32, [m]` loads 4 bytes — sizes mismatch, so the rewrite
    `mov reg, [m]; cmp reg, X` would compare all 32 bits, giving
    different ZF for non-low-byte-zero values."""
    asm = (
        "_f:\n"
        "        cmp     byte [ebp + 8], 0\n"
        "        jz      .L_end\n"
        "        mov     eax, [ebp + 8]\n"
        "        ret\n"
        ".L_end:\n"
        "        xor     eax, eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("uncollapse_cmp_when_reload", 0) == 0


def test_uncollapse_cmp_when_reload_skips_when_mem_uses_reg():
    """If [m] references the load's destination register (e.g.
    [eax + 4]), the rewrite would change the address calculation."""
    asm = (
        "_f:\n"
        "        cmp     dword [eax + 4], 0\n"
        "        jz      .L_end\n"
        "        mov     eax, [eax + 4]\n"  # eax used as base for [m]
        "        ret\n"
        ".L_end:\n"
        "        xor     eax, eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("uncollapse_cmp_when_reload", 0) == 0


def test_uncollapse_cmp_when_reload_jne_form():
    """Any conditional jump variant works (jne, jl, jg, etc.)."""
    asm = (
        "_f:\n"
        "        cmp     dword [ebp + 8], 0\n"
        "        jne     .L_body\n"
        "        ret\n"
        ".L_body:\n"
        "        xor     eax, eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # No reload after jne — nothing to do.
    assert opt.stats.get("uncollapse_cmp_when_reload", 0) == 0


# ── redundant_cmp_at_label ────────────────────────────────────────


def test_redundant_cmp_at_label_basic():
    """If/elif chain testing the same expression. Each label after
    a Jcc has flags from the original cmp; the duplicate cmp is
    redundant.

    Original:                       After:
        cmp [m], 0                      cmp [m], 0
        jge .L_else                     jge .L_else
        mov eax, -1                     mov eax, -1
        jmp .epilogue                   jmp .epilogue
        .L_else:                        .L_else:
        cmp [m], 0     ; DROP           jne .L_other
        jne .L_other
    """
    asm = (
        "_f:\n"
        "        cmp     dword [ebp + 8], 0\n"
        "        jge     .L_else\n"
        "        mov     eax, -1\n"
        "        jmp     .epilogue\n"
        ".L_else:\n"
        "        cmp     dword [ebp + 8], 0\n"
        "        jne     .L_other\n"
        "        xor     eax, eax\n"
        "        jmp     .epilogue\n"
        ".L_other:\n"
        "        mov     eax, 1\n"
        ".epilogue:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("redundant_cmp_at_label") == 1
    # The first cmp is preserved, the duplicate at .L_else: is dropped.
    assert out.count("cmp     dword [ebp + 8], 0") == 1


def test_redundant_cmp_at_label_skips_when_operands_differ():
    """If the cmp at the label has different operands, it's not
    a duplicate."""
    asm = (
        "_f:\n"
        "        cmp     dword [ebp + 8], 0\n"
        "        jge     .L_else\n"
        "        mov     eax, -1\n"
        "        jmp     .epilogue\n"
        ".L_else:\n"
        "        cmp     dword [ebp + 8], 10\n"  # different RHS
        "        jne     .L_other\n"
        "        xor     eax, eax\n"
        ".L_other:\n"
        "        mov     eax, 1\n"
        ".epilogue:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("redundant_cmp_at_label", 0) == 0


def test_redundant_cmp_at_label_skips_when_label_has_other_refs():
    """If the label has other references (more than one Jcc), the
    duplicate cmp can't be safely dropped because flags from one
    Jcc's cmp may differ from another's."""
    asm = (
        "_f:\n"
        "        cmp     dword [ebp + 8], 0\n"
        "        jge     .L_else\n"
        "        cmp     dword [ebp + 8], 5\n"  # different cmp
        "        jne     .L_else\n"  # second ref to .L_else
        "        mov     eax, -1\n"
        "        jmp     .epilogue\n"
        ".L_else:\n"
        "        cmp     dword [ebp + 8], 0\n"
        "        jne     .L_other\n"
        ".L_other:\n"
        "        mov     eax, 1\n"
        ".epilogue:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("redundant_cmp_at_label", 0) == 0


def test_redundant_cmp_at_label_skips_when_label_falls_through():
    """If the label is preceded by a non-terminator (e.g. text
    fallthrough is possible), flags may not be preserved."""
    asm = (
        "_f:\n"
        "        cmp     dword [ebp + 8], 0\n"
        "        jge     .L_else\n"
        "        nop\n"  # falls through to .L_else
        ".L_else:\n"
        "        cmp     dword [ebp + 8], 0\n"
        "        jne     .L_other\n"
        ".L_other:\n"
        "        mov     eax, 1\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("redundant_cmp_at_label", 0) == 0


def test_redundant_cmp_at_label_test_form():
    """`test` form also works."""
    asm = (
        "_f:\n"
        "        test    eax, eax\n"
        "        jz      .L_zero\n"
        "        mov     eax, 1\n"
        "        jmp     .epilogue\n"
        ".L_zero:\n"
        "        test    eax, eax\n"
        "        js      .L_neg\n"
        ".L_neg:\n"
        "        xor     eax, eax\n"
        ".epilogue:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("redundant_cmp_at_label") == 1
    assert out.count("test    eax, eax") == 1


def test_redundant_cmp_at_label_jne_path():
    """The jcc can be any conditional flavor."""
    asm = (
        "_f:\n"
        "        cmp     dword [ebp + 8], 5\n"
        "        jne     .L_branch\n"
        "        mov     eax, 0\n"
        "        jmp     .epilogue\n"
        ".L_branch:\n"
        "        cmp     dword [ebp + 8], 5\n"
        "        jl      .L_other\n"
        ".L_other:\n"
        "        mov     eax, 1\n"
        ".epilogue:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("redundant_cmp_at_label") == 1


# ── op_mem_to_reg_collapse ────────────────────────────────────────


def test_op_mem_to_reg_collapse_add():
    """`add eax, [m]; mov [m], eax` → `add [m], eax`. Saves 3 bytes."""
    asm = (
        "_f:\n"
        "        call    _helper\n"
        "        add     eax, dword [ebp - 4]\n"
        "        mov     [ebp - 4], eax\n"
        "        xor     eax, eax\n"  # eax dead
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("op_mem_to_reg_collapse") == 1
    assert "add     [ebp - 4], eax" in out
    # Original load+store gone.
    assert "add     eax, dword [ebp - 4]" not in out


def test_op_mem_to_reg_collapse_or():
    """`or eax, [m]; mov [m], eax` → `or [m], eax`."""
    asm = (
        "_f:\n"
        "        call    _helper\n"
        "        or      eax, [ebp - 4]\n"
        "        mov     [ebp - 4], eax\n"
        "        xor     eax, eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("op_mem_to_reg_collapse") == 1
    assert "or      [ebp - 4], eax" in out


def test_op_mem_to_reg_collapse_skips_sub():
    """sub is non-commutative: `sub reg, [m]; mov [m], reg` ≠
    `sub [m], reg`. Pass should skip."""
    asm = (
        "_f:\n"
        "        call    _helper\n"
        "        sub     eax, [ebp - 4]\n"
        "        mov     [ebp - 4], eax\n"
        "        xor     eax, eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("op_mem_to_reg_collapse", 0) == 0


def test_op_mem_to_reg_collapse_skips_when_reg_live():
    """If reg is read after the store, the rewrite changes its
    value (reg keeps its pre-OP value rather than being modified
    by the OP). Unsafe."""
    asm = (
        "_f:\n"
        "        call    _helper\n"
        "        add     eax, [ebp - 4]\n"
        "        mov     [ebp - 4], eax\n"
        "        add     ecx, eax\n"  # reads eax
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("op_mem_to_reg_collapse", 0) == 0


def test_op_mem_to_reg_collapse_skips_when_mem_differs():
    """A's [m] must equal B's [m]."""
    asm = (
        "_f:\n"
        "        call    _helper\n"
        "        add     eax, [ebp - 4]\n"
        "        mov     [ebp - 8], eax\n"  # different slot
        "        xor     eax, eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("op_mem_to_reg_collapse", 0) == 0


def test_op_mem_to_reg_collapse_and_xor():
    """`and` and `xor` also work."""
    asm = (
        "_f:\n"
        "        call    _h1\n"
        "        and     eax, [ebp - 4]\n"
        "        mov     [ebp - 4], eax\n"
        "        call    _h2\n"
        "        xor     eax, [ebp - 8]\n"
        "        mov     [ebp - 8], eax\n"
        "        xor     eax, eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("op_mem_to_reg_collapse") == 2
    assert "and     [ebp - 4], eax" in out
    assert "xor     [ebp - 8], eax" in out


# ── push_pop_to_free_reg ──────────────────────────────────────────


def test_push_pop_to_free_reg_basic():
    """Dot-product loop body: push a[i] saves across b[i]
    computation; pop ecx + imul consumes. With EDX free, we save
    in EDX directly."""
    asm = (
        "_compute:\n"
        "        push    dword [eax + ecx*4]\n"
        "        mov     eax, [ebp + 12]\n"
        "        mov     eax, [eax + ecx*4]\n"
        "        pop     ecx\n"
        "        imul    eax, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("push_pop_to_free_reg") == 1
    # Push and pop both gone.
    assert "push    dword [eax + ecx*4]" not in out
    assert "pop     ecx" not in out
    # mov edx, [eax + ecx*4] now at the push position.
    assert "mov     edx, [eax + ecx*4]" in out
    # imul retargeted to use EDX.
    assert "imul    eax, edx" in out


def test_push_pop_to_free_reg_skips_when_no_free_reg():
    """If all GP regs are touched in the function, no free reg is
    available."""
    asm = (
        "_compute:\n"
        "        push    dword [eax + ecx*4]\n"
        "        mov     eax, [ebp + 12]\n"
        "        mov     eax, [eax + ecx*4]\n"
        "        pop     ecx\n"
        "        imul    eax, ecx\n"
        "        mov     edx, 5\n"  # uses EDX
        "        mov     esi, 7\n"  # uses ESI
        "        mov     edi, 9\n"  # uses EDI
        "        mov     ebx, 1\n"  # uses EBX
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("push_pop_to_free_reg", 0) == 0


def test_push_pop_to_free_reg_skips_when_chain_uses_esp():
    """Chain referencing [esp + N] would break after the push is
    dropped — ESP doesn't decrement, so addresses shift."""
    asm = (
        "_f:\n"
        "        push    dword [eax + ecx*4]\n"
        "        mov     eax, [esp + 4]\n"  # ESP-relative!
        "        pop     ecx\n"
        "        imul    eax, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("push_pop_to_free_reg", 0) == 0


def test_push_pop_to_free_reg_skips_when_consumer_doesnt_use_reg():
    """If the consumer (next instr after pop) doesn't read REG,
    the pop is doing something we don't model — bail."""
    asm = (
        "_f:\n"
        "        push    dword [eax + ecx*4]\n"
        "        mov     eax, [ebp + 12]\n"
        "        pop     ecx\n"
        "        ret\n"  # consumer doesn't ref ECX
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("push_pop_to_free_reg", 0) == 0


def test_push_pop_to_free_reg_skips_when_reg_live_after():
    """If REG is read after the consumer, we can't retarget just
    the consumer — would leave subsequent uses with wrong value."""
    asm = (
        "_f:\n"
        "        push    dword [eax + ecx*4]\n"
        "        mov     eax, [ebp + 12]\n"
        "        pop     ecx\n"
        "        imul    eax, ecx\n"
        "        add     eax, ecx\n"  # second use of ECX
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("push_pop_to_free_reg", 0) == 0


def test_push_pop_to_free_reg_skips_imul_with_implicit_reg():
    """Implicit-reg consumers (cdq, idiv, mul, etc.) can't be
    safely retargeted."""
    asm = (
        "_f:\n"
        "        push    dword [eax + ecx*4]\n"
        "        mov     eax, [ebp + 12]\n"
        "        pop     ecx\n"
        "        idiv    ecx\n"  # implicit EDX:EAX
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("push_pop_to_free_reg", 0) == 0


# ── redundant_recompute_after_cmp ─────────────────────────────────


def test_redundant_recompute_after_cmp_basic():
    """Ternary `m = arr[i] > m ? arr[i] : m` recomputes arr[i]
    after the cmp+jcc. Drop the second compute."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     eax, [eax + ecx*4]\n"
        "        cmp     eax, [ebp - 4]\n"
        "        jle     .L_skip\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     eax, [eax + ecx*4]\n"
        "        mov     [ebp - 4], eax\n"
        ".L_skip:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("redundant_recompute_after_cmp") == 1
    # Only ONE pair of `mov eax, [ebp + 8]` + SIB load remains.
    assert out.count("mov     eax, [ebp + 8]") == 1
    assert out.count("mov     eax, [eax + ecx*4]") == 1


def test_redundant_recompute_after_cmp_with_disp():
    """SIB form with displacement also handled."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     eax, [eax + ecx*4 + 8]\n"
        "        cmp     eax, [ebp - 4]\n"
        "        jle     .L_skip\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     eax, [eax + ecx*4 + 8]\n"
        "        mov     [ebp - 4], eax\n"
        ".L_skip:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("redundant_recompute_after_cmp") == 1


def test_redundant_recompute_after_cmp_test_form():
    """`test reg, reg` form also works."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     eax, [eax + ecx*4]\n"
        "        test    eax, eax\n"
        "        jz      .L_skip\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     eax, [eax + ecx*4]\n"
        "        mov     [ebp - 4], eax\n"
        ".L_skip:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("redundant_recompute_after_cmp") == 1


def test_redundant_recompute_after_cmp_skips_when_a_e_differ():
    """If A and E load from different memory, can't dedup."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     eax, [eax + ecx*4]\n"
        "        cmp     eax, [ebp - 4]\n"
        "        jle     .L_skip\n"
        "        mov     eax, [ebp + 12]\n"  # different m
        "        mov     eax, [eax + ecx*4]\n"
        "        mov     [ebp - 4], eax\n"
        ".L_skip:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("redundant_recompute_after_cmp", 0) == 0


def test_redundant_recompute_after_cmp_skips_when_b_f_differ():
    """If B and F have different scale or idx, can't dedup."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     eax, [eax + ecx*4]\n"
        "        cmp     eax, [ebp - 4]\n"
        "        jle     .L_skip\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     eax, [eax + ecx*8]\n"  # different scale
        "        mov     [ebp - 4], eax\n"
        ".L_skip:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("redundant_recompute_after_cmp", 0) == 0


def test_redundant_recompute_after_cmp_with_idx_setup():
    """Real-world `find_max` shape: A, idx_setup (mov ecx, [...]),
    B, cmp, jcc, E (= A), F (= B). The duplicate omits idx_setup
    because ECX is unchanged."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     ecx, [ebp - 8]\n"  # idx_setup
        "        mov     eax, [eax + ecx*4]\n"
        "        cmp     eax, [ebp - 4]\n"
        "        jle     .L_skip\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     eax, [eax + ecx*4]\n"
        "        mov     [ebp - 4], eax\n"
        ".L_skip:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("redundant_recompute_after_cmp") == 1
    # The duplicate at the end is gone.
    assert out.count("mov     eax, [ebp + 8]") == 1
    assert out.count("mov     eax, [eax + ecx*4]") == 1


def test_redundant_recompute_after_cmp_skips_when_intermediate_writes_reg():
    """If intermediate writes EAX, the SIB load uses a different
    base than the duplicate would."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     eax, [some_other]\n"  # writes EAX
        "        mov     eax, [eax + ecx*4]\n"
        "        cmp     eax, [ebp - 4]\n"
        "        jle     .L_skip\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     eax, [eax + ecx*4]\n"
        ".L_skip:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("redundant_recompute_after_cmp", 0) == 0


# ── xfer_store_collapse ───────────────────────────────────────────


def test_xfer_store_collapse_basic():
    """`mov eax, edx; mov [m], eax` → `mov [m], edx` when EAX dead."""
    asm = (
        "_f:\n"
        "        cdq\n"
        "        idiv    dword [ebp + 12]\n"
        "        mov     eax, edx\n"
        "        mov     [ebp + 12], eax\n"
        "        mov     eax, [ebp - 4]\n"  # eax overwritten = dead
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     [ebp + 12], edx" in out
    out_lines = out.split("\n")
    has_xfer = any(
        line.strip() == "mov     eax, edx" for line in out_lines
    )
    assert not has_xfer
    assert opt.stats.get("xfer_store_collapse") == 1


def test_xfer_store_collapse_skips_when_r2_live():
    """If R2 is read after the store, can't drop the transfer."""
    asm = (
        "_f:\n"
        "        mov     eax, edx\n"
        "        mov     [ebp - 4], eax\n"
        "        push    eax\n"  # eax read here
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("xfer_store_collapse", 0) == 0


def test_xfer_store_collapse_skips_when_dst_uses_r2():
    """If the store address references R2, the rewrite changes
    addressing semantics."""
    asm = (
        "_f:\n"
        "        mov     eax, edx\n"
        "        mov     [eax + 4], eax\n"  # address uses eax
        "        mov     eax, [ebp - 4]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("xfer_store_collapse", 0) == 0


def test_xfer_store_collapse_with_size_prefix():
    """Works with explicit size prefix."""
    asm = (
        "_f:\n"
        "        mov     eax, ebx\n"
        "        mov     dword [ebp - 4], eax\n"
        "        mov     eax, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     dword [ebp - 4], ebx" in out
    assert opt.stats.get("xfer_store_collapse") == 1


# ── load_add_xfer_forward ─────────────────────────────────────────


def test_load_add_xfer_forward_basic():
    """`mov R1, [m]; add R1, IMM; mov R2, R1` collapses to
    `mov R2, [m]; add R2, IMM` when R1 is dead after.

    Cascade note: after this pass converts the load+add+xfer to
    `mov ecx, [ebp+8]; add ecx, 4`, disp_store_collapse may further
    fold the trailing `mov [ecx], eax` into `mov [ecx + 4], eax`,
    dropping the add too."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        add     eax, 4\n"
        "        mov     ecx, eax\n"
        "        mov     eax, [ebp + 16]\n"  # eax overwritten ⇒ dead
        "        mov     [ecx], eax\n"
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # Pass fired at least once.
    assert opt.stats.get("load_add_xfer_forward", 0) >= 1
    # The transfer should be gone (cascading effect).
    out_lines = out.split("\n")
    has_xfer = any(
        line.strip() == "mov     ecx, eax" for line in out_lines
    )
    assert not has_xfer


def test_load_add_xfer_forward_skips_when_r1_live():
    """If R1 is read after the transfer, we can't drop it."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        add     eax, 4\n"
        "        mov     ecx, eax\n"
        "        mov     [ebp - 4], eax\n"  # eax used here
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("load_add_xfer_forward", 0) == 0


def test_load_add_xfer_forward_with_sub():
    """`mov R1, [m]; sub R1, IMM; mov R2, R1` also collapses."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        sub     eax, 8\n"
        "        mov     ecx, eax\n"
        "        mov     eax, edx\n"  # eax overwritten = dead
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("load_add_xfer_forward", 0) >= 1


def test_load_add_xfer_forward_rejects_self_reference():
    """If SRC references R2, the rewrite would self-reference."""
    asm = (
        "_f:\n"
        "        mov     eax, [ecx + 8]\n"  # SRC references ECX
        "        add     eax, 4\n"
        "        mov     ecx, eax\n"  # would become mov ecx, [ecx + 8]
        "        mov     eax, edx\n"
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("load_add_xfer_forward", 0) == 0


def test_load_add_xfer_forward_rejects_non_literal_imm():
    """Non-numeric add operand doesn't collapse."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        add     eax, _label\n"
        "        mov     ecx, eax\n"
        "        mov     eax, edx\n"
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("load_add_xfer_forward", 0) == 0


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


def test_redundant_ecx_load_through_register_base_store():
    """When the function is address-clean (no `lea` instructions),
    a `mov [ecx + offset], src` write doesn't invalidate ECX tracking
    because the destination can't alias `[ebp + N]`."""
    asm = (
        "_set_point:\n"
        "        push    ebp\n"
        "        mov     ebp, esp\n"
        "        mov     ecx, [ebp + 8]\n"
        "        mov     eax, [ebp + 12]\n"
        "        mov     [ecx], eax\n"
        "        mov     ecx, [ebp + 8]\n"  # redundant — pure store via [ecx]
        "        mov     eax, [ebp + 16]\n"
        "        mov     [ecx + 4], eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("redundant_ecx_load") == 1


def test_redundant_ecx_load_NOT_collapsed_when_lea_present():
    """When the function has `lea reg, [ebp ± N]`, `[ebp + N]` may
    be address-taken — be conservative and invalidate on register-base
    stores."""
    asm = (
        "_f:\n"
        "        push    ebp\n"
        "        mov     ebp, esp\n"
        "        lea     edx, [ebp + 8]\n"  # &param_8 escapes
        "        mov     ecx, [ebp + 8]\n"
        "        mov     eax, 5\n"
        "        mov     [edx], eax\n"      # might write [ebp + 8]
        "        mov     ecx, [ebp + 8]\n"  # NOT redundant — [edx] could be [ebp + 8]
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("redundant_ecx_load", 0) == 0


def test_redundant_ecx_load_collapsed_when_lea_targets_other_slot():
    """`lea reg, [ebp - 4]` makes `-4` address-taken, but `[ebp + 8]`
    is still address-clean — its reload through a register-base store
    can be collapsed."""
    asm = (
        "_f:\n"
        "        push    ebp\n"
        "        mov     ebp, esp\n"
        "        lea     edx, [ebp - 4]\n"  # only [ebp - 4] is taken
        "        mov     ecx, [ebp + 8]\n"
        "        mov     eax, 5\n"
        "        mov     [ecx], eax\n"      # store via [ecx], ECX held [ebp + 8]
        "        mov     ecx, [ebp + 8]\n"  # redundant — [ecx] != [ebp + 8]
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


# ── indirect_call_collapse ───────────────────────────────────────


def test_indirect_call_collapse_ebp_rel():
    """`mov eax, [ebp + 8]; call eax` → `call dword [ebp + 8]`."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        call    eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     eax, [ebp + 8]" not in out
    assert "call    dword [ebp + 8]" in out
    assert opt.stats.get("indirect_call_collapse") == 1


def test_indirect_call_collapse_sib_form():
    """`mov eax, [eax + 4]; call eax` → `call dword [eax + 4]`.
    Even when MEM references the target reg, the rewrite is correct
    because the call reads the same address."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     eax, [eax + 4]\n"
        "        call    eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "call    dword [eax + 4]" in out
    assert opt.stats.get("indirect_call_collapse") == 1


def test_indirect_call_collapse_label():
    """`mov eax, [_glob]; call eax` → `call dword [_glob]`."""
    asm = (
        "_f:\n"
        "        mov     eax, [_fnptr]\n"
        "        call    eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     eax, [_fnptr]" not in out
    assert "call    dword [_fnptr]" in out
    assert opt.stats.get("indirect_call_collapse") == 1


def test_indirect_call_collapse_other_reg():
    """The pass works for any 32-bit GP register, not just eax."""
    asm = (
        "_f:\n"
        "        mov     edx, [ebp - 4]\n"
        "        call    edx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     edx, [ebp - 4]" not in out
    assert "call    dword [ebp - 4]" in out
    assert opt.stats.get("indirect_call_collapse") == 1


def test_indirect_call_collapse_skips_immediate_source():
    """`mov eax, _foo; call eax` is NOT this pattern (codegen would
    emit `call _foo` directly anyway). Source must be a memory
    operand."""
    asm = (
        "_f:\n"
        "        mov     eax, _foo\n"
        "        call    eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("indirect_call_collapse", 0) == 0


def test_indirect_call_collapse_skips_register_mismatch():
    """`mov eax, [m]; call edx` is NOT a match — different regs."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        call    edx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("indirect_call_collapse", 0) == 0


def test_indirect_call_collapse_skips_direct_call():
    """A direct `call _foo` (no preceding mov) is unaffected."""
    asm = (
        "_f:\n"
        "        call    _foo\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "call    _foo" in out
    assert opt.stats.get("indirect_call_collapse", 0) == 0


def test_dead_cleanup_before_leave_pop_ecx():
    """`pop ecx` immediately before `leave; ret` is dead — drop it."""
    asm = (
        "_f:\n"
        "        enter   0, 0\n"
        "        call    _g\n"
        "        pop     ecx\n"
        ".epilogue:\n"
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "pop     ecx" not in out
    assert opt.stats.get("dead_cleanup_before_leave") == 1


def test_dead_cleanup_before_leave_multiple_pops():
    """Multiple consecutive `pop ecx` are all dropped."""
    asm = (
        "_f:\n"
        "        enter   0, 0\n"
        "        call    _g\n"
        "        pop     ecx\n"
        "        pop     ecx\n"
        "        pop     ecx\n"
        ".epilogue:\n"
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "pop     ecx" not in out
    assert opt.stats.get("dead_cleanup_before_leave") == 3


def test_dead_cleanup_before_leave_add_esp():
    """`add esp, N` before `leave; ret` is dead — drop it."""
    asm = (
        "_f:\n"
        "        enter   0, 0\n"
        "        call    _g\n"
        "        add     esp, 12\n"
        ".epilogue:\n"
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "add     esp, 12" not in out
    assert opt.stats.get("dead_cleanup_before_leave") == 1


def test_dead_cleanup_before_leave_pop_edx():
    """`pop edx` is also droppable (caller-saved scratch)."""
    asm = (
        "_f:\n"
        "        enter   0, 0\n"
        "        pop     edx\n"
        ".epilogue:\n"
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "pop     edx" not in out
    assert opt.stats.get("dead_cleanup_before_leave") == 1


def test_dead_cleanup_before_leave_skips_pop_eax():
    """`pop eax` is NOT dropped — EAX is the return value."""
    asm = (
        "_f:\n"
        "        enter   0, 0\n"
        "        pop     eax\n"
        ".epilogue:\n"
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "pop     eax" in out
    assert opt.stats.get("dead_cleanup_before_leave", 0) == 0


def test_dead_cleanup_before_leave_skips_callee_saved():
    """Pops of EBX/ESI/EDI/EBP are NOT dropped — they're restoring
    callee-saved values from the prologue."""
    for reg in ("ebx", "esi", "edi", "ebp"):
        asm = (
            "_f:\n"
            "        push    {0}\n"
            "        enter   0, 0\n"
            "        pop     {0}\n"
            ".epilogue:\n"
            "        leave\n"
            "        ret\n"
        ).format(reg)
        opt = PeepholeOptimizer()
        out = opt.optimize(asm)
        assert f"pop     {reg}" in out
        assert opt.stats.get("dead_cleanup_before_leave", 0) == 0


def test_dead_cleanup_before_leave_label_between_cleanup_and_leave():
    """A label between cleanup and leave doesn't block dropping —
    the cleanup is still dead."""
    asm = (
        "_f:\n"
        "        enter   0, 0\n"
        "        call    _g\n"
        "        pop     ecx\n"
        ".epilogue:\n"
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "pop     ecx" not in out
    assert ".epilogue:" in out  # label preserved
    assert opt.stats.get("dead_cleanup_before_leave") == 1


def test_dead_cleanup_before_leave_skips_real_work():
    """Non-cleanup instruction before leave is not dropped."""
    asm = (
        "_f:\n"
        "        enter   0, 0\n"
        "        mov     eax, 42\n"
        ".epilogue:\n"
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     eax, 42" in out
    assert opt.stats.get("dead_cleanup_before_leave", 0) == 0


def test_dead_cleanup_before_leave_codegen_integration():
    """End-to-end: a function that calls another then exits drops
    the cleanup pop."""
    from uc386.codegen import CodeGenerator
    from uc_core.lexer import Lexer
    from uc_core.parser import Parser

    src = (
        "int g(int x) { return x * 2; }\n"
        "int f(int x) { return g(x); }\n"
        "int main(void) { return f(5); }\n"
    )
    tokens = list(Lexer(src, "test.c").tokenize())
    tu = Parser(tokens).parse()
    cg = CodeGenerator(peephole=True)
    asm = cg.generate(tu)
    # Function `_f` calls `_g`, then needs no cleanup before leave.
    # Find `_f:` to `_main:` block.
    fstart = asm.index("_f:")
    fend = asm.index("_main:")
    f_block = asm[fstart:fend]
    # The cleanup `pop ecx` between the call and `.epilogue:` should
    # be dropped.
    assert "pop     ecx" not in f_block


def test_cmp_load_promote_basic():
    """Canonical loop-top pattern fires the rewrite."""
    asm = (
        "_f:\n"
        "        enter   8, 0\n"
        "        xor     eax, eax\n"
        "        mov     [ebp - 8], eax\n"
        ".L1_for_top:\n"
        "        mov     eax, [ebp - 8]\n"
        "        cmp     eax, [ebp + 12]\n"
        "        jge     .L3_for_end\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     ecx, [ebp - 8]\n"
        "        mov     eax, [eax + ecx*4]\n"
        "        add     [ebp - 4], eax\n"
        "        inc     dword [ebp - 8]\n"
        "        jmp     .L1_for_top\n"
        ".L3_for_end:\n"
        "        mov     eax, [ebp - 4]\n"
        ".epilogue:\n"
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # Promoted: load to ecx instead of eax.
    assert "mov     ecx, [ebp - 8]" in out
    # Cmp retargeted to use ecx.
    assert "cmp     ecx, [ebp + 12]" in out
    # The duplicate `mov ecx, [ebp - 8]` after `mov eax, [ebp + 8]`
    # is dropped — only one ecx-load now.
    assert out.count("mov     ecx, [ebp - 8]") == 1
    assert opt.stats.get("cmp_load_promote") == 1


def test_cmp_load_promote_skips_non_loop_top():
    """Without a preceding label, the pass doesn't fire."""
    asm = (
        "_f:\n"
        "        enter   0, 0\n"
        "        mov     eax, [ebp - 8]\n"  # not after label
        "        cmp     eax, [ebp + 12]\n"
        "        jge     .L_end\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     ecx, [ebp - 8]\n"
        "        mov     eax, [eax + ecx*4]\n"
        ".L_end:\n"
        "        mov     eax, 0\n"
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("cmp_load_promote", 0) == 0


def test_cmp_load_promote_skips_self_rmw_d():
    """If D references REG1 (self-RMW), the rewrite is unsafe — skip."""
    asm = (
        "_f:\n"
        "        enter   0, 0\n"
        ".L_top:\n"
        "        mov     eax, [ebp - 8]\n"
        "        cmp     eax, 10\n"
        "        jge     .L_end\n"
        "        mov     eax, [eax + 4]\n"  # D references EAX (self-RMW)
        "        mov     ecx, [ebp - 8]\n"
        "        ret\n"
        ".L_end:\n"
        "        mov     eax, 0\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("cmp_load_promote", 0) == 0


def test_cmp_load_promote_skips_b_refs_reg2():
    """If B's second operand references REG2, retargeting changes
    cmp's semantics — skip."""
    asm = (
        "_f:\n"
        "        enter   0, 0\n"
        ".L_top:\n"
        "        mov     eax, [ebp - 8]\n"
        "        cmp     eax, ecx\n"  # B's other operand is ecx
        "        jge     .L_end\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     ecx, [ebp - 8]\n"
        "        ret\n"
        ".L_end:\n"
        "        mov     eax, 0\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("cmp_load_promote", 0) == 0


def test_cmp_load_promote_skips_mismatched_mem():
    """If A's [m] differs from E's [m], no redundancy — skip."""
    asm = (
        "_f:\n"
        "        enter   0, 0\n"
        ".L_top:\n"
        "        mov     eax, [ebp - 8]\n"
        "        cmp     eax, 10\n"
        "        jge     .L_end\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     ecx, [ebp - 12]\n"  # different memory
        "        ret\n"
        ".L_end:\n"
        "        mov     eax, 0\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("cmp_load_promote", 0) == 0


def test_cmp_load_promote_skips_target_reads_reg1():
    """If at the jcc target REG1 is read before being overwritten,
    the rewrite would leak the wrong value to the target — skip."""
    asm = (
        "_f:\n"
        "        enter   0, 0\n"
        ".L_top:\n"
        "        mov     eax, [ebp - 8]\n"
        "        cmp     eax, 10\n"
        "        jge     .L_end\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     ecx, [ebp - 8]\n"
        "        ret\n"
        ".L_end:\n"
        "        add     [ebp - 4], eax\n"  # reads eax at target
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("cmp_load_promote", 0) == 0


def test_cmp_load_promote_codegen_integration():
    """End-to-end: a typical for-loop indexing pattern triggers
    the rewrite."""
    from uc386.codegen import CodeGenerator
    from uc_core.lexer import Lexer
    from uc_core.parser import Parser

    src = (
        "int sum_arr(int *arr, int n) {\n"
        "    int s = 0;\n"
        "    for (int i = 0; i < n; i++) {\n"
        "        s += arr[i];\n"
        "    }\n"
        "    return s;\n"
        "}\n"
        "int main(void) { return 0; }\n"
    )
    tokens = list(Lexer(src, "test.c").tokenize())
    tu = Parser(tokens).parse()
    cg = CodeGenerator(peephole=True)
    asm = cg.generate(tu)
    # The loop top should now load i directly into ecx for the cmp.
    assert "mov     ecx, [ebp - 8]" in asm
    assert "cmp     ecx, [ebp + 12]" in asm
    # No second `mov ecx, [ebp - 8]` (duplicate dropped).
    assert asm.count("mov     ecx, [ebp - 8]") == 1


def test_indirect_call_collapse_codegen_integration():
    """End-to-end: function pointer call lowers to a single
    `call dword [ebp + N]` after peephole."""
    from uc386.codegen import CodeGenerator
    from uc_core.lexer import Lexer
    from uc_core.parser import Parser

    src = (
        "int dispatch(int (*fp)(int, int), int x, int y) {\n"
        "    return fp(x, y);\n"
        "}\n"
        "int main(void) { return 0; }\n"
    )
    tokens = list(Lexer(src, "test.c").tokenize())
    tu = Parser(tokens).parse()
    cg = CodeGenerator(peephole=True)
    asm = cg.generate(tu)
    # The function pointer load + call should fuse to a single
    # mem-form call.
    assert "call    dword [ebp + 8]" in asm
    assert "mov     eax, [ebp + 8]\n        call    eax" not in asm


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


# ── transfer_pop_cmp_collapse ────────────────────────────────────


def test_transfer_pop_cmp_collapse_jne():
    """`mov ecx, eax; pop eax; cmp eax, ecx; jne L` → `pop ecx; cmp; jne L`.
    ZF is symmetric, so jne sees the same outcome.

    Use a multi-instruction chain so binop_collapse can't fire on
    push/load/transfer/pop and consume the pattern."""
    asm = (
        "_f:\n"
        "        push    eax\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     ecx, [ebp - 8]\n"
        "        mov     eax, [eax + ecx*4]\n"
        "        mov     ecx, eax\n"
        "        pop     eax\n"
        "        cmp     eax, ecx\n"
        "        jne     .L_end\n"
        "        mov     eax, 1\n"
        ".L_end:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "pop     ecx" in out
    assert "cmp     eax, ecx" in out
    assert opt.stats.get("transfer_pop_cmp_collapse") == 1


def test_transfer_pop_cmp_collapse_je():
    """je is the symmetric inverse of jne — also safe."""
    asm = (
        "_f:\n"
        "        push    eax\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     ecx, [ebp - 8]\n"
        "        mov     eax, [eax + ecx*4]\n"
        "        mov     ecx, eax\n"
        "        pop     eax\n"
        "        cmp     eax, ecx\n"
        "        je      .L_eq\n"
        "        mov     eax, 0\n"
        "        ret\n"
        ".L_eq:\n"
        "        mov     eax, 1\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("transfer_pop_cmp_collapse") == 1


def test_transfer_pop_cmp_collapse_skips_jl():
    """jl reads SF and OF — flips when operand order swaps; skip."""
    asm = (
        "_f:\n"
        "        push    eax\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     ecx, [ebp - 8]\n"
        "        mov     eax, [eax + ecx*4]\n"
        "        mov     ecx, eax\n"
        "        pop     eax\n"
        "        cmp     eax, ecx\n"
        "        jl      .L_lt\n"
        "        ret\n"
        ".L_lt:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("transfer_pop_cmp_collapse", 0) == 0


def test_transfer_pop_cmp_collapse_skips_jb():
    """jb reads CF — flips when operand order swaps; skip."""
    asm = (
        "_f:\n"
        "        push    eax\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     ecx, [ebp - 8]\n"
        "        mov     eax, [eax + ecx*4]\n"
        "        mov     ecx, eax\n"
        "        pop     eax\n"
        "        cmp     eax, ecx\n"
        "        jb      .L_lt\n"
        "        ret\n"
        ".L_lt:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("transfer_pop_cmp_collapse", 0) == 0


def test_transfer_pop_cmp_collapse_setne():
    """setne reads only ZF — safe."""
    asm = (
        "_f:\n"
        "        push    eax\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     ecx, [ebp - 8]\n"
        "        mov     eax, [eax + ecx*4]\n"
        "        mov     ecx, eax\n"
        "        pop     eax\n"
        "        cmp     eax, ecx\n"
        "        setne   al\n"
        "        movzx   eax, al\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("transfer_pop_cmp_collapse") == 1


def test_transfer_pop_cmp_collapse_skips_setl():
    """setl reads SF and OF — flip when operand order swaps; skip."""
    asm = (
        "_f:\n"
        "        push    eax\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     ecx, [ebp - 8]\n"
        "        mov     eax, [eax + ecx*4]\n"
        "        mov     ecx, eax\n"
        "        pop     eax\n"
        "        cmp     eax, ecx\n"
        "        setl    al\n"
        "        movzx   eax, al\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("transfer_pop_cmp_collapse", 0) == 0


def test_transfer_pop_cmp_collapse_chained_jne_safe():
    """Multiple ZF-only readers between cmp and flag-clobber are safe."""
    asm = (
        "_f:\n"
        "        push    eax\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     ecx, [ebp - 8]\n"
        "        mov     eax, [eax + ecx*4]\n"
        "        mov     ecx, eax\n"
        "        pop     eax\n"
        "        cmp     eax, ecx\n"
        "        jne     .L_ne\n"
        "        sete    al\n"
        "        movzx   eax, al\n"
        "        ret\n"
        ".L_ne:\n"
        "        xor     eax, eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("transfer_pop_cmp_collapse") == 1


def test_transfer_pop_cmp_collapse_skips_mixed_readers():
    """If a non-ZF reader appears before a fence/clobber, skip."""
    asm = (
        "_f:\n"
        "        push    eax\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     ecx, [ebp - 8]\n"
        "        mov     eax, [eax + ecx*4]\n"
        "        mov     ecx, eax\n"
        "        pop     eax\n"
        "        cmp     eax, ecx\n"
        "        jne     .L_skip\n"
        "        jl      .L_lt\n"
        ".L_skip:\n"
        "        ret\n"
        ".L_lt:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("transfer_pop_cmp_collapse", 0) == 0


# ── label_load_collapse ──────────────────────────────────────────


def test_label_load_collapse_basic():
    """`mov eax, _label; mov ecx, [eax]` → `mov ecx, [_label]`
    when EAX is dead after.

    Cascades with value_forward_to_reg: the trailing `mov eax, ecx`
    gets folded into the load (`mov ecx, [_glob]; mov eax, ecx` →
    `mov eax, [_glob]`), so the final form drops both intermediates.
    """
    asm = (
        "_f:\n"
        "        mov     eax, _glob\n"
        "        mov     ecx, [eax]\n"
        "        mov     eax, ecx\n"  # eax overwritten (dead before)
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     eax, [_glob]" in out
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
    """The label can be a label-arithmetic expression. Cascades with
    value_forward_to_reg the same as the basic case."""
    asm = (
        "_f:\n"
        "        mov     eax, _b + 8\n"
        "        mov     ecx, [eax]\n"
        "        mov     eax, ecx\n"  # eax overwritten
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     eax, [_b + 8]" in out
    assert opt.stats.get("label_load_collapse") == 1


def test_label_load_collapse_sib():
    """SIB form: `mov eax, _label; mov ecx, [eax + ecx*4]` →
    `mov ecx, [_label + ecx*4]`. Cascades with value_forward_to_reg
    when EAX is overwritten by a register copy from ECX."""
    asm = (
        "_f:\n"
        "        mov     eax, _arr\n"
        "        mov     ecx, [eax + ecx*4]\n"
        "        mov     eax, ecx\n"  # eax overwritten
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # The cascaded form folds the value-forward into the load;
    # the new `mov eax, [_arr + ecx*4]` reads the original (pre-A)
    # ECX as the index, same as the original line.
    assert "mov     eax, [_arr + ecx*4]" in out
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


def test_value_forward_to_reg_handles_memory_source():
    """Memory sources fold the same way: `mov reg1, [m]; mov reg2, reg1`
    becomes `mov reg2, [m]` when reg1 is dead. The new load is the
    same encoded width, so we drop the 2-byte register transfer."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp - 4]\n"
        "        mov     ebx, eax\n"
        "        mov     eax, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("value_forward_to_reg", 0) == 1
    # `mov ebx, eax` folded into `mov ebx, [ebp - 4]`.
    assert "mov     ebx, [ebp - 4]" in out
    # The original `mov eax, [ebp - 4]` is gone.
    assert "mov     eax, [ebp - 4]" not in out


def test_value_forward_to_reg_skips_when_memory_references_dest():
    """If the memory source references the destination register, we
    can't substitute (the new load would self-reference and read a
    different address)."""
    asm = (
        "_f:\n"
        "        mov     eax, [ecx]\n"
        "        mov     ecx, eax\n"
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


# ── sib_const_index_fold ─────────────────────────────────────────


def test_sib_const_index_fold_basic():
    """`mov ecx, 1; mov eax, [eax + ecx*4]` → `mov eax, [eax + 4]`.
    The constant index gets folded into the displacement."""
    asm = (
        "_f:\n"
        "        mov     ecx, 1\n"
        "        mov     eax, [eax + ecx*4]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     eax, [eax + 4]" in out
    assert "mov     ecx, 1" not in out
    assert opt.stats.get("sib_const_index_fold") == 1


def test_sib_const_index_fold_with_displacement():
    """`mov ecx, 2; mov eax, [eax + ecx*4 + 8]` →
    `mov eax, [eax + 16]` (2*4 + 8)."""
    asm = (
        "_f:\n"
        "        mov     ecx, 2\n"
        "        mov     eax, [eax + ecx*4 + 8]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     eax, [eax + 16]" in out
    assert opt.stats.get("sib_const_index_fold") == 1


def test_sib_const_index_fold_zero_index():
    """`mov ecx, 0; mov eax, [eax + ecx*4]` → `mov eax, [eax]`.
    Zero displacement collapses to plain deref."""
    asm = (
        "_f:\n"
        "        mov     ecx, 0\n"
        "        mov     eax, [eax + ecx*4]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     eax, [eax]" in out
    assert opt.stats.get("sib_const_index_fold") == 1


def test_sib_const_index_fold_skips_when_idx_live():
    """If IDX != DST and IDX is read after the load, can't drop
    the const-load."""
    asm = (
        "_f:\n"
        "        mov     ecx, 5\n"
        "        mov     edx, [eax + ecx*4]\n"  # DST=edx, not ecx
        "        mov     [_glob], ecx\n"  # ECX live!
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("sib_const_index_fold", 0) == 0


def test_sib_const_index_fold_skips_when_base_eq_idx():
    """If BASE == IDX, dropping the const-load changes BASE's
    value too — unsafe."""
    asm = (
        "_f:\n"
        "        mov     eax, 3\n"
        "        mov     edx, [eax + eax*4]\n"  # BASE=IDX=eax
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("sib_const_index_fold", 0) == 0


def test_sib_const_index_fold_negative_displacement():
    """`mov ecx, 3; mov eax, [eax + ecx*4 - 8]` →
    `mov eax, [eax + 4]` (3*4 - 8 = 4)."""
    asm = (
        "_f:\n"
        "        mov     ecx, 3\n"
        "        mov     eax, [eax + ecx*4 - 8]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     eax, [eax + 4]" in out


def test_sib_const_index_fold_negative_result_disp():
    """`mov ecx, 1; mov eax, [eax + ecx*4 - 16]` → `[eax - 12]`."""
    asm = (
        "_f:\n"
        "        mov     ecx, 1\n"
        "        mov     eax, [eax + ecx*4 - 16]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     eax, [eax - 12]" in out


# ── push_pop_to_mov ──────────────────────────────────────────────


def test_push_pop_to_mov_label():
    """`push _label; chain; pop ecx` becomes either:
    - `mov ecx, _label; chain; consumer` via push_pop_to_mov
    - `mov edx, _label; chain; consumer-using-edx` via the
      newer push_pop_to_free_reg (which fires first and produces
      tighter code by dropping BOTH push and pop entirely).

    Either way the push and pop are gone; we assert only that
    structural property."""
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
    # Either pass should have fired.
    fired = (
        opt.stats.get("push_pop_to_mov", 0)
        + opt.stats.get("push_pop_to_free_reg", 0)
    )
    assert fired >= 1


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


def test_push_pop_to_mov_memory_param_save():
    """`push [ebp + N]` (function param) saved across a chain that
    doesn't touch the slot — drops to `mov reg, [ebp + N]`. Saves
    1 byte. Common in struct-copy retptr-save patterns."""
    asm = (
        "_f:\n"
        "        push    dword [ebp + 8]\n"
        "        lea     edx, [ebp - 12]\n"
        "        pop     ecx\n"
        "        mov     eax, [edx + 0]\n"
        "        mov     [ecx + 0], eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "push    dword [ebp + 8]" not in out
    # The pop got replaced with `mov ecx, [ebp + 8]`.
    assert "mov     ecx, [ebp + 8]" in out
    assert opt.stats.get("push_pop_to_mov") == 1


def test_push_pop_to_mov_memory_skips_alias_write():
    """If the chain writes to memory aliasing X, bail."""
    asm = (
        "_f:\n"
        "        push    dword [ebp + 8]\n"
        "        mov     [ebp + 8], 99\n"  # writes X!
        "        pop     ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("push_pop_to_mov", 0) == 0


def test_push_pop_to_mov_memory_disjoint_writes_ok():
    """Disjoint ebp-relative writes don't alias — collapse fires."""
    asm = (
        "_f:\n"
        "        push    dword [ebp + 8]\n"
        "        mov     dword [ebp - 4], 99\n"  # disjoint
        "        pop     ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("push_pop_to_mov") == 1
    assert "mov     ecx, [ebp + 8]" in out


def test_push_pop_to_mov_skips_sib_form():
    """`push [reg + idx*scale]` references registers that the chain
    may modify — bail. ebp-relative literal offsets only.

    Regression: c-testsuite 00015 had `push [eax + ecx*4]` followed
    by `mov eax, [ebp - 4]` which clobbers eax — the rewrite would
    have read from the wrong address.
    """
    asm = (
        "_f:\n"
        "        push    dword [eax + ecx*4]\n"
        "        mov     eax, [ebp - 4]\n"  # clobbers eax!
        "        pop     ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("push_pop_to_mov", 0) == 0


def test_push_pop_to_mov_skips_label_memory():
    """`push [_glob]` (label-addressed memory) — bail since the
    chain may write to globals through pointers."""
    asm = (
        "_f:\n"
        "        push    dword [_glob]\n"
        "        mov     ecx, [ebp - 4]\n"
        "        pop     edx\n"
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


# ── dup_push_pop_self_op ─────────────────────────────────────────


def test_dup_push_pop_self_op_imul():
    """The `arr[i] * arr[i]` pattern: codegen pushes a copy of the
    left operand (memory) then reloads from the same memory for the
    right operand. Drops the push/pop pair and rewrites `imul eax,
    ecx` to `imul eax, eax`.
    """
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     ecx, [ebp + 12]\n"
        "        push    dword [eax + ecx*4]\n"
        "        mov     eax, [eax + ecx*4]\n"
        "        pop     ecx\n"
        "        imul    eax, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "push    dword [eax + ecx*4]" not in out
    assert "pop     ecx\n" not in out
    assert "imul    eax, eax" in out
    assert opt.stats.get("dup_push_pop_self_op") == 1


def test_dup_push_pop_self_op_add():
    """add is commutative — also collapses."""
    asm = (
        "_f:\n"
        "        push    dword [ebp - 4]\n"
        "        mov     eax, [ebp - 4]\n"
        "        pop     ecx\n"
        "        add     eax, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "add     eax, eax" in out
    assert opt.stats.get("dup_push_pop_self_op") == 1


def test_dup_push_pop_self_op_skips_sub():
    """sub is NOT commutative — must not collapse."""
    asm = (
        "_f:\n"
        "        push    dword [ebp - 4]\n"
        "        mov     eax, [ebp - 4]\n"
        "        pop     ecx\n"
        "        sub     eax, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("dup_push_pop_self_op", 0) == 0


def test_dup_push_pop_self_op_skips_different_x():
    """When push X1 and mov X2 differ, no rewrite."""
    asm = (
        "_f:\n"
        "        push    dword [ebp - 4]\n"
        "        mov     eax, [ebp - 8]\n"  # different memory!
        "        pop     ecx\n"
        "        imul    eax, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("dup_push_pop_self_op", 0) == 0


def test_dup_push_pop_self_op_skips_esp_dependent_x():
    """X must not reference esp — push/pop modify esp, so the mov's
    read after the push would land at a different byte than what
    push wrote (esp shifted by 4)."""
    asm = (
        "_f:\n"
        "        push    dword [esp + 4]\n"
        "        mov     eax, [esp + 4]\n"
        "        pop     ecx\n"
        "        imul    eax, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("dup_push_pop_self_op", 0) == 0


def test_dup_push_pop_self_op_skips_when_reg2_live():
    """reg2 (the popped reg) holds X after the OP in the original;
    in the rewrite reg2 is unchanged. If reg2 is live after, the
    rewrite is unsafe."""
    asm = (
        "_f:\n"
        "        push    dword [ebp - 4]\n"
        "        mov     eax, [ebp - 4]\n"
        "        pop     ecx\n"
        "        imul    eax, ecx\n"
        "        mov     edx, ecx\n"  # ECX is read here!
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("dup_push_pop_self_op", 0) == 0


def test_dup_push_pop_self_op_skips_pop_into_eax():
    """If the pop reg matches the mov dest reg (both eax), the
    pattern is different — pop overwrites the loaded value."""
    asm = (
        "_f:\n"
        "        push    dword [ebp - 4]\n"
        "        mov     eax, [ebp - 4]\n"
        "        pop     eax\n"  # would overwrite eax!
        "        imul    eax, eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("dup_push_pop_self_op", 0) == 0


def test_dup_push_pop_self_op_register_form():
    """When push pushes a register and the same register is loaded
    via a mem ref (different shape) — should NOT match because push
    operand is a reg name, mov src is `[mem]` — different text."""
    asm = (
        "_f:\n"
        "        push    eax\n"  # push reg
        "        mov     eax, [ebp - 4]\n"
        "        pop     ecx\n"
        "        add     eax, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    # push eax != mov eax, [ebp-4] — different operands.
    assert opt.stats.get("dup_push_pop_self_op", 0) == 0


# ── push_pop_op_to_memop ─────────────────────────────────────────


def test_push_pop_op_to_memop_factorial_imul():
    """The recursive `n * factorial(n - 1)` shape: codegen pushes n
    (an ebp-relative param) before the recursive call, then pops it
    back into ECX after the call. Drops the push/pop and uses memory
    operand directly: `imul eax, dword [ebp + 8]`.

    Note: `add_esp_to_pop` runs before this pass and rewrites
    `add esp, 4` (the call arg cleanup) to `pop ecx`. So the chain
    after that pass has TWO `pop ecx` lines: one for cleanup, one
    for restoring n. push_pop_op_to_memop folds the OUTER push and
    its matching pop, leaving the cleanup pop in place.
    """
    asm = (
        "_factorial:\n"
        "        push    ebp\n"
        "        mov     ebp, esp\n"
        "        push    dword [ebp + 8]\n"
        "        mov     eax, [ebp + 8]\n"
        "        dec     eax\n"
        "        push    eax\n"
        "        call    _factorial\n"
        "        add     esp, 4\n"
        "        pop     ecx\n"
        "        imul    eax, ecx\n"
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # The original `push dword [ebp + 8]` and its matching pop are
    # eliminated; the call cleanup `pop ecx` remains (from
    # add_esp_to_pop).
    assert "push    dword [ebp + 8]" not in out
    assert "imul    eax, dword [ebp + 8]" in out
    assert opt.stats.get("push_pop_op_to_memop") == 1


def test_push_pop_op_to_memop_skips_when_addr_taken():
    """When the function takes the address of X via `lea reg, [ebp +
    N]`, a call in the chain might mutate X via a captured pointer.
    The codegen's saved-value-vs-current-value distinction matters
    in this case — bail."""
    asm = (
        "_test:\n"
        "        push    ebp\n"
        "        mov     ebp, esp\n"
        "        push    dword [ebp + 8]\n"
        "        lea     eax, [ebp + 8]\n"  # captures &X
        "        push    eax\n"
        "        call    _mut2\n"
        "        add     esp, 4\n"
        "        pop     ecx\n"
        "        add     eax, ecx\n"
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("push_pop_op_to_memop", 0) == 0


def test_push_pop_op_to_memop_skips_chain_writes_x():
    """If the chain writes to X's slot, the rewrite would use the
    new value instead of the saved one. Bail."""
    asm = (
        "_test:\n"
        "        push    ebp\n"
        "        mov     ebp, esp\n"
        "        push    dword [ebp - 4]\n"
        "        mov     eax, 99\n"
        "        mov     [ebp - 4], eax\n"  # writes X!
        "        call    _foo\n"
        "        pop     ecx\n"
        "        add     eax, ecx\n"
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("push_pop_op_to_memop", 0) == 0


def test_push_pop_op_to_memop_skips_register_push():
    """X must be ebp-relative (function param or local). Register
    pushes are not safe (the register's value isn't fixed)."""
    asm = (
        "_test:\n"
        "        push    ebp\n"
        "        mov     ebp, esp\n"
        "        mov     eax, [ebp + 8]\n"
        "        push    eax\n"  # register push, not ebp-relative
        "        call    _foo\n"
        "        pop     ecx\n"
        "        add     eax, ecx\n"
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("push_pop_op_to_memop", 0) == 0


def test_push_pop_op_to_memop_skips_short_chain():
    """The narrow 4-line case (chain length 1 with same X) is handled
    by `_pass_dup_push_pop_self_op` and saves 1 byte more via reg-reg
    form. This pass requires chain length ≥ 2 to avoid duplicating
    optimization opportunities."""
    asm = (
        "_test:\n"
        "        push    ebp\n"
        "        mov     ebp, esp\n"
        "        push    dword [ebp - 4]\n"
        "        mov     eax, [ebp - 4]\n"
        "        pop     ecx\n"
        "        add     eax, ecx\n"
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # The narrow pass fires instead.
    assert opt.stats.get("dup_push_pop_self_op", 0) == 1
    assert opt.stats.get("push_pop_op_to_memop", 0) == 0
    assert "add     eax, eax" in out


def test_push_pop_op_to_memop_skips_sub():
    """sub is NOT commutative — must not collapse."""
    asm = (
        "_test:\n"
        "        push    ebp\n"
        "        mov     ebp, esp\n"
        "        push    dword [ebp + 8]\n"
        "        mov     eax, [ebp + 8]\n"
        "        dec     eax\n"
        "        push    eax\n"
        "        call    _foo\n"
        "        add     esp, 4\n"
        "        pop     ecx\n"
        "        sub     eax, ecx\n"
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("push_pop_op_to_memop", 0) == 0


# ── push_const_index_fold ───────────────────────────────────────


def test_push_const_index_fold_zero():
    """`xor ecx, ecx; push dword [eax + ecx*4]` →
    `push dword [eax]`. Zero displacement collapses to plain deref.
    """
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        xor     ecx, ecx\n"
        "        push    dword [eax + ecx*4]\n"
        "        mov     eax, [eax + 4]\n"
        "        pop     edx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "xor     ecx, ecx" not in out
    assert "push    dword [eax]" in out
    assert opt.stats.get("push_const_index_fold") == 1


def test_push_const_index_fold_with_imm():
    """`mov ecx, 2; push dword [eax + ecx*4]` →
    `push dword [eax + 8]`. Constant index folded."""
    asm = (
        "_f:\n"
        "        mov     ecx, 2\n"
        "        push    dword [eax + ecx*4]\n"
        "        mov     eax, [eax + 4]\n"
        "        pop     edx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "push    dword [eax + 8]" in out
    assert opt.stats.get("push_const_index_fold") == 1


def test_push_const_index_fold_skips_idx_reused():
    """If IDX is read after the push (not dead), bail."""
    asm = (
        "_f:\n"
        "        mov     ecx, 1\n"
        "        push    dword [eax + ecx*4]\n"
        "        mov     edx, ecx\n"  # ecx still read here!
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("push_const_index_fold", 0) == 0


def test_push_const_index_fold_skips_base_eq_idx():
    """If BASE == IDX, the rewrite is unsafe (BASE's value would
    change after dropping the const-load)."""
    asm = (
        "_f:\n"
        "        xor     ecx, ecx\n"
        "        push    dword [ecx + ecx*4]\n"
        "        mov     eax, 0\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("push_const_index_fold", 0) == 0


# ── pop_op_chain_retarget ──────────────────────────────────────


def test_pop_op_chain_retarget_basic():
    """`push eax; chain; pop ecx; add eax, ecx` (commutative tail)
    where chain produces RHS in EAX. Drops push + pop, retargets
    chain to write ECX. Saves 2 bytes."""
    asm = (
        "_f:\n"
        "        push    eax\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     eax, [eax + 8]\n"
        "        pop     ecx\n"
        "        add     eax, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "push    eax\n" not in out
    assert "pop     ecx\n" not in out
    # Chain retargeted to ECX.
    assert "mov     ecx, [ebp + 8]" in out
    assert "mov     ecx, [ecx + 8]" in out
    # OP unchanged (reads EAX as LHS, ECX as RHS).
    assert "add     eax, ecx" in out
    assert opt.stats.get("pop_op_chain_retarget") == 1


def test_pop_op_chain_retarget_skips_sub():
    """sub is NOT commutative — can't retarget."""
    asm = (
        "_f:\n"
        "        push    eax\n"
        "        mov     eax, [ebp + 8]\n"
        "        pop     ecx\n"
        "        sub     eax, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("pop_op_chain_retarget", 0) == 0


def test_pop_op_chain_retarget_skips_self_rmw_first():
    """First chain instr must be a fresh write — `mov eax, [eax]`
    reads eax (= LHS) and would change semantics if retargeted."""
    asm = (
        "_f:\n"
        "        push    eax\n"
        "        mov     eax, [eax]\n"  # self-RMW: reads eax!
        "        pop     ecx\n"
        "        add     eax, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("pop_op_chain_retarget", 0) == 0


# ── index_store_xfer_collapse ────────────────────────────────────


def test_index_store_xfer_collapse_basic():
    """`shl ecx, 2; add eax, ecx; mov ecx, eax; mov eax, [ebp + 16];
    mov [ecx], eax` → full SIB rewrite using EDX as free reg. Saves
    3 instructions when EDX is unused in the function (drops shl,
    add, and the xfer)."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     ecx, [ebp - 4]\n"
        "        shl     ecx, 2\n"
        "        add     eax, ecx\n"
        "        mov     ecx, eax\n"
        "        mov     eax, [ebp + 16]\n"
        "        mov     [ecx], eax\n"
        "        xor     eax, eax\n"  # kill EAX so it's dead after store
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "shl     ecx, 2" not in out
    assert "add     eax, ecx" not in out
    assert "mov     edx, [ebp + 16]" in out
    assert "mov     [eax + ecx*4], edx" in out
    assert opt.stats.get("index_store_sib_full") == 1


def test_index_store_xfer_collapse_simple_when_all_free_busy():
    """When all candidate free regs (EDX, ESI, EDI, EBX) are
    referenced in the function, fall back to the simpler
    xfer-collapse (drop only the xfer, save 1 instruction)."""
    asm = (
        "_f:\n"
        "        cdq\n"             # uses EDX implicitly
        "        lodsb\n"            # uses ESI implicitly
        "        stosb\n"            # uses EDI implicitly
        "        push    ebx\n"      # uses EBX explicitly
        "        pop     ebx\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     ecx, [ebp - 4]\n"
        "        shl     ecx, 2\n"
        "        add     eax, ecx\n"
        "        mov     ecx, eax\n"
        "        mov     eax, [ebp + 16]\n"
        "        mov     [ecx], eax\n"
        "        xor     eax, eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # Simple xfer-collapse fired
    assert "mov     ecx, [ebp + 16]" in out
    assert "mov     [eax], ecx" in out
    assert opt.stats.get("index_store_xfer_collapse") == 1
    assert opt.stats.get("index_store_sib_full", 0) == 0


def test_index_store_xfer_collapse_skips_when_src_uses_xfer():
    """If SRC references the xfer reg, the rewrite is unsafe — XFER's
    value differs at the SRC-read site after dropping the xfer."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     ecx, [ebp - 4]\n"
        "        shl     ecx, 2\n"
        "        add     eax, ecx\n"
        "        mov     ecx, eax\n"
        "        mov     eax, [ecx]\n"  # SRC refs ECX (= xfer)
        "        mov     [ecx], eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("index_store_xfer_collapse", 0) == 0


def test_index_store_xfer_collapse_skips_when_xfer_live():
    """If XFER is live after the store, dropping the xfer would
    change behavior — skip the rewrite. (Here we pretend XFER is
    used in the next basic block by adding a `mov reg, ecx` after.)"""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     ecx, [ebp - 4]\n"
        "        shl     ecx, 2\n"
        "        add     eax, ecx\n"
        "        mov     edx, eax\n"  # XFER = EDX
        "        mov     eax, [ebp + 16]\n"
        "        mov     [edx], eax\n"
        "        mov     esi, edx\n"  # uses EDX after store — live
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("index_store_xfer_collapse", 0) == 0


def test_index_store_xfer_collapse_size_prefix_dword():
    """Size prefix `dword [...]` flows through to the SIB form."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     ecx, [ebp - 4]\n"
        "        shl     ecx, 2\n"
        "        add     eax, ecx\n"
        "        mov     ecx, eax\n"
        "        mov     eax, [ebp + 16]\n"
        "        mov     dword [ecx], eax\n"
        "        xor     eax, eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # SIB rewrite preserves the size prefix.
    assert "mov     dword [eax + ecx*4], edx" in out
    assert opt.stats.get("index_store_sib_full") == 1


# ── index_load_collapse_label ─────────────────────────────────


def test_index_load_collapse_label_basic():
    """`shl ecx, 1; add ecx, _g; movzx ecx, word [ecx]` →
    `movzx ecx, word [_g + ecx*2]`. Drops shl + add (~8 bytes)."""
    asm = (
        "_f:\n"
        "        mov     ecx, [ebp - 8]\n"
        "        shl     ecx, 1\n"
        "        add     ecx, _g\n"
        "        movzx   ecx, word [ecx]\n"
        "        mov     eax, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "shl     ecx, 1" not in out
    assert "add     ecx, _g" not in out
    assert "movzx   ecx, word [_g + ecx*2]" in out
    assert opt.stats.get("index_load_collapse_label") == 1


def test_index_load_collapse_label_int_array():
    """Scale 4 case for int globals. Cascades with value_forward_to_reg
    when the loaded ECX is forwarded to EAX."""
    asm = (
        "_f:\n"
        "        mov     ecx, [ebp - 8]\n"
        "        shl     ecx, 2\n"
        "        add     ecx, _g\n"
        "        mov     ecx, [ecx]\n"
        "        mov     eax, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # After cascade: `mov ecx, [ebp - 8]; mov eax, [_g + ecx*4]; ret`.
    # The `mov ecx, [_g + ecx*4]; mov eax, ecx` pair folds via
    # value_forward_to_reg.
    assert "mov     eax, [_g + ecx*4]" in out
    assert opt.stats.get("index_load_collapse_label") == 1


def test_index_load_collapse_label_skips_register_base():
    """If the second `add` operand is a register (not label), this
    is the standard `index_load_collapse` pattern, not ours. Bail."""
    asm = (
        "_f:\n"
        "        shl     ecx, 1\n"
        "        add     ecx, eax\n"  # base is a register
        "        movzx   ecx, word [ecx]\n"
        "        mov     eax, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    # The standard index_load_collapse may fire instead — we just
    # check that OUR new pass didn't fire.
    assert opt.stats.get("index_load_collapse_label", 0) == 0


def test_index_load_collapse_label_skips_numeric_disp():
    """If the second `add` operand is a numeric literal, this is
    `disp_load_collapse` territory, not ours. Bail."""
    asm = (
        "_f:\n"
        "        shl     ecx, 1\n"
        "        add     ecx, 8\n"  # numeric, not label
        "        movzx   ecx, word [ecx]\n"
        "        mov     eax, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("index_load_collapse_label", 0) == 0


def test_index_load_collapse_label_skips_idx_reused():
    """If IDX is read after the load (and DST != IDX), the rewrite
    leaves IDX with its pre-shl value (= original idx) instead of
    the original code's post-add value (= label + idx*scale). Bail."""
    asm = (
        "_f:\n"
        "        shl     ecx, 1\n"
        "        add     ecx, _g\n"
        "        mov     eax, [ecx]\n"  # DST = eax, IDX = ecx
        "        mov     ebx, ecx\n"  # ECX read after!
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("index_load_collapse_label", 0) == 0


def test_mov_label_shl_add_load_to_sib_basic():
    """`mov edx, _g; shl eax, 2; add eax, edx; mov eax, [eax]`
    → `mov eax, [_g + eax*4]`. Drops 3 instructions / ~10 bytes.
    Common shape: for-loop body where index is in EAX and codegen
    materializes the global base into EDX as a scratch."""
    asm = (
        "_f:\n"
        "        mov     edx, _g\n"
        "        shl     eax, 2\n"
        "        add     eax, edx\n"
        "        mov     eax, [eax]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     edx, _g" not in out
    assert "shl     eax, 2" not in out
    assert "add     eax, edx" not in out
    assert "mov     eax, [_g + eax*4]" in out
    assert opt.stats.get("mov_label_shl_add_load_to_sib") == 1


def test_mov_label_shl_add_load_to_sib_distinct_dst():
    """DST != IDX: `mov edx, _g; shl eax, 1; add eax, edx; movzx
    ecx, word [eax]` → `movzx ecx, word [_g + eax*2]`. Requires IDX
    dead after (eax not read past the load)."""
    asm = (
        "_f:\n"
        "        mov     edx, _g\n"
        "        shl     eax, 1\n"
        "        add     eax, edx\n"
        "        movzx   ecx, word [eax]\n"
        "        mov     eax, ecx\n"  # eax overwritten before any read
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "movzx   ecx, word [_g + eax*2]" in out
    assert opt.stats.get("mov_label_shl_add_load_to_sib") == 1


def test_mov_label_shl_add_load_to_sib_skips_idx_alive():
    """If IDX is alive after the load and DST != IDX, the rewrite
    leaves IDX with the unscaled index value (instead of the
    original post-add LABEL + idx*scale). Must not fire."""
    asm = (
        "_f:\n"
        "        mov     edx, _g\n"
        "        shl     eax, 1\n"
        "        add     eax, edx\n"
        "        movzx   ecx, word [eax]\n"
        "        mov     [ebp - 4], eax\n"  # eax read after — alive
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("mov_label_shl_add_load_to_sib", 0) == 0


def test_mov_label_shl_add_load_to_sib_skips_base_alive():
    """If BASE is alive after (read before being overwritten), the
    rewrite would observe a different value (pre-A vs LABEL). Must
    not fire."""
    asm = (
        "_f:\n"
        "        mov     edx, _g\n"
        "        shl     eax, 2\n"
        "        add     eax, edx\n"
        "        mov     eax, [eax]\n"
        "        mov     [ebp - 4], edx\n"  # edx read after
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("mov_label_shl_add_load_to_sib", 0) == 0


def test_mov_label_shl_add_load_to_sib_skips_numeric_label():
    """A's source must be a label/symbolic expression, not a numeric
    literal (that's plain `disp_load_collapse` territory)."""
    asm = (
        "_f:\n"
        "        mov     edx, 100\n"  # numeric, not label
        "        shl     eax, 2\n"
        "        add     eax, edx\n"
        "        mov     eax, [eax]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("mov_label_shl_add_load_to_sib", 0) == 0


def test_mov_label_shl_add_load_to_sib_skips_invalid_scale():
    """Shift must be 1, 2, or 3 (scale 2/4/8). Other counts bail."""
    asm = (
        "_f:\n"
        "        mov     edx, _g\n"
        "        shl     eax, 5\n"  # scale 32 not supported
        "        add     eax, edx\n"
        "        mov     eax, [eax]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("mov_label_shl_add_load_to_sib", 0) == 0


def test_mov_label_shl_add_load_to_sib_skips_same_reg():
    """BASE != IDX. If same, the pattern can't make sense (the shl
    would already need IDX = LABEL, which is wrong for an index)."""
    asm = (
        "_f:\n"
        "        mov     eax, _g\n"
        "        shl     eax, 2\n"  # base == idx == eax
        "        add     eax, eax\n"
        "        mov     eax, [eax]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("mov_label_shl_add_load_to_sib", 0) == 0


def test_mov_label_shl_add_load_to_sib_with_disp_8():
    """`pts[i].member` for a struct array. The deref has a positive
    member-offset displacement: `mov eax, [eax + 4]`. Pass folds the
    displacement into the SIB form: `mov eax, [_pts + eax*8 + 4]`."""
    asm = (
        "_f:\n"
        "        mov     edx, _pts\n"
        "        shl     eax, 3\n"
        "        add     eax, edx\n"
        "        mov     eax, [eax + 4]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     edx, _pts" not in out
    assert "shl     eax, 3" not in out
    assert "add     eax, edx" not in out
    assert "mov     eax, [_pts + eax*8 + 4]" in out
    assert opt.stats.get("mov_label_shl_add_load_to_sib") == 1


def test_mov_label_shl_add_load_to_sib_with_disp_negative():
    """Negative displacement on the deref folds into the SIB form."""
    asm = (
        "_f:\n"
        "        mov     edx, _arr\n"
        "        shl     eax, 2\n"
        "        add     eax, edx\n"
        "        mov     eax, [eax - 4]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     eax, [_arr + eax*4 - 4]" in out
    assert opt.stats.get("mov_label_shl_add_load_to_sib") == 1


def test_mov_label_shl_add_load_to_sib_with_disp_hex():
    """Hex displacement also folds."""
    asm = (
        "_f:\n"
        "        mov     edx, _arr\n"
        "        shl     eax, 2\n"
        "        add     eax, edx\n"
        "        mov     eax, [eax + 0x10]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     eax, [_arr + eax*4 + 0x10]" in out
    assert opt.stats.get("mov_label_shl_add_load_to_sib") == 1


def test_mov_label_shl_add_load_to_sib_loop_body():
    """Realistic loop body: cmp at top with EAX = i, body derefs
    `g[i]`. EDX is dead at function exit; .L1_for_top is a back-edge
    target. Pass should fire and the loop body shrinks dramatically."""
    asm = (
        "_f:\n"
        "        enter   8, 0\n"
        ".L1_for_top:\n"
        "        mov     eax, [ebp - 8]\n"
        "        cmp     eax, [ebp + 8]\n"
        "        jge     .L3_for_end\n"
        "        mov     edx, _g\n"
        "        shl     eax, 2\n"
        "        add     eax, edx\n"
        "        mov     eax, [eax]\n"
        "        add     [ebp - 4], eax\n"
        "        inc     dword [ebp - 8]\n"
        "        jmp     .L1_for_top\n"
        ".L3_for_end:\n"
        "        mov     eax, [ebp - 4]\n"
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     eax, [_g + eax*4]" in out
    assert opt.stats.get("mov_label_shl_add_load_to_sib") == 1


def test_dead_unused_slot_stores_basic():
    """Function with `enter` and a slot that's never read or
    address-taken — all stores to that slot are dead."""
    asm = (
        "_f:\n"
        "        enter   4, 0\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     [ebp - 4], eax\n"  # dead store
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     [ebp - 4], eax" not in out
    assert opt.stats.get("dead_unused_slot_stores") == 1


def test_dead_unused_slot_stores_skips_no_prologue():
    """Without a prologue (synthetic test fragments), my pass
    doesn't fire — too risky to assume the test asm is a complete
    function."""
    asm = (
        "_f:\n"
        "        mov     [ebp - 4], eax\n"  # would be dead, but no prologue
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("dead_unused_slot_stores", 0) == 0


def test_dead_unused_slot_stores_skips_when_slot_read():
    """If the slot is read anywhere in the function, all stores
    to it are alive. Use `add eax, [ebp - 4]` as the read since
    that doesn't get optimized away by other passes."""
    asm = (
        "_f:\n"
        "        enter   4, 0\n"
        "        mov     [ebp - 4], eax\n"
        "        mov     eax, 100\n"
        "        add     eax, [ebp - 4]\n"  # read of slot
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     [ebp - 4], eax" in out
    assert opt.stats.get("dead_unused_slot_stores", 0) == 0


def test_dead_unused_slot_stores_skips_when_address_taken():
    """If a `lea reg, [ebp - N]` exists in the function, the slot's
    address might be passed elsewhere; can't drop stores."""
    asm = (
        "_f:\n"
        "        enter   4, 0\n"
        "        mov     [ebp - 4], eax\n"
        "        lea     ecx, [ebp - 4]\n"  # address-take
        "        push    ecx\n"
        "        call    _helper\n"
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     [ebp - 4], eax" in out
    assert opt.stats.get("dead_unused_slot_stores", 0) == 0


def test_dead_unused_slot_stores_skips_rmw():
    """RMW ops on the slot count as reads (in the read part of
    read-modify-write), so the slot is alive."""
    asm = (
        "_f:\n"
        "        enter   4, 0\n"
        "        mov     [ebp - 4], eax\n"
        "        add     dword [ebp - 4], 1\n"  # RMW
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     [ebp - 4], eax" in out
    assert opt.stats.get("dead_unused_slot_stores", 0) == 0


def test_dead_unused_slot_stores_drops_multiple_stores():
    """All stores to the unused slot are dropped. The earlier
    `dead_stack_store` pass drops stores immediately overwritten
    by another store; my pass picks up any remaining ones."""
    asm = (
        "_f:\n"
        "        enter   4, 0\n"
        "        mov     [ebp - 4], eax\n"  # store 1 — dead
        "        add     eax, 5\n"
        "        mov     [ebp - 4], eax\n"  # store 2 — dead
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     [ebp - 4], eax" not in out
    # Combined: dead_stack_store drops the first; my pass drops
    # the second (now no further overwrite, but slot never read).
    total_drops = (
        opt.stats.get("dead_stack_store", 0)
        + opt.stats.get("dead_unused_slot_stores", 0)
    )
    assert total_drops == 2


def test_dead_unused_slot_stores_independent_slots():
    """Slot N1 with reads is alive; slot N2 without reads is dead.
    Use `add edx, [ebp - 4]` as the read so other passes don't
    optimize it away."""
    asm = (
        "_f:\n"
        "        enter   8, 0\n"
        "        mov     [ebp - 4], eax\n"  # alive (read below)
        "        mov     [ebp - 8], ecx\n"  # dead
        "        mov     edx, 100\n"
        "        add     edx, [ebp - 4]\n"  # read of -4
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     [ebp - 4], eax" in out  # alive
    assert "mov     [ebp - 8], ecx" not in out  # dead
    assert opt.stats.get("dead_unused_slot_stores") == 1


def test_dead_unused_slot_stores_skips_sib_ebp_access():
    """A SIB-form access through ebp like `[ebp + ecx*4 - 32]` reads
    a slot at runtime-variable offset. The pass can't analyze which
    slot is read, so it must bail on the entire function. Without
    this, array initializers (`int a[8] = {0..7}`) get dropped
    when the array is read via SIB form in a loop."""
    asm = (
        "_f:\n"
        "        enter   40, 0\n"
        "        mov     dword [ebp - 32], 0\n"  # a[0]
        "        mov     dword [ebp - 28], 1\n"  # a[1]
        "        mov     dword [ebp - 24], 2\n"
        "        mov     dword [ebp - 20], 3\n"
        "        mov     dword [ebp - 16], 4\n"
        "        mov     dword [ebp - 12], 5\n"
        "        mov     dword [ebp - 8], 6\n"
        "        mov     dword [ebp - 4], 7\n"  # a[7]
        "        xor     eax, eax\n"
        "        mov     ecx, 0\n"
        ".L_top:\n"
        "        cmp     ecx, 8\n"
        "        jge     .L_end\n"
        "        add     eax, [ebp + ecx*4 - 32]\n"  # SIB-form read
        "        inc     ecx\n"
        "        jmp     .L_top\n"
        ".L_end:\n"
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # All array stores must be preserved (even though
    # imm_store_collapse may rewrite `mov dword [ebp - 32], 0`
    # to `mov [ebp - 32], eax` after `xor eax, eax`).
    assert "[ebp - 32]" in out
    assert "mov     dword [ebp - 28], 1" in out
    assert "mov     dword [ebp - 4], 7" in out
    assert opt.stats.get("dead_unused_slot_stores", 0) == 0


def test_dead_unused_slot_stores_skips_sib_ebp_no_disp():
    """SIB-form `[ebp + ecx*4]` (no displacement) also bails."""
    asm = (
        "_f:\n"
        "        enter   8, 0\n"
        "        mov     [ebp - 8], eax\n"  # would be dead w/o SIB
        "        mov     [ebp - 4], eax\n"
        "        mov     edx, [ebp + ecx*4]\n"  # SIB read
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     [ebp - 8], eax" in out
    assert "mov     [ebp - 4], eax" in out
    assert opt.stats.get("dead_unused_slot_stores", 0) == 0


def test_dup_load_to_copy_basic():
    """`mov esi, [m]; mov eax, [m]` → `mov esi, [m]; mov eax, esi`.
    Saves 5 bytes for SIB-form memory (7-byte load → 2-byte copy)."""
    asm = (
        "_f:\n"
        "        mov     esi, [_g + eax*4]\n"
        "        mov     eax, [_g + eax*4]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     esi, [_g + eax*4]" in out
    assert "mov     eax, esi" in out
    # Only one SIB-form load remains.
    assert out.count("[_g + eax*4]") == 1
    assert opt.stats.get("dup_load_to_copy") == 1


def test_dup_load_to_copy_ebp_relative():
    """ebp-relative source: saves 1-2 bytes."""
    asm = (
        "_f:\n"
        "        mov     ebx, [ebp - 4]\n"
        "        mov     ecx, [ebp - 4]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     ebx, [ebp - 4]" in out
    assert "mov     ecx, ebx" in out
    assert opt.stats.get("dup_load_to_copy") == 1


def test_dup_load_to_copy_skips_when_addr_refs_r1():
    """If the memory address references R1, A modifies the meaning
    of the address for B."""
    asm = (
        "_f:\n"
        "        mov     eax, [eax + 4]\n"  # writes EAX
        "        mov     ecx, [eax + 4]\n"  # different address!
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("dup_load_to_copy", 0) == 0


def test_dup_load_to_copy_skips_when_memory_differs():
    """Different memory operands are not a dup."""
    asm = (
        "_f:\n"
        "        mov     ebx, [ebp - 4]\n"
        "        mov     ecx, [ebp - 8]\n"  # different memory
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("dup_load_to_copy", 0) == 0


def test_dup_load_to_copy_skips_when_same_reg():
    """R1 == R2 would be a self-mov; pass skips."""
    asm = (
        "_f:\n"
        "        mov     ebx, [ebp - 4]\n"
        "        mov     ebx, [ebp - 4]\n"  # same reg
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("dup_load_to_copy", 0) == 0


def test_dup_load_to_copy_with_size_prefix():
    """Size-prefixed memory operands (different sizes) are different."""
    asm = (
        "_f:\n"
        "        mov     ebx, dword [ebp - 4]\n"
        "        mov     ecx, dword [ebp - 4]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # Both have size prefix; same memory after stripping.
    assert "mov     ecx, ebx" in out
    assert opt.stats.get("dup_load_to_copy") == 1


def test_lea_sib_label_load_collapse_cmp():
    """`lea eax, [_g + eax*4]; cmp dword [eax], edx` collapses to
    `cmp dword [_g + eax*4], edx`. Drops the 6-byte+ LEA. Common in
    `if (g[i] == target)` patterns."""
    asm = (
        "_f:\n"
        "        lea     eax, [_g + eax*4]\n"
        "        cmp     dword [eax], edx\n"
        "        je      .L_found\n"
        "        ret\n"
        ".L_found:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "lea     eax, [_g + eax*4]" not in out
    assert "cmp     dword [_g + eax*4], edx" in out
    assert opt.stats.get("lea_sib_label_load_collapse") == 1


def test_lea_sib_label_load_collapse_load():
    """`lea eax, [_g + eax*4]; mov esi, [eax]` collapses. DST != REG
    means we need REG dead after the load."""
    asm = (
        "_f:\n"
        "        lea     eax, [_g + eax*4]\n"
        "        mov     esi, [eax]\n"
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # EAX dead at ret (treat_as_scratch=True), so collapse fires.
    assert "lea     eax, [_g + eax*4]" not in out
    assert "mov     esi, [_g + eax*4]" in out
    assert opt.stats.get("lea_sib_label_load_collapse") == 1


def test_lea_sib_label_load_collapse_load_dst_eq_reg():
    """`lea eax, [_g + eax*4]; mov eax, [eax]` collapses. DST == REG
    means the load overwrites REG, so we don't need REG dead after."""
    asm = (
        "_f:\n"
        "        lea     eax, [_g + eax*4]\n"
        "        mov     eax, [eax]\n"
        "        mov     [ebp - 4], eax\n"  # eax read after — alive!
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # DST == REG, so REG-alive doesn't matter — the load itself
    # overwrites REG.
    assert "lea     eax, [_g + eax*4]" not in out
    assert "mov     eax, [_g + eax*4]" in out
    assert opt.stats.get("lea_sib_label_load_collapse") == 1


def test_lea_sib_label_load_collapse_with_disp():
    """`lea eax, [_g + eax*4 + 4]; mov eax, [eax]` folds the
    displacement into the SIB form."""
    asm = (
        "_f:\n"
        "        lea     eax, [_g + eax*4 + 4]\n"
        "        mov     eax, [eax]\n"
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     eax, [_g + eax*4 + 4]" in out
    assert opt.stats.get("lea_sib_label_load_collapse") == 1


def test_lea_sib_label_load_collapse_disp_combine():
    """Both A and B have displacements — they combine."""
    asm = (
        "_f:\n"
        "        lea     eax, [_g + eax*4 + 8]\n"
        "        mov     eax, [eax + 4]\n"
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # Combined disp = 8 + 4 = 12.
    assert "mov     eax, [_g + eax*4 + 12]" in out


def test_lea_sib_label_load_collapse_skips_non_sib_form():
    """If A's source doesn't match `[LABEL + REG*N]`, bail."""
    asm = (
        "_f:\n"
        "        lea     eax, [ebp - 8]\n"  # not LABEL+REG*N
        "        mov     ebx, [eax]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("lea_sib_label_load_collapse", 0) == 0


def test_lea_sib_label_load_collapse_skips_invalid_scale():
    """Scale must be in {1, 2, 4, 8} (SIB scale)."""
    asm = (
        "_f:\n"
        "        lea     eax, [_g + eax*16]\n"  # invalid scale
        "        mov     ebx, [eax]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("lea_sib_label_load_collapse", 0) == 0


def test_push_memory_to_push_reg_basic():
    """`mov eax, [m]; cmp eax, X; jge L; push dword [m]` collapses
    push's memory operand to `push eax`. Saves 2 bytes per match.

    Common shape: for-loop body where the cmp at the top loaded
    the loop counter into EAX, then a function-call arg push
    re-reads the same memory."""
    asm = (
        "_f:\n"
        ".L1_top:\n"
        "        mov     eax, [ebp - 8]\n"
        "        cmp     eax, [ebp + 8]\n"
        "        jge     .L_end\n"
        "        push    dword [ebp - 8]\n"
        "        call    _helper\n"
        "        pop     ecx\n"
        "        jmp     .L1_top\n"
        ".L_end:\n"
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "push    dword [ebp - 8]" not in out
    assert "push    eax" in out
    assert opt.stats.get("push_memory_to_push_reg", 0) >= 1


def test_push_memory_to_push_reg_skips_when_reg_modified():
    """If REG is overwritten between the load and the push, my
    pass should NOT fire."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp - 8]\n"
        "        mov     eax, 42\n"  # eax overwritten
        "        push    dword [ebp - 8]\n"
        "        call    _helper\n"
        "        pop     ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("push_memory_to_push_reg", 0) == 0


def test_push_memory_to_push_reg_skips_after_label():
    """Labels are control-flow boundaries; cross-label tracking is
    invalidated."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp - 8]\n"
        ".L_mid:\n"  # label invalidates tracking
        "        push    dword [ebp - 8]\n"
        "        call    _helper\n"
        "        pop     ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("push_memory_to_push_reg", 0) == 0


def test_push_memory_to_push_reg_skips_after_call():
    """Calls clobber caller-saved regs."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp - 8]\n"
        "        call    _helper\n"  # clobbers eax
        "        push    dword [ebp - 8]\n"
        "        call    _other\n"
        "        pop     ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("push_memory_to_push_reg", 0) == 0


def test_push_memory_to_push_reg_skips_when_memory_written():
    """If the source memory is written between the mov and the push,
    the cached register's value is stale."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp - 8]\n"
        "        mov     [ebp - 8], 42\n"  # memory write
        "        push    dword [ebp - 8]\n"
        "        call    _helper\n"
        "        pop     ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("push_memory_to_push_reg", 0) == 0


def test_push_memory_to_push_reg_label_memory():
    """Pass also fires for global-label memory `[_glob]`. Saves
    more bytes (5+ → 1 byte per push)."""
    asm = (
        "_f:\n"
        "        mov     eax, [_glob]\n"
        "        cmp     eax, 0\n"
        "        push    dword [_glob]\n"
        "        call    _helper\n"
        "        pop     ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "push    dword [_glob]" not in out
    assert "push    eax" in out
    assert opt.stats.get("push_memory_to_push_reg", 0) == 1


def test_xfer_load_store_collapse_basic():
    """`mov ecx, eax; mov eax, [m]; mov [ecx], eax` collapses to
    `mov ecx, [m]; mov [eax], ecx`. Drops the xfer; saves 1 instr.

    Common shape: codegen pattern after computing an address in
    EAX — xfer to ECX, load value into EAX, store via ECX. After
    rewrite: load value into ECX directly, store via EAX.

    Use an address-arithmetic preamble (add) so value_forward_to_reg
    can't pre-collapse the `mov ecx, eax` — the existing helpers
    only look at simple mov-then-mov patterns."""
    asm = (
        "_f:\n"
        "        add     eax, _g_arr\n"  # address arithmetic
        "        mov     ecx, eax\n"  # xfer
        "        mov     eax, [ebp + 12]\n"  # load value
        "        mov     [ecx], eax\n"  # store via xfer
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # The xfer line is gone.
    assert "mov     ecx, eax" not in out
    # The new sequence has the SRC loaded into ECX directly.
    assert "mov     ecx, [ebp + 12]" in out
    # The store goes through the original BASE register.
    assert "mov     [eax], ecx" in out
    assert opt.stats.get("xfer_load_store_collapse") == 1


def test_xfer_load_store_collapse_skips_when_xfer_alive():
    """If XFER is read after the store, the rewrite would observe
    a different XFER value (= SRC instead of address). Don't fire."""
    asm = (
        "_f:\n"
        "        mov     ecx, eax\n"
        "        mov     eax, [ebp + 12]\n"
        "        mov     [ecx], eax\n"
        "        mov     [ebp - 4], ecx\n"  # ECX read after
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("xfer_load_store_collapse", 0) == 0


def test_xfer_load_store_collapse_skips_when_base_alive():
    """If BASE is read after the store, the rewrite would observe
    a different BASE value (= address instead of SRC). Don't fire."""
    asm = (
        "_f:\n"
        "        mov     ecx, eax\n"
        "        mov     eax, [ebp + 12]\n"
        "        mov     [ecx], eax\n"
        "        mov     [ebp - 4], eax\n"  # EAX read after
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("xfer_load_store_collapse", 0) == 0


def test_xfer_load_store_collapse_skips_when_src_refs_xfer():
    """If SRC references XFER, the rewrite's `mov XFER, SRC` would
    self-reference (changing semantics). Don't fire."""
    asm = (
        "_f:\n"
        "        mov     ecx, eax\n"
        "        mov     eax, [ecx + 12]\n"  # SRC refs XFER (ECX)
        "        mov     [ecx], eax\n"
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("xfer_load_store_collapse", 0) == 0


def test_xfer_load_store_collapse_skips_when_store_addr_has_disp():
    """The store address must be plain `[XFER]` (no disp/SIB). The
    cascade with disp_store_collapse handles the post-rewrite case
    if the original had `add eax, K` before the xfer."""
    asm = (
        "_f:\n"
        "        mov     ecx, eax\n"
        "        mov     eax, [ebp + 12]\n"
        "        mov     [ecx + 4], eax\n"  # disp on store
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    # Currently the pass requires plain `[XFER]`; the disp case is
    # left alone. (Could be extended in the future.)
    assert opt.stats.get("xfer_load_store_collapse", 0) == 0


def test_xfer_load_store_collapse_struct_store_pattern():
    """End-to-end struct-store pattern. Should collapse via
    cascade with chain_label_to_add_operand and xfer_load_store."""
    asm = (
        "_f:\n"
        "        mov     edx, _g_arr\n"
        "        mov     eax, [ebp + 8]\n"
        "        imul    eax, eax, 12\n"
        "        add     eax, edx\n"
        "        mov     ecx, eax\n"
        "        mov     eax, [ebp + 16]\n"
        "        mov     [ecx], eax\n"
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # After cascade: `mov eax, [ebp+8]; imul eax, eax, 12;
    # add eax, _g_arr; mov ecx, [ebp+16]; mov [eax], ecx`.
    assert "mov     edx, _g_arr" not in out
    assert "mov     ecx, eax" not in out  # xfer dropped
    assert "mov     ecx, [ebp + 16]" in out  # SRC into ECX
    assert "mov     [eax], ecx" in out  # store via EAX
    assert opt.stats.get("xfer_load_store_collapse") == 1
    assert opt.stats.get("chain_label_to_add_operand") == 1


def test_chain_label_to_add_operand_basic():
    """`mov edx, _g; mov eax, [ebp + 8]; imul eax, eax, 12;
    add eax, edx` → drops the mov, rewrites add to use the LABEL
    directly. Saves 1 instruction (~3 bytes net for non-EAX, 5 for
    EAX dest)."""
    asm = (
        "_f:\n"
        "        mov     edx, _g_arr\n"
        "        mov     eax, [ebp + 8]\n"
        "        imul    eax, eax, 12\n"
        "        add     eax, edx\n"
        "        mov     [eax], ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     edx, _g_arr" not in out
    assert "add     eax, _g_arr" in out
    assert opt.stats.get("chain_label_to_add_operand") == 1


def test_chain_label_to_add_operand_through_intermediates():
    """Up to 8 intermediate instructions allowed between the mov
    and the add, as long as none touch BASE."""
    asm = (
        "_f:\n"
        "        mov     edx, _glob\n"
        "        mov     eax, [ebp + 8]\n"
        "        shl     eax, 2\n"
        "        mov     ecx, [ebp + 12]\n"
        "        and     eax, ecx\n"
        "        add     eax, edx\n"
        "        mov     [ebp - 4], eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     edx, _glob" not in out
    assert "add     eax, _glob" in out
    assert opt.stats.get("chain_label_to_add_operand") == 1


def test_chain_label_to_add_operand_skips_when_base_read():
    """If intermediate code reads BASE, the chain is broken."""
    asm = (
        "_f:\n"
        "        mov     edx, _glob\n"
        "        mov     eax, edx\n"  # reads edx
        "        add     eax, edx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("chain_label_to_add_operand", 0) == 0


def test_chain_label_to_add_operand_skips_when_base_written():
    """If intermediate code overwrites BASE before the add, the
    pass should fire only on the SECOND mov (matching the second
    add). Verify it doesn't double-fire."""
    asm = (
        "_f:\n"
        "        mov     edx, _foo\n"
        "        mov     edx, _bar\n"  # overwrites edx
        "        add     eax, edx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # Only the second `mov edx, _bar` is a valid chain start; the
    # first `mov edx, _foo` is dead before any use.
    assert "add     eax, _bar" in out
    assert opt.stats.get("chain_label_to_add_operand") == 1


def test_chain_label_to_add_operand_skips_call():
    """Calls clobber caller-saved regs, so a call between mov and
    add invalidates the chain when BASE is caller-saved."""
    asm = (
        "_f:\n"
        "        mov     edx, _glob\n"
        "        call    _helper\n"  # clobbers edx
        "        add     eax, edx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("chain_label_to_add_operand", 0) == 0


def test_chain_label_to_add_operand_handles_other_ops():
    """Pass also fires for sub/and/or/xor/cmp/test/adc/sbb."""
    for op in ("sub", "and", "or", "xor", "cmp", "test"):
        asm = (
            "_f:\n"
            f"        mov     edx, _glob\n"
            f"        {op:<8}eax, edx\n"
            f"        ret\n"
        )
        opt = PeepholeOptimizer()
        out = opt.optimize(asm)
        assert "mov     edx, _glob" not in out, f"failed for {op}"
        assert f"{op:<8}eax, _glob" in out, f"failed for {op}"


def test_chain_label_to_add_operand_skips_imul():
    """imul has no `imul reg, imm` 2-operand form; only 3-operand
    `imul reg, reg, imm`. Pass should not fire on imul."""
    asm = (
        "_f:\n"
        "        mov     edx, _glob\n"
        "        imul    eax, edx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("chain_label_to_add_operand", 0) == 0


def test_chain_label_to_add_operand_skips_label_jump():
    """Control-flow boundary (label) ends the chain."""
    asm = (
        "_f:\n"
        "        mov     edx, _glob\n"
        "        cmp     eax, 0\n"
        "        jne     .L_skip\n"  # control flow boundary
        "        add     eax, edx\n"
        ".L_skip:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("chain_label_to_add_operand", 0) == 0


def test_same_memory_operand_reuse_add():
    """`mov reg, [m]; add reg, [m]` → `mov reg, [m]; add reg, reg`.

    After the load, REG holds [m]. The second operand can use REG
    instead of re-reading. Saves 1 byte (3-byte mem-form add → 2-byte
    reg-reg add)."""
    asm = (
        "_f:\n"
        "        mov     ecx, [ebp - 4]\n"
        "        add     ecx, [ebp - 4]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "add     ecx, ecx" in out
    assert "add     ecx, [ebp - 4]" not in out
    assert opt.stats.get("same_memory_operand_reuse", 0) == 1


def test_same_memory_operand_reuse_and_or_xor():
    """All commutative ops {and, or, xor} fire. `add` already covered."""
    for op in ["and", "or", "xor"]:
        asm = (
            "_f:\n"
            f"        mov     eax, [ebp + 8]\n"
            f"        {op}     eax, [ebp + 8]\n"
            "        ret\n"
        )
        opt = PeepholeOptimizer()
        out = opt.optimize(asm)
        spacer = " " * max(1, 8 - len(op))
        assert f"{op}{spacer}eax, eax" in out
        assert opt.stats.get("same_memory_operand_reuse", 0) == 1


def test_same_memory_operand_reuse_imul():
    """imul is commutative — `int x; x*x` lowers to `mov eax, [m]; imul
    eax, [m]` which collapses to `mov eax, [m]; imul eax, eax`."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        imul    eax, [ebp + 8]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "imul    eax, eax" in out
    assert "imul    eax, [ebp + 8]" not in out
    assert opt.stats.get("same_memory_operand_reuse", 0) == 1


def test_same_memory_operand_reuse_skips_sub():
    """`sub` is not commutative — `mov reg, [m]; sub reg, [m]` IS
    always 0, but the rewrite to `sub reg, reg` would be a different
    optimization (zero idiom). Skip for safety; that's a separate slice."""
    asm = (
        "_f:\n"
        "        mov     ecx, [ebp - 4]\n"
        "        sub     ecx, [ebp - 4]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("same_memory_operand_reuse", 0) == 0


def test_same_memory_operand_reuse_skips_cmp_test():
    """`cmp` and `test` set flags differently when the second operand
    is replaced by REG. `cmp REG, [m]` where REG=[m] gives ZF=1, all
    other flags 0. `cmp REG, REG` where REG=[m] gives the same flags.
    But `test REG, REG` gives ZF=([m]==0), SF=high_bit. Different! For
    safety, skip both. (cmp could be safe, but conservative beats
    clever.)"""
    for op in ["cmp", "test"]:
        asm = (
            "_f:\n"
            "        mov     ecx, [ebp - 4]\n"
            f"        {op}     ecx, [ebp - 4]\n"
            "        ret\n"
        )
        opt = PeepholeOptimizer()
        opt.optimize(asm)
        assert opt.stats.get("same_memory_operand_reuse", 0) == 0


def test_same_memory_operand_reuse_skips_different_mem():
    """Different memory operands — must NOT collapse."""
    asm = (
        "_f:\n"
        "        mov     ecx, [ebp - 4]\n"
        "        add     ecx, [ebp - 8]\n"  # different offset
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "add     ecx, [ebp - 8]" in out
    assert opt.stats.get("same_memory_operand_reuse", 0) == 0


def test_same_memory_operand_reuse_skips_different_reg():
    """Different destination registers — load goes into ECX, op goes
    into EAX. EAX doesn't hold [m], so we can't substitute."""
    asm = (
        "_f:\n"
        "        mov     ecx, [ebp - 4]\n"
        "        add     eax, [ebp - 4]\n"  # different dest
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "add     eax, [ebp - 4]" in out
    assert opt.stats.get("same_memory_operand_reuse", 0) == 0


def test_same_memory_operand_reuse_skips_label_mem():
    """Source is `[_glob]` — label memory could be volatile MMIO. We
    restrict to ebp-relative for safety."""
    asm = (
        "_f:\n"
        "        mov     ecx, [_glob]\n"
        "        add     ecx, [_glob]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("same_memory_operand_reuse", 0) == 0


def test_same_memory_operand_reuse_skips_sib_mem():
    """Source is `[ebp + ecx*4]` — SIB form, not a frame slot. Skip."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + ecx*4]\n"
        "        add     eax, [ebp + ecx*4]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("same_memory_operand_reuse", 0) == 0


def test_same_memory_operand_reuse_chain_pattern():
    """Realistic chain after store_chain_retarget: end-to-end test
    that the i+i pattern collapses through the pipeline."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp - 8]\n"
        "        lea     eax, [_g + eax*4]\n"
        "        push    eax\n"
        "        mov     eax, [ebp - 4]\n"
        "        add     eax, [ebp - 4]\n"
        "        pop     ecx\n"
        "        mov     [ecx], eax\n"
        "        xor     eax, eax\n"
        "        xor     ecx, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # store_chain_retarget rewrote chain to ECX, then
    # same_memory_operand_reuse simplified add ecx, [ebp - 4] to add ecx, ecx.
    assert "add     ecx, ecx" in out
    assert opt.stats.get("store_chain_retarget", 0) == 1
    assert opt.stats.get("same_memory_operand_reuse", 0) == 1


def test_same_memory_operand_reuse_negative_offset_match():
    """`[ebp - 4]` and `[ebp - 4]` match (same negative offset)."""
    asm = (
        "_f:\n"
        "        mov     edx, [ebp - 16]\n"
        "        and     edx, [ebp - 16]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "and     edx, edx" in out
    assert opt.stats.get("same_memory_operand_reuse", 0) == 1


def test_same_memory_operand_reuse_whitespace_tolerant():
    """Memory operand text comparison should be whitespace-normalized."""
    asm = (
        "_f:\n"
        "        mov     ecx, [ebp - 4]\n"
        "        add     ecx, [ ebp - 4 ]\n"  # extra whitespace
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "add     ecx, ecx" in out
    assert opt.stats.get("same_memory_operand_reuse", 0) == 1


def test_shift_const_imm_shl():
    """`mov ecx, 3; shl eax, cl` → `shl eax, 3`. Saves 5 bytes."""
    asm = (
        "_f:\n"
        "        mov     ecx, 3\n"
        "        shl     eax, cl\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "shl     eax, 3" in out
    assert "mov     ecx, 3" not in out
    assert opt.stats.get("shift_const_imm", 0) == 1


def test_shift_const_imm_all_ops():
    """All shift/rotate ops fire."""
    for op in ["shl", "shr", "sar", "sal", "rol", "ror", "rcl", "rcr"]:
        asm = (
            "_f:\n"
            f"        mov     ecx, 5\n"
            f"        {op}     eax, cl\n"
            "        ret\n"
        )
        opt = PeepholeOptimizer()
        out = opt.optimize(asm)
        spacer = " " * max(1, 8 - len(op))
        assert f"{op}{spacer}eax, 5" in out
        assert opt.stats.get("shift_const_imm", 0) == 1


def test_shift_const_imm_three_operand():
    """shld/shrd are 3-operand: `shld dst, src, cl` → `shld dst, src, IMM`."""
    asm = (
        "_f:\n"
        "        mov     ecx, 4\n"
        "        shld    eax, ebx, cl\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "shld    eax, ebx, 4" in out
    assert "mov     ecx, 4" not in out
    assert opt.stats.get("shift_const_imm", 0) == 1


def test_shift_const_imm_skips_imm_too_large():
    """IMM > 31 — skip. x86 hardware masks to 31 anyway, but the
    explicit form is invalid for >31. Conservative: only allow 0..31."""
    asm = (
        "_f:\n"
        "        mov     ecx, 32\n"
        "        shl     eax, cl\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("shift_const_imm", 0) == 0


def test_shift_const_imm_skips_negative_imm():
    """Negative IMM — skip."""
    asm = (
        "_f:\n"
        "        mov     ecx, -1\n"
        "        shl     eax, cl\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("shift_const_imm", 0) == 0


def test_shift_const_imm_skips_non_numeric():
    """Source is a label or memory — skip."""
    asm = (
        "_f:\n"
        "        mov     ecx, _shift_count\n"  # label
        "        shl     eax, cl\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("shift_const_imm", 0) == 0


def test_shift_const_imm_skips_dest_ecx():
    """Destination is ECX — skip (we'd be dropping the only ECX def
    that drives the destination)."""
    asm = (
        "_f:\n"
        "        mov     ecx, 3\n"
        "        shl     ecx, cl\n"  # dest = ecx
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("shift_const_imm", 0) == 0


def test_shift_const_imm_skips_ecx_live_after():
    """ECX has subsequent reader — skip."""
    asm = (
        "_f:\n"
        "        mov     ecx, 2\n"
        "        shl     eax, cl\n"
        "        mov     edx, ecx\n"  # reads ECX
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("shift_const_imm", 0) == 0


def test_shift_const_imm_zero_shift_handled_by_mov_zero_to_xor():
    """Shift by 0 is degenerate — the optimizer folds `x << 0` to
    `x` before codegen, so this case doesn't arise in practice. If
    `mov ecx, 0` somehow reaches the peephole, `mov_zero_to_xor`
    runs before my pass and converts to `xor ecx, ecx`. My pass
    doesn't recognize that form (and there's no point — shift-by-0
    doesn't actually arise), so the result is unchanged from
    mov_zero_to_xor's output."""
    asm = (
        "_f:\n"
        "        mov     ecx, 0\n"
        "        shl     eax, cl\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # mov_zero_to_xor fires first; shift_const_imm doesn't fire.
    assert "xor     ecx, ecx" in out
    assert opt.stats.get("shift_const_imm", 0) == 0


def test_shift_const_imm_max_shift():
    """Shift by 31 (max for 32-bit) — should fire."""
    asm = (
        "_f:\n"
        "        mov     ecx, 31\n"
        "        shl     eax, cl\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "shl     eax, 31" in out
    assert opt.stats.get("shift_const_imm", 0) == 1


def test_shift_const_imm_hex_imm():
    """Hex immediate works."""
    asm = (
        "_f:\n"
        "        mov     ecx, 0x10\n"  # 16
        "        shl     eax, cl\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "shl     eax, 16" in out
    assert opt.stats.get("shift_const_imm", 0) == 1


def test_div_mem_form_idiv():
    """`mov ecx, [m]; cdq; idiv ecx` → `cdq; idiv [m]`. Saves 2 bytes."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     ecx, [ebp + 12]\n"
        "        cdq\n"
        "        idiv    ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "idiv    dword [ebp + 12]" in out
    assert "mov     ecx, [ebp + 12]" not in out
    assert opt.stats.get("div_mem_form", 0) == 1


def test_div_mem_form_div_unsigned():
    """`mov ecx, [m]; xor edx, edx; div ecx` → `xor edx, edx; div [m]`."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     ecx, [ebp + 12]\n"
        "        xor     edx, edx\n"
        "        div     ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "div     dword [ebp + 12]" in out
    assert "mov     ecx, [ebp + 12]" not in out
    assert opt.stats.get("div_mem_form", 0) == 1


def test_div_mem_form_other_regs():
    """Works for ebx/esi/edi too."""
    for reg in ["ebx", "esi", "edi"]:
        asm = (
            "_f:\n"
            f"        mov     {reg}, [ebp + 12]\n"
            f"        cdq\n"
            f"        idiv    {reg}\n"
            "        ret\n"
        )
        opt = PeepholeOptimizer()
        out = opt.optimize(asm)
        assert "idiv    dword [ebp + 12]" in out
        assert opt.stats.get("div_mem_form", 0) == 1


def test_div_mem_form_skips_eax_edx():
    """Source register can't be EAX or EDX (those hold the dividend)."""
    for reg in ["eax", "edx"]:
        asm = (
            "_f:\n"
            f"        mov     {reg}, [ebp + 12]\n"
            "        cdq\n"
            f"        idiv    {reg}\n"
            "        ret\n"
        )
        opt = PeepholeOptimizer()
        opt.optimize(asm)
        assert opt.stats.get("div_mem_form", 0) == 0


def test_div_mem_form_skips_mem_references_eax():
    """If [m] references EAX (e.g., `[eax + 4]`), the rewrite would
    read from eax AFTER cdq has clobbered EDX (and idiv reads
    EDX:EAX). Bail."""
    asm = (
        "_f:\n"
        "        mov     ecx, [eax + 4]\n"  # [m] references eax
        "        cdq\n"
        "        idiv    ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("div_mem_form", 0) == 0


def test_div_mem_form_skips_mem_references_edx():
    """If [m] references EDX, similar issue: cdq clobbers EDX before
    idiv reads [m]."""
    asm = (
        "_f:\n"
        "        mov     ecx, [edx + 4]\n"  # [m] references edx
        "        cdq\n"
        "        idiv    ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("div_mem_form", 0) == 0


def test_div_mem_form_skips_mismatched_pair():
    """`xor edx, edx` paired with `idiv` doesn't fire (signed div
    needs cdq, not xor). Conservative."""
    asm = (
        "_f:\n"
        "        mov     ecx, [ebp + 12]\n"
        "        xor     edx, edx\n"
        "        idiv    ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("div_mem_form", 0) == 0


def test_div_mem_form_skips_reg_live_after():
    """If REG is read after the idiv, dropping the load breaks."""
    asm = (
        "_f:\n"
        "        mov     ecx, [ebp + 12]\n"
        "        cdq\n"
        "        idiv    ecx\n"
        "        mov     edx, ecx\n"  # ECX live after
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("div_mem_form", 0) == 0


def test_div_mem_form_skips_register_source():
    """Source must be memory (with `[...]`), not a register."""
    asm = (
        "_f:\n"
        "        mov     ecx, ebx\n"  # source is reg
        "        cdq\n"
        "        idiv    ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("div_mem_form", 0) == 0


def test_div_mem_form_label_memory():
    """Label memory `[_glob]` works (no register references)."""
    asm = (
        "_f:\n"
        "        mov     ecx, [_glob]\n"
        "        cdq\n"
        "        idiv    ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "idiv    dword [_glob]" in out
    assert opt.stats.get("div_mem_form", 0) == 1


def test_mov_test_setcc_movzx_collapse_eq():
    """`mov eax, [m]; test eax, eax; sete al; movzx eax, al` →
    `cmp dword [m], 0; sete al; movzx eax, al`. Saves 1 byte."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        test    eax, eax\n"
        "        sete    al\n"
        "        movzx   eax, al\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "cmp     dword [ebp + 8], 0" in out
    assert "test    eax, eax" not in out
    assert "mov     eax, [ebp + 8]" not in out
    assert "sete    al" in out
    assert "movzx   eax, al" in out
    assert opt.stats.get("mov_test_setcc_movzx_collapse", 0) == 1


def test_mov_test_setcc_movzx_collapse_all_setcc_variants():
    """All standard setCC variants fire."""
    for setcc in ["sete", "setne", "setl", "setg", "setle", "setge",
                   "seta", "setb", "setae", "setbe", "sets", "setns"]:
        asm = (
            "_f:\n"
            "        mov     eax, [ebp + 8]\n"
            "        test    eax, eax\n"
            f"        {setcc:<8}al\n"
            "        movzx   eax, al\n"
            "        ret\n"
        )
        opt = PeepholeOptimizer()
        out = opt.optimize(asm)
        assert "cmp     dword [ebp + 8], 0" in out
        assert opt.stats.get("mov_test_setcc_movzx_collapse", 0) == 1


def test_mov_test_setcc_movzx_collapse_other_reg():
    """Pattern fires for any GP32 source register, not just EAX."""
    for reg in ["ebx", "ecx", "edx", "esi", "edi"]:
        asm = (
            "_f:\n"
            f"        mov     {reg}, [ebp + 8]\n"
            f"        test    {reg}, {reg}\n"
            "        sete    al\n"
            "        movzx   eax, al\n"
            "        ret\n"
        )
        opt = PeepholeOptimizer()
        out = opt.optimize(asm)
        assert "cmp     dword [ebp + 8], 0" in out
        assert opt.stats.get("mov_test_setcc_movzx_collapse", 0) == 1


def test_mov_test_setcc_movzx_collapse_skips_register_source():
    """Source must be memory, not a register (no `[...]`)."""
    asm = (
        "_f:\n"
        "        mov     eax, ebx\n"  # source is reg
        "        test    eax, eax\n"
        "        sete    al\n"
        "        movzx   eax, al\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("mov_test_setcc_movzx_collapse", 0) == 0


def test_mov_test_setcc_movzx_collapse_skips_test_different_reg():
    """`test ecx, ecx` after `mov eax, [m]` — different reg, skip."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        test    ecx, ecx\n"  # different reg
        "        sete    al\n"
        "        movzx   eax, al\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("mov_test_setcc_movzx_collapse", 0) == 0


def test_mov_test_setcc_movzx_collapse_skips_setcc_other_reg():
    """`setCC bl` (not al) — skip. The movzx reads only AL; if
    setCC writes elsewhere, we'd break the chain."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        test    eax, eax\n"
        "        sete    bl\n"  # not al
        "        movzx   eax, al\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("mov_test_setcc_movzx_collapse", 0) == 0


def test_mov_test_setcc_movzx_collapse_skips_other_movzx():
    """`movzx ebx, al` (not eax) — skip. The full pattern
    requires the final dest to be EAX."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        test    eax, eax\n"
        "        sete    al\n"
        "        movzx   ebx, al\n"  # not eax
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("mov_test_setcc_movzx_collapse", 0) == 0


def test_mov_test_setcc_movzx_collapse_skips_movzx_from_other_reg():
    """`movzx eax, bl` (not al) — skip. Only AL is the canonical
    setCC destination."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        test    eax, eax\n"
        "        sete    al\n"
        "        movzx   eax, bl\n"  # from bl
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("mov_test_setcc_movzx_collapse", 0) == 0


def test_mov_test_setcc_movzx_collapse_skips_mem_with_eax():
    """If [m] references EAX (e.g., `[eax + 4]`), we'd need EAX to
    be loaded first. Skip."""
    asm = (
        "_f:\n"
        "        mov     eax, [eax + 4]\n"  # mem references eax
        "        test    eax, eax\n"
        "        sete    al\n"
        "        movzx   eax, al\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("mov_test_setcc_movzx_collapse", 0) == 0


def test_mov_test_setcc_movzx_collapse_global_memory():
    """Global memory `[_glob]` works (no register references)."""
    asm = (
        "_f:\n"
        "        mov     eax, [_glob]\n"
        "        test    eax, eax\n"
        "        sete    al\n"
        "        movzx   eax, al\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "cmp     dword [_glob], 0" in out
    assert opt.stats.get("mov_test_setcc_movzx_collapse", 0) == 1


def test_rmw_mem_src_collapse_basic():
    """`mov eax, [m1]; add eax, [m2]; mov [m1], eax; <eax-dead>` →
    `mov eax, [m2]; add [m1], eax`. Saves 3 bytes.

    Tests use `xor eax, eax` after the pattern to make EAX dead
    (the rewrite changes EAX's final value, so we need EAX dead)."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp - 4]\n"
        "        add     eax, [ebp - 8]\n"
        "        mov     [ebp - 4], eax\n"
        "        xor     eax, eax\n"  # EAX dead
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     eax, [ebp - 8]" in out
    assert "add     [ebp - 4], eax" in out
    # The original load+store of [ebp - 4] are gone.
    assert "mov     eax, [ebp - 4]" not in out
    assert "mov     [ebp - 4], eax" not in out
    assert opt.stats.get("rmw_mem_src_collapse", 0) == 1


def test_rmw_mem_src_collapse_all_ops():
    """All supported OPs fire."""
    for op in ["add", "sub", "and", "or", "xor"]:
        asm = (
            "_f:\n"
            "        mov     eax, [ebp - 4]\n"
            f"        {op}     eax, [ebp - 8]\n"
            "        mov     [ebp - 4], eax\n"
            "        xor     eax, eax\n"  # EAX dead
            "        ret\n"
        )
        opt = PeepholeOptimizer()
        out = opt.optimize(asm)
        spacer = " " * max(1, 8 - len(op))
        assert f"{op}{spacer}[ebp - 4], eax" in out
        assert opt.stats.get("rmw_mem_src_collapse", 0) == 1


def test_rmw_mem_src_collapse_skips_imul():
    """imul has no `imul r/m32, imm/r32` form for two memory operands.
    Single-operand imul writes EDX:EAX. Skip imul (and imul-mem-form
    isn't a 2-byte savings anyway)."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp - 4]\n"
        "        imul    eax, [ebp - 8]\n"
        "        mov     [ebp - 4], eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("rmw_mem_src_collapse", 0) == 0


def test_rmw_mem_src_collapse_label_memory():
    """Label memory `[_g]` and `[_h]` work."""
    asm = (
        "_f:\n"
        "        mov     eax, [_g]\n"
        "        add     eax, [_h]\n"
        "        mov     [_g], eax\n"
        "        xor     eax, eax\n"  # EAX dead
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     eax, [_h]" in out
    assert "add     [_g], eax" in out
    assert opt.stats.get("rmw_mem_src_collapse", 0) == 1


def test_rmw_mem_src_collapse_skips_same_mem():
    """If [m1] == [m2], same_memory_operand_reuse handles a
    different optimization — and rmw_collapse can't apply because
    the OP source would be EAX after the rewrite (rejected by
    rmw_collapse). Just bail."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp - 4]\n"
        "        add     eax, [ebp - 4]\n"  # same m
        "        mov     [ebp - 4], eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("rmw_mem_src_collapse", 0) == 0


def test_rmw_mem_src_collapse_skips_register_base_addr():
    """If [m1] uses register-base addressing, aliasing with [m2]
    is possible. Skip for safety."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebx]\n"  # register-base
        "        add     eax, [ebp - 8]\n"
        "        mov     [ebx], eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("rmw_mem_src_collapse", 0) == 0


def test_rmw_mem_src_collapse_skips_eax_live_after():
    """If EAX is read after the store, the rewrite changes EAX's
    final value (orig: eax = [m1] + [m2]; rewrite: eax = [m2])."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp - 4]\n"
        "        add     eax, [ebp - 8]\n"
        "        mov     [ebp - 4], eax\n"
        "        mov     [ebp - 12], eax\n"  # EAX read again
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("rmw_mem_src_collapse", 0) == 0


def test_rmw_mem_src_collapse_skips_mismatched_dest():
    """If `mov [m1'], eax` stores to a DIFFERENT slot, this isn't a
    compound assign on m1. Skip."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp - 4]\n"
        "        add     eax, [ebp - 8]\n"
        "        mov     [ebp - 12], eax\n"  # different from [ebp - 4]
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("rmw_mem_src_collapse", 0) == 0


def test_rmw_mem_src_collapse_skips_src_references_dest_reg():
    """If [m2] references the dest register (e.g., `[eax + 4]`),
    skip — the rewrite's `mov reg, [m2]` would self-reference."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp - 4]\n"
        "        add     eax, [eax + 4]\n"  # references eax
        "        mov     [ebp - 4], eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("rmw_mem_src_collapse", 0) == 0


def test_rmw_mem_src_collapse_other_reg():
    """Pattern works for non-EAX registers too."""
    asm = (
        "_f:\n"
        "        mov     ecx, [ebp - 4]\n"
        "        add     ecx, [ebp - 8]\n"
        "        mov     [ebp - 4], ecx\n"
        "        xor     ecx, ecx\n"  # ECX dead
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     ecx, [ebp - 8]" in out
    assert "add     [ebp - 4], ecx" in out
    assert opt.stats.get("rmw_mem_src_collapse", 0) == 1


def test_dead_stack_store_no_size_prefix_eax():
    """`mov [ebp - 4], eax; mov [ebp - 4], eax` (no size prefix —
    NASM infers dword from eax) — first store is dead. Common in
    chained assigns like `y = x; y += 5; y *= 2;` where each
    assignment to y is a register store with no explicit size."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     [ebp - 4], eax\n"
        "        add     eax, 5\n"
        "        mov     [ebp - 4], eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # The first `mov [ebp - 4], eax` is dead (overwritten by second).
    # Both stores have no size prefix.
    occurrences = out.count("mov     [ebp - 4], eax")
    assert occurrences == 1, f"Expected 1 store, got {occurrences}: {out}"
    assert opt.stats.get("dead_stack_store", 0) >= 1


def test_dead_stack_store_no_size_prefix_ecx():
    """Pattern works for any 32-bit GP register, not just eax."""
    asm = (
        "_f:\n"
        "        mov     ecx, 5\n"
        "        mov     [ebp - 4], ecx\n"
        "        mov     ecx, 10\n"
        "        mov     [ebp - 4], ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    occurrences = out.count("mov     [ebp - 4], ecx")
    assert occurrences == 1
    assert opt.stats.get("dead_stack_store", 0) >= 1


def test_dead_stack_store_no_size_prefix_skips_subreg():
    """`mov [ebp - 4], al` (byte) → don't drop subsequent dword
    store, since the sub-byte-write doesn't fully overwrite the
    target slot. Conservative."""
    asm = (
        "_f:\n"
        "        mov     [ebp - 4], al\n"  # byte store
        "        mov     [ebp - 4], eax\n"  # dword store
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    # The byte store isn't recognized as a dword store, so
    # dead_stack_store doesn't fire.
    assert opt.stats.get("dead_stack_store", 0) == 0


def test_dead_stack_store_chained_three_assigns():
    """`y = x; y += 5; y *= 2;` — two intermediate stores die."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     [ebp - 4], eax\n"  # y = x (dead)
        "        add     eax, 5\n"
        "        mov     [ebp - 4], eax\n"  # y += 5 (dead)
        "        imul    eax, 2\n"
        "        mov     [ebp - 4], eax\n"  # y *= 2 (live - final)
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # Only the FINAL store survives.
    occurrences = out.count("mov     [ebp - 4], eax")
    assert occurrences == 1, f"Expected 1 final store, got {occurrences}"
    assert opt.stats.get("dead_stack_store", 0) >= 2


def test_reg_copy_addr_forward_basic():
    """`mov ecx, eax; mov [ecx], V` → `mov [eax], V`. Saves 2 bytes."""
    asm = (
        "_f:\n"
        "        mov     ecx, eax\n"
        "        mov     dword [ecx], 5\n"
        "        xor     ecx, ecx\n"  # ecx dead after
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     dword [eax], 5" in out
    assert "mov     ecx, eax" not in out
    assert opt.stats.get("reg_copy_addr_forward", 0) == 1


def test_reg_copy_addr_forward_load():
    """`mov ecx, eax; mov edx, [ecx]` → `mov edx, [eax]`."""
    asm = (
        "_f:\n"
        "        mov     ecx, eax\n"
        "        mov     edx, [ecx]\n"
        "        xor     ecx, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     edx, [eax]" in out
    assert opt.stats.get("reg_copy_addr_forward", 0) == 1


def test_reg_copy_addr_forward_disp():
    """`mov ecx, eax; mov edx, [ecx + 8]` → `mov edx, [eax + 8]`."""
    asm = (
        "_f:\n"
        "        mov     ecx, eax\n"
        "        mov     edx, [ecx + 8]\n"
        "        xor     ecx, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     edx, [eax + 8]" in out
    assert opt.stats.get("reg_copy_addr_forward", 0) == 1


def test_reg_copy_addr_forward_skips_no_addr_use():
    """If line B doesn't use REGB in addressing, don't fire."""
    asm = (
        "_f:\n"
        "        mov     ecx, eax\n"
        "        mov     edx, ecx\n"  # uses ecx as src, not in [...]
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    # value_forward_to_reg might fire here, but reg_copy_addr_forward
    # should NOT.
    assert opt.stats.get("reg_copy_addr_forward", 0) == 0


def test_reg_copy_addr_forward_skips_regb_live_after():
    """If REGB is read after line B, don't drop the copy."""
    asm = (
        "_f:\n"
        "        mov     ecx, eax\n"
        "        mov     edx, [ecx]\n"
        "        mov     esi, ecx\n"  # ecx still live
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("reg_copy_addr_forward", 0) == 0


def test_reg_copy_addr_forward_skips_b_writes_rega():
    """If line B writes to REGA, REGA's value would change before
    addressing resolves. Conservative: skip."""
    asm = (
        "_f:\n"
        "        mov     ecx, eax\n"
        "        mov     eax, [ecx]\n"  # writes eax (= REGA)
        "        xor     ecx, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("reg_copy_addr_forward", 0) == 0


def test_reg_copy_addr_forward_index_use():
    """`mov ecx, eax; mov edx, [ebx + ecx*4]` → `mov edx, [ebx + eax*4]`."""
    asm = (
        "_f:\n"
        "        mov     ecx, eax\n"
        "        mov     edx, [ebx + ecx*4]\n"
        "        xor     ecx, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     edx, [ebx + eax*4]" in out
    assert opt.stats.get("reg_copy_addr_forward", 0) == 1


def test_reg_copy_addr_forward_skips_self_mov():
    """`mov eax, eax` is a self-mov; skip (handled by self_mov_elim)."""
    asm = (
        "_f:\n"
        "        mov     eax, eax\n"  # self-mov
        "        mov     ecx, [eax]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    # The self-mov gets eliminated by another pass; we don't fire.
    assert opt.stats.get("reg_copy_addr_forward", 0) == 0


def test_reg_copy_addr_forward_store_through_pointer():
    """Realistic case: `*p = V` lowering. Codegen often emits
    `lea reg, addr; mov ecx, reg; mov [ecx], V`."""
    asm = (
        "_f:\n"
        "        lea     eax, [ebp + 8]\n"
        "        mov     ecx, eax\n"
        "        mov     dword [ecx], 5\n"
        "        xor     ecx, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # mov ecx, eax dropped; substituted to use eax directly.
    assert "mov     ecx, eax" not in out
    assert "mov     dword [eax], 5" in out
    assert opt.stats.get("reg_copy_addr_forward", 0) == 1


def test_lea_sib_load_collapse_basic():
    """Non-adjacent lea + SIB load: `lea eax, [ebp - 20]; <indep>;
    mov eax, [eax + ecx*4]` → `<indep>; mov eax, [ebp + ecx*4 - 20]`.
    Saves 3 bytes (drops the lea)."""
    asm = (
        "_f:\n"
        "        lea     eax, [ebp - 20]\n"
        "        mov     ecx, [ebp - 32]\n"
        "        mov     eax, [eax + ecx*4]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "lea     eax, [ebp - 20]" not in out
    assert "ebp + ecx*4 - 20" in out
    assert opt.stats.get("lea_sib_load_collapse", 0) == 1


def test_lea_sib_load_collapse_plain_deref():
    """Non-adjacent lea + plain deref: `lea eax, [ebp - 20]; <indep>;
    mov eax, [eax]` → `<indep>; mov eax, [ebp - 20]`."""
    asm = (
        "_f:\n"
        "        lea     eax, [ebp - 20]\n"
        "        mov     ecx, 0\n"
        "        mov     eax, [eax]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "lea     eax, [ebp - 20]" not in out
    assert "mov     eax, [ebp - 20]" in out
    assert opt.stats.get("lea_sib_load_collapse", 0) == 1


def test_lea_sib_load_collapse_disp_deref():
    """Non-adjacent lea + disp deref: `lea eax, [ebp - 20]; <indep>;
    mov eax, [eax + 8]` → `<indep>; mov eax, [ebp - 12]`."""
    asm = (
        "_f:\n"
        "        lea     eax, [ebp - 20]\n"
        "        mov     ecx, 0\n"
        "        mov     eax, [eax + 8]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "lea     eax, [ebp - 20]" not in out
    assert "mov     eax, [ebp - 12]" in out
    assert opt.stats.get("lea_sib_load_collapse", 0) == 1


def test_lea_sib_load_collapse_skips_intermediate_writes_lea_reg():
    """If an intermediate writes to the lea target, can't fold."""
    asm = (
        "_f:\n"
        "        lea     eax, [ebp - 20]\n"
        "        mov     eax, 0\n"  # writes eax
        "        mov     eax, [eax + 4]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("lea_sib_load_collapse", 0) == 0


def test_lea_sib_load_collapse_skips_intermediate_reads_lea_reg():
    """If an intermediate reads the lea target, the value is being
    used elsewhere — bail."""
    asm = (
        "_f:\n"
        "        lea     eax, [ebp - 20]\n"
        "        mov     ecx, eax\n"  # reads eax
        "        mov     edx, [eax + 4]\n"
        "        xor     eax, eax\n"  # eax dead
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    # The intermediate `mov ecx, eax` reads eax, so we bail.
    assert opt.stats.get("lea_sib_load_collapse", 0) == 0


def test_lea_sib_load_collapse_skips_lea_reg_live_after():
    """If lea target is live after the load, can't drop the lea."""
    asm = (
        "_f:\n"
        "        lea     eax, [ebp - 20]\n"
        "        mov     ecx, [ebp - 32]\n"
        "        mov     edx, [eax + ecx*4]\n"  # eax not overwritten
        "        mov     [_glob], eax\n"  # eax read
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("lea_sib_load_collapse", 0) == 0


def test_lea_sib_load_collapse_dest_overwrites_lea_reg():
    """If the load destination IS the lea reg, the load overwrites
    the lea result — fine, lea is dead by virtue of overwrite."""
    asm = (
        "_f:\n"
        "        lea     eax, [ebp - 20]\n"
        "        mov     ecx, [ebp - 32]\n"
        "        mov     eax, [eax + ecx*4]\n"  # eax = dest = lea reg
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "lea     eax" not in out
    assert opt.stats.get("lea_sib_load_collapse", 0) == 1


# ── dup_addr_compute_collapse ─────────────────────────────────────


def test_dup_addr_compute_collapse_basic():
    """The canonical loop-body shape: a 3-line address compute
    followed by a deref preserving the address reg, followed by a
    4-line recompute, followed by another deref. The recompute is
    redundant and gets dropped.

    Loop top has `mov ecx, [ebp - 8]` setting up R2's pre-A value.
    """
    asm = (
        ".L1_for_top:\n"
        "        mov     ecx, [ebp - 8]\n"
        "        cmp     ecx, [ebp + 12]\n"
        "        jge     .L_end\n"
        "        mov     eax, [ebp + 8]\n"     # A
        "        imul    ecx, ecx, 56\n"       # C
        "        add     eax, ecx\n"           # D
        "        mov     edx, [eax]\n"         # E (R3=edx, != eax/ecx)
        "        mov     eax, [ebp + 8]\n"     # F = A (drop)
        "        mov     ecx, [ebp - 8]\n"     # G (drop)
        "        imul    ecx, ecx, 56\n"       # H = C (drop)
        "        add     eax, ecx\n"           # I = D (drop)
        "        mov     eax, [eax + 4]\n"     # J: 2nd deref
        "        ret\n"
        ".L_end:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # The recompute (4 lines after edx deref) should be dropped.
    # The remaining structure: mov eax,[ebp+8]; imul ...; add ...;
    # mov edx,[eax]; mov eax,[eax+4]
    # We should see only ONE `mov eax, [ebp + 8]` (the first one).
    assert out.count("mov     eax, [ebp + 8]") == 1
    # The pass should fire 4 times (4 lines dropped).
    assert opt.stats.get("dup_addr_compute_collapse", 0) == 4


def test_dup_addr_compute_collapse_skips_when_pre_value_differs():
    """If R2's pre-A value (loaded at loop top) differs from G's
    source MEM2_X, refuse to drop (would change R2 semantics)."""
    asm = (
        ".L1_for_top:\n"
        "        mov     ecx, [ebp - 4]\n"     # pre-A: ECX = -4
        "        mov     eax, [ebp + 8]\n"     # A
        "        imul    ecx, ecx, 56\n"       # C
        "        add     eax, ecx\n"           # D
        "        mov     edx, [eax]\n"         # E
        "        mov     eax, [ebp + 8]\n"     # F
        "        mov     ecx, [ebp - 8]\n"     # G: loads from -8 (MISMATCH)
        "        imul    ecx, ecx, 56\n"       # H
        "        add     eax, ecx\n"           # I
        "        mov     eax, [eax + 4]\n"     # J
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # Should NOT drop the recompute since pre-value mismatch.
    assert out.count("mov     ecx, [ebp - 8]") == 1
    assert opt.stats.get("dup_addr_compute_collapse", 0) == 0


def test_dup_addr_compute_collapse_skips_when_r3_eq_r1():
    """If E's destination IS R1 (e.g., `mov eax, [eax]`), R1 gets
    clobbered by the deref. Recompute is necessary, must not drop."""
    asm = (
        ".L_top:\n"
        "        mov     ecx, [ebp - 8]\n"
        "        mov     eax, [ebp + 8]\n"     # A
        "        imul    ecx, ecx, 56\n"       # C
        "        add     eax, ecx\n"           # D
        "        mov     eax, [eax]\n"         # E: R3 = eax = R1!
        "        mov     eax, [ebp + 8]\n"     # F (DON'T drop)
        "        mov     ecx, [ebp - 8]\n"     # G
        "        imul    ecx, ecx, 56\n"       # H
        "        add     eax, ecx\n"           # I
        "        mov     eax, [eax + 4]\n"     # J
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # All 5 instances of `mov eax, [ebp + 8]` etc. preserved
    # since E clobbers R1 (eax).
    assert out.count("mov     eax, [ebp + 8]") == 2
    assert opt.stats.get("dup_addr_compute_collapse", 0) == 0


def test_dup_addr_compute_collapse_with_imul_non_power_of_two():
    """Pattern fires for non-power-of-2 scales (12, 20, 56, etc.)
    where SIB-form addressing isn't possible. The `index_load_collapse`
    pass already handles the shl/scale-pow-of-2 case via SIB."""
    asm = (
        ".L_top:\n"
        "        mov     ecx, [ebp - 8]\n"
        "        mov     eax, [ebp + 8]\n"
        "        imul    ecx, ecx, 12\n"
        "        add     eax, ecx\n"
        "        mov     edx, [eax]\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     ecx, [ebp - 8]\n"
        "        imul    ecx, ecx, 12\n"
        "        add     eax, ecx\n"
        "        mov     eax, [eax + 4]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert out.count("mov     eax, [ebp + 8]") == 1
    assert opt.stats.get("dup_addr_compute_collapse", 0) == 4


def test_dup_addr_compute_collapse_skips_r2_modified_before_a():
    """If R2 is modified by something other than `mov R2, [m]`
    in the basic block before A, the pre-value is unknown and we
    must bail."""
    asm = (
        ".L_top:\n"
        "        mov     ecx, [ebp - 8]\n"
        "        add     ecx, 1\n"             # modifies ECX
        "        mov     eax, [ebp + 8]\n"     # A
        "        imul    ecx, ecx, 56\n"
        "        add     eax, ecx\n"
        "        mov     edx, [eax]\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     ecx, [ebp - 8]\n"
        "        imul    ecx, ecx, 56\n"
        "        add     eax, ecx\n"
        "        mov     eax, [eax + 4]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("dup_addr_compute_collapse", 0) == 0


# ── same_imm_store_share_reg ──────────────────────────────────────


def test_same_imm_store_share_reg_basic():
    """4 same-imm dword stores collapse to mov eax, IMM + 4 reg-stores.
    Use lea to take addresses so the address-taken bail prevents
    dead_unused_slot_stores from dropping the stores."""
    asm = (
        "_f:\n"
        "        enter   16, 0\n"
        "        mov     dword [ebp - 4], 7\n"
        "        mov     dword [ebp - 8], 7\n"
        "        mov     dword [ebp - 12], 7\n"
        "        mov     dword [ebp - 16], 7\n"
        "        lea     ecx, [ebp - 16]\n"  # address-take: bails dead_unused_slot_stores
        "        push    ecx\n"
        "        call    _helper\n"
        "        pop     ecx\n"
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    # Disable redundant_eax_load tracking by using imms that won't
    # become register-cached. Just verify the pass fires.
    out = opt.optimize(asm)
    assert opt.stats.get("same_imm_store_share_reg", 0) >= 2


def test_same_imm_store_share_reg_skips_single():
    """Single store doesn't collapse (would lose 1 byte)."""
    asm = (
        "_f:\n"
        "        enter   4, 0\n"
        "        mov     dword [ebp - 4], 7\n"
        "        mov     eax, [ebp - 4]\n"  # keep slot alive
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     dword [ebp - 4], 7" in out
    assert opt.stats.get("same_imm_store_share_reg", 0) == 0


def test_same_imm_store_share_reg_skips_different_imms():
    """Different imms: doesn't collapse."""
    asm = (
        "_f:\n"
        "        enter   8, 0\n"
        "        mov     dword [ebp - 4], 7\n"
        "        mov     dword [ebp - 8], 11\n"
        "        mov     eax, [ebp - 4]\n"
        "        add     eax, [ebp - 8]\n"
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # Both stores preserved
    assert "mov     dword [ebp - 4], 7" in out
    assert "mov     dword [ebp - 8], 11" in out
    assert opt.stats.get("same_imm_store_share_reg", 0) == 0


def test_same_imm_store_share_reg_skips_zero():
    """Zero imm is handled by zero_init_collapse, not this pass."""
    asm = (
        "_f:\n"
        "        enter   8, 0\n"
        "        mov     dword [ebp - 4], 0\n"
        "        mov     dword [ebp - 8], 0\n"
        "        mov     eax, 100\n"
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # zero_init_collapse handles this; my pass doesn't fire
    assert opt.stats.get("same_imm_store_share_reg", 0) == 0


def test_same_imm_store_share_reg_skips_eax_live():
    """If EAX is live (read) after the chain, can't clobber it."""
    asm = (
        "_f:\n"
        "        enter   8, 0\n"
        "        mov     dword [ebp - 4], 7\n"
        "        mov     dword [ebp - 8], 7\n"
        "        push    eax\n"  # EAX read — live
        "        mov     ecx, [ebp - 4]\n"  # keep slots alive
        "        add     ecx, [ebp - 8]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     dword [ebp - 4], 7" in out
    assert "mov     dword [ebp - 8], 7" in out
    assert opt.stats.get("same_imm_store_share_reg", 0) == 0


def test_same_imm_store_share_reg_byte_size():
    """Byte stores share AL."""
    asm = (
        "_f:\n"
        "        enter   8, 0\n"
        "        mov     byte [ebp - 1], 0x42\n"
        "        mov     byte [ebp - 2], 0x42\n"
        "        mov     byte [ebp - 3], 0x42\n"
        "        mov     eax, 100\n"
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     eax, 0x42" in out
    assert "mov     [ebp - 1], al" in out
    assert "mov     [ebp - 2], al" in out
    assert "mov     [ebp - 3], al" in out
    assert opt.stats.get("same_imm_store_share_reg", 0) == 3


def test_same_imm_store_share_reg_word_size():
    """Word stores share AX."""
    asm = (
        "_f:\n"
        "        enter   8, 0\n"
        "        mov     word [ebp - 2], 0x1234\n"
        "        mov     word [ebp - 4], 0x1234\n"
        "        mov     eax, 100\n"
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "mov     eax, 0x1234" in out
    assert "mov     [ebp - 2], ax" in out
    assert "mov     [ebp - 4], ax" in out
    assert opt.stats.get("same_imm_store_share_reg", 0) == 2


def test_same_imm_store_share_reg_negative_imm():
    """Negative imms work."""
    asm = (
        "_f:\n"
        "        enter   8, 0\n"
        "        mov     dword [ebp - 4], -1\n"
        "        mov     dword [ebp - 8], -1\n"
        "        lea     ecx, [ebp - 8]\n"  # address-take
        "        push    ecx\n"
        "        call    _helper\n"
        "        pop     ecx\n"
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("same_imm_store_share_reg", 0) >= 2


def test_cmp_load_promote_with_lea_d():
    """Extended: D may be `lea reg1, [...]` (in addition to `mov
    reg1, RHS`). The lea writes reg1 cleanly (doesn't read reg1),
    so it qualifies the same as mov for the promote rewrite.

    Common in struct-array index loops where the body opens with
    `lea reg1, [ebp - struct_base]` rather than a slot load.

    Need .L_end to have an EAX overwrite before leave/ret so the
    EAX-dead-at-target check passes."""
    asm = (
        ".L_top:\n"
        "        mov     eax, [ebp - 8]\n"     # A
        "        cmp     eax, [ebp + 8]\n"     # B
        "        jge     .L_end\n"             # C
        "        lea     eax, [ebp - 120]\n"   # D (lea, not mov)
        "        mov     ecx, [ebp - 8]\n"     # E
        "        imul    ecx, ecx, 12\n"
        "        add     eax, ecx\n"
        "        mov     [eax], ecx\n"
        "        jmp     .L_top\n"
        ".L_end:\n"
        "        mov     eax, 0\n"             # EAX overwrite (so EAX dead at jump target)
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # A/B should now use ECX (promoted), and E should be dropped.
    assert "cmp     ecx, [ebp + 8]" in out
    assert "mov     eax, [ebp - 8]" not in out  # original A gone
    # E was `mov ecx, [ebp - 8]` and gets dropped — but the
    # promoted A is `mov ecx, [ebp - 8]`. So one occurrence remains.
    assert out.count("mov     ecx, [ebp - 8]") == 1
    assert opt.stats.get("cmp_load_promote", 0) == 1


def test_same_imm_store_share_reg_skips_size_mismatch():
    """Mixed sizes don't merge into one chain."""
    asm = (
        "_f:\n"
        "        enter   8, 0\n"
        "        mov     dword [ebp - 4], 7\n"
        "        mov     byte [ebp - 5], 7\n"  # different size
        "        mov     dword [ebp - 12], 7\n"
        "        mov     eax, 100\n"
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # The chain breaks at the byte store. The dword stores aren't
    # adjacent (byte breaks them). No 2+ same-size adjacent. Skip.
    assert opt.stats.get("same_imm_store_share_reg", 0) == 0


def test_dead_addr_recompute_basic_with_stores():
    """Struct-array write loop pattern. Two address computes with
    intermediate stores. The second compute is a redundant
    recompute and gets dropped.

    Block 1: computes &pts[i], stores to [eax + 0]
    Block 2: recomputes &pts[i] (DEAD), stores to [eax + 4]

    After: only block 1's compute survives; block 2's store uses
    the same EAX with offset 4.
    """
    asm = (
        "_f:\n"
        "        enter   16, 0\n"
        ".L_top:\n"
        "        mov     ecx, [ebp - 4]\n"     # loop top: ECX = i
        "        cmp     ecx, [ebp + 12]\n"
        "        jge     .L_end\n"
        # Block 1: pts[i].x = i
        "        mov     eax, [ebp + 8]\n"     # A1
        "        imul    ecx, ecx, 12\n"       # C1
        "        add     eax, ecx\n"           # D1
        "        mov     ecx, [ebp - 4]\n"     # rhs1: load i
        "        mov     [eax], ecx\n"         # STORE1
        # Block 2: pts[i].y = 2*i
        "        mov     eax, [ebp + 8]\n"     # A2 (DROP)
        "        imul    ecx, ecx, 12\n"       # C2 (DROP)
        "        add     eax, ecx\n"           # D2 (DROP)
        "        mov     ecx, [ebp - 4]\n"     # rhs2
        "        add     ecx, ecx\n"
        "        mov     [eax + 4], ecx\n"     # STORE2
        "        inc     dword [ebp - 4]\n"
        "        jmp     .L_top\n"
        ".L_end:\n"
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # Only ONE `mov eax, [ebp + 8]` (the first compute).
    assert out.count("mov     eax, [ebp + 8]") == 1
    # The pass should fire 3 times (3-line recompute).
    assert opt.stats.get("dead_addr_recompute", 0) == 3


def test_dead_addr_recompute_4line_with_explicit_g():
    """When R2 is dirty between the two computes (e.g., post-rhs
    has scaled value), the second compute uses an explicit G to
    reload the index. The 4-line recompute is still droppable.
    """
    asm = (
        "_f:\n"
        "        enter   16, 0\n"
        ".L_top:\n"
        "        mov     ecx, [ebp - 4]\n"     # ECX = i
        "        cmp     ecx, [ebp + 12]\n"
        "        jge     .L_end\n"
        "        mov     eax, [ebp + 8]\n"     # A1
        "        imul    ecx, ecx, 12\n"       # C1
        "        add     eax, ecx\n"           # D1
        "        mov     ecx, [ebp - 4]\n"     # rhs1
        "        add     ecx, ecx\n"           # ECX = 2*i (DIRTY)
        "        mov     [eax], ecx\n"         # STORE1
        # Second compute: 4-line with explicit G (reload i)
        "        mov     eax, [ebp + 8]\n"     # A2 (DROP)
        "        mov     ecx, [ebp - 4]\n"     # G2 (DROP — reload i)
        "        imul    ecx, ecx, 12\n"       # C2 (DROP)
        "        add     eax, ecx\n"           # D2 (DROP)
        "        mov     ecx, [ebp - 4]\n"     # rhs2
        "        imul    ecx, 3\n"
        "        mov     [eax + 4], ecx\n"     # STORE2
        "        inc     dword [ebp - 4]\n"
        "        jmp     .L_top\n"
        ".L_end:\n"
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # Only ONE `mov eax, [ebp + 8]`
    assert out.count("mov     eax, [ebp + 8]") == 1
    # The pass should fire 4 times (4-line recompute dropped).
    assert opt.stats.get("dead_addr_recompute", 0) == 4


def test_dead_addr_recompute_three_blocks():
    """Three consecutive compute+store blocks — block 2 and block 3
    are both redundant."""
    asm = (
        "_f:\n"
        "        enter   16, 0\n"
        ".L_top:\n"
        "        mov     ecx, [ebp - 4]\n"     # loop top
        "        cmp     ecx, [ebp + 12]\n"
        "        jge     .L_end\n"
        "        mov     eax, [ebp + 8]\n"     # A1
        "        imul    ecx, ecx, 12\n"       # C1
        "        add     eax, ecx\n"           # D1
        "        mov     ecx, [ebp - 4]\n"
        "        mov     [eax], ecx\n"
        "        mov     eax, [ebp + 8]\n"     # A2 (DROP)
        "        imul    ecx, ecx, 12\n"       # C2 (DROP)
        "        add     eax, ecx\n"           # D2 (DROP)
        "        mov     ecx, [ebp - 4]\n"
        "        add     ecx, ecx\n"
        "        mov     [eax + 4], ecx\n"
        "        mov     eax, [ebp + 8]\n"     # A3 (DROP)
        "        mov     ecx, [ebp - 4]\n"     # G3 (DROP)
        "        imul    ecx, ecx, 12\n"       # C3 (DROP)
        "        add     eax, ecx\n"           # D3 (DROP)
        "        mov     ecx, [ebp - 4]\n"
        "        imul    ecx, 3\n"
        "        mov     [eax + 8], ecx\n"
        "        inc     dword [ebp - 4]\n"
        "        jmp     .L_top\n"
        ".L_end:\n"
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # Only ONE `mov eax, [ebp + 8]`
    assert out.count("mov     eax, [ebp + 8]") == 1
    # 3 + 4 = 7 lines dropped
    assert opt.stats.get("dead_addr_recompute", 0) == 7


def test_dead_addr_recompute_skips_when_r1_modified():
    """If R1 is modified between the two computes (e.g., via direct
    `mov eax, X` reassignment), the recompute is NOT redundant —
    the second compute is needed."""
    asm = (
        "_f:\n"
        "        enter   16, 0\n"
        ".L_top:\n"
        "        mov     ecx, [ebp - 4]\n"
        "        mov     eax, [ebp + 8]\n"     # A1
        "        imul    ecx, ecx, 12\n"       # C1
        "        add     eax, ecx\n"           # D1
        "        mov     [eax], ecx\n"
        "        mov     eax, 999\n"           # R1 modified! Bail.
        "        mov     [ebp - 8], eax\n"
        "        mov     eax, [ebp + 8]\n"     # A2 (KEEP)
        "        imul    ecx, ecx, 12\n"       # C2 (KEEP)
        "        add     eax, ecx\n"           # D2 (KEEP)
        "        mov     [eax + 4], ecx\n"
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("dead_addr_recompute", 0) == 0
    # Both `mov eax, [ebp + 8]` survive
    assert out.count("mov     eax, [ebp + 8]") == 2


def test_dead_addr_recompute_skips_across_label():
    """Basic block boundary: a label between the two computes
    breaks equivalence (R1 might be re-entered with different
    value)."""
    asm = (
        "_f:\n"
        "        enter   16, 0\n"
        "        mov     ecx, [ebp - 4]\n"
        "        mov     eax, [ebp + 8]\n"     # A1
        "        imul    ecx, ecx, 12\n"       # C1
        "        add     eax, ecx\n"           # D1
        "        mov     [eax], ecx\n"
        ".L_inner:\n"                          # label breaks BB
        "        mov     eax, [ebp + 8]\n"     # A2 (KEEP)
        "        imul    ecx, ecx, 12\n"       # C2 (KEEP)
        "        add     eax, ecx\n"           # D2 (KEEP)
        "        mov     [eax + 4], ecx\n"
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("dead_addr_recompute", 0) == 0
    assert out.count("mov     eax, [ebp + 8]") == 2


def test_dead_addr_recompute_skips_when_mem1_changed():
    """If MEM1's slot is written between the two computes, the
    second compute reads a different value — NOT redundant."""
    asm = (
        "_f:\n"
        "        enter   16, 0\n"
        ".L_top:\n"
        "        mov     ecx, [ebp - 4]\n"
        "        mov     eax, [ebp + 8]\n"     # A1
        "        imul    ecx, ecx, 12\n"
        "        add     eax, ecx\n"
        "        mov     [eax], ecx\n"
        "        mov     dword [ebp + 8], 0\n"  # MEM1 modified!
        "        mov     eax, [ebp + 8]\n"     # A2 (KEEP)
        "        imul    ecx, ecx, 12\n"
        "        add     eax, ecx\n"
        "        mov     [eax + 4], ecx\n"
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("dead_addr_recompute", 0) == 0


def test_dead_addr_recompute_skips_when_mem2_changed():
    """If MEM2's slot is written between, the second compute reads
    a different index — NOT redundant."""
    asm = (
        "_f:\n"
        "        enter   16, 0\n"
        ".L_top:\n"
        "        mov     ecx, [ebp - 4]\n"
        "        mov     eax, [ebp + 8]\n"     # A1
        "        imul    ecx, ecx, 12\n"
        "        add     eax, ecx\n"
        "        mov     [eax], ecx\n"
        "        inc     dword [ebp - 4]\n"     # MEM2 modified!
        "        mov     eax, [ebp + 8]\n"     # A2 (KEEP)
        "        mov     ecx, [ebp - 4]\n"
        "        imul    ecx, ecx, 12\n"
        "        add     eax, ecx\n"
        "        mov     [eax + 4], ecx\n"
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("dead_addr_recompute", 0) == 0


def test_dead_addr_recompute_with_shl():
    """Power-of-two struct size uses shl instead of imul."""
    asm = (
        "_f:\n"
        "        enter   16, 0\n"
        ".L_top:\n"
        "        mov     ecx, [ebp - 4]\n"
        "        mov     eax, [ebp + 8]\n"     # A1
        "        shl     ecx, 4\n"             # C1 (struct size 16)
        "        add     eax, ecx\n"           # D1
        "        mov     ecx, [ebp - 4]\n"
        "        mov     [eax], ecx\n"
        "        mov     eax, [ebp + 8]\n"     # A2 (DROP)
        "        shl     ecx, 4\n"             # C2 (DROP — wait, ECX was just reloaded as i, so this is i*16 again)
        "        add     eax, ecx\n"           # D2 (DROP)
        "        mov     ecx, [ebp - 4]\n"
        "        mov     [eax + 4], ecx\n"
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("dead_addr_recompute", 0) == 3


def test_dead_addr_recompute_skips_indirect_write_to_addr_taken():
    """If MEM1's ebp-offset is address-taken, an indirect write
    through a register-base operand could alias and so we bail."""
    asm = (
        "_f:\n"
        "        enter   16, 0\n"
        "        lea     edx, [ebp + 8]\n"     # ebp+8 is addr-taken
        ".L_top:\n"
        "        mov     ecx, [ebp - 4]\n"
        "        mov     eax, [ebp + 8]\n"     # A1
        "        imul    ecx, ecx, 12\n"
        "        add     eax, ecx\n"
        "        mov     [edx], 999\n"          # indirect write, may alias [ebp + 8]
        "        mov     eax, [ebp + 8]\n"     # A2 (KEEP — alias possible)
        "        imul    ecx, ecx, 12\n"
        "        add     eax, ecx\n"
        "        mov     [eax + 4], ecx\n"
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("dead_addr_recompute", 0) == 0


def test_dead_addr_recompute_allows_indirect_write_when_not_addr_taken():
    """If neither MEM1 nor MEM2 is address-taken, indirect stores
    via [eax] etc. can't alias them — pass should fire."""
    asm = (
        "_f:\n"
        "        enter   16, 0\n"
        ".L_top:\n"
        "        mov     ecx, [ebp - 4]\n"
        "        mov     eax, [ebp + 8]\n"     # A1
        "        imul    ecx, ecx, 12\n"
        "        add     eax, ecx\n"
        "        mov     ecx, [ebp - 4]\n"
        "        mov     [eax], ecx\n"          # store via eax (not addr-taken alias)
        "        mov     eax, [ebp + 8]\n"     # A2 (DROP)
        "        imul    ecx, ecx, 12\n"
        "        add     eax, ecx\n"
        "        mov     [eax + 4], ecx\n"
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("dead_addr_recompute", 0) == 3


def test_redundant_movsx_after_cmp():
    """The strncmp-style pattern: load + cmp + jcc + reload (where
    the reload is redundant because the registers still hold their
    loaded values on the not-taken path).
    """
    asm = (
        "_f:\n"
        "        enter   8, 0\n"
        "        movsx   eax, byte [ebp - 4]\n"
        "        movsx   ecx, byte [ebp - 8]\n"
        "        cmp     eax, ecx\n"
        "        je      .L_eq\n"
        "        movsx   eax, byte [ebp - 4]\n"  # REDUNDANT
        "        movsx   ecx, byte [ebp - 8]\n"  # REDUNDANT
        "        sub     eax, ecx\n"
        "        leave\n"
        "        ret\n"
        ".L_eq:\n"
        "        xor     eax, eax\n"
        "        leave\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # Both `movsx eax, byte [ebp - 4]` collapse to one occurrence.
    assert out.count("movsx   eax, byte [ebp - 4]") == 1
    # Both `movsx ecx, byte [ebp - 8]` collapse to one occurrence.
    assert out.count("movsx   ecx, byte [ebp - 8]") == 1


def test_redundant_movsx_word():
    """movsx with `word` size also tracked."""
    asm = (
        "_f:\n"
        "        movsx   eax, word [ebp - 4]\n"
        "        cmp     eax, 100\n"
        "        je      .L_eq\n"
        "        movsx   eax, word [ebp - 4]\n"  # REDUNDANT
        "        sub     eax, 1\n"
        "        ret\n"
        ".L_eq:\n"
        "        xor     eax, eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert out.count("movsx   eax, word [ebp - 4]") == 1


def test_redundant_movzx_byte():
    """movzx tracked separately from movsx — different operations."""
    asm = (
        "_f:\n"
        "        movzx   eax, byte [ebp - 4]\n"
        "        test    eax, eax\n"
        "        je      .L_zero\n"
        "        movzx   eax, byte [ebp - 4]\n"  # REDUNDANT
        "        ret\n"
        ".L_zero:\n"
        "        xor     eax, eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert out.count("movzx   eax, byte [ebp - 4]") == 1


def test_redundant_movsx_different_size_not_collapsed():
    """movsx byte and movsx word on the same memory are different
    operations — second must be preserved (not dropped as redundant).
    """
    asm = (
        "_f:\n"
        "        movsx   eax, byte [ebp - 4]\n"
        "        movsx   ecx, byte [ebp - 5]\n"
        "        cmp     eax, ecx\n"
        "        je      .L_zero\n"
        "        movsx   eax, word [ebp - 4]\n"  # NOT redundant — different size
        "        sub     eax, 1\n"
        "        ret\n"
        ".L_zero:\n"
        "        xor     eax, eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # The byte and word movsx are different ops — the word one
    # must survive (not dropped by my pass since it's different).
    # Other passes may modify but my pass shouldn't drop it.
    assert "movsx   eax, word [ebp - 4]" in out


def test_redundant_mov_then_movsx_not_collapsed():
    """mov dword and movsx byte on the same memory are different
    operations — second must NOT be dropped by my pass."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp - 4]\n"
        "        mov     ecx, [ebp - 8]\n"
        "        cmp     eax, ecx\n"
        "        je      .L_zero\n"
        "        movsx   eax, byte [ebp - 4]\n"  # NOT redundant — different op
        "        ret\n"
        ".L_zero:\n"
        "        xor     eax, eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # The movsx should survive — different op from the mov.
    assert "movsx   eax, byte [ebp - 4]" in out


def test_redundant_movsx_invalidated_by_mem_write():
    """If the source memory is written between the two movsx loads,
    the second is NOT redundant (different value)."""
    asm = (
        "_f:\n"
        "        movsx   eax, byte [ebp - 4]\n"
        "        mov     [ebp - 8], eax\n"        # save first value
        "        mov     byte [ebp - 4], 99\n"     # writes the slot
        "        movsx   eax, byte [ebp - 4]\n"   # reload — NOT redundant
        "        add     eax, [ebp - 8]\n"         # use both
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # Both movsx survive (the mem write invalidates the cached state)
    assert out.count("movsx   eax, byte [ebp - 4]") == 2


def test_redundant_movsx_invalidated_by_reg_write():
    """If EAX is written to a different value between the two movsx
    loads, the second is NOT a textual duplicate of the first
    (the cached state was reset)."""
    asm = (
        "_f:\n"
        "        movsx   eax, byte [ebp - 4]\n"
        "        mov     [ebp - 12], eax\n"  # use first value
        "        add     eax, eax\n"          # EAX clobbered (RMW)
        "        mov     [ebp - 16], eax\n"
        "        movsx   eax, byte [ebp - 4]\n"  # reload — NOT redundant
        "        add     eax, [ebp - 16]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert out.count("movsx   eax, byte [ebp - 4]") == 2


def test_redundant_movzx_for_ecx():
    """The ECX tracker also handles movzx."""
    asm = (
        "_f:\n"
        "        movzx   ecx, byte [ebp - 4]\n"
        "        cmp     ecx, 0\n"
        "        je      .L_zero\n"
        "        movzx   ecx, byte [ebp - 4]\n"  # REDUNDANT
        "        sub     ecx, 1\n"
        "        ret\n"
        ".L_zero:\n"
        "        xor     ecx, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert out.count("movzx   ecx, byte [ebp - 4]") == 1


def test_self_extension_movsx_after_movsx_byte():
    """The strcpy idiom: `movsx eax, byte [m]` (or `[reg]`) sets
    EAX = sign-extend(byte). A subsequent `movsx eax, al` is
    redundant — AL is already that byte's value, sign-extending
    gives the same EAX value."""
    asm = (
        "_f:\n"
        "        movsx   eax, byte [eax]\n"     # non-ebp source
        "        mov     byte [ebp - 4], al\n"   # store AL elsewhere
        "        movsx   eax, al\n"              # REDUNDANT
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "movsx   eax, al" not in out


def test_self_extension_movzx():
    """Same pattern for movzx."""
    asm = (
        "_f:\n"
        "        movzx   eax, byte [eax]\n"
        "        mov     byte [ebp - 4], al\n"
        "        movzx   eax, al\n"              # REDUNDANT
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "movzx   eax, al" not in out


def test_self_extension_invalidated_by_al_write():
    """If AL is written between the movsx and the self-extension,
    the property is broken — the second movsx is NOT redundant."""
    asm = (
        "_f:\n"
        "        movsx   eax, byte [eax]\n"
        "        mov     al, 5\n"             # AL clobbered
        "        movsx   eax, al\n"           # NOT redundant
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "movsx   eax, al" in out


def test_self_extension_movsx_after_movzx_byte_not_dropped():
    """movsx eax, al is NOT redundant after movzx eax, byte [m] —
    sign vs zero extension gives different values for negative
    bytes."""
    asm = (
        "_f:\n"
        "        movzx   eax, byte [eax]\n"   # zero-extends
        "        mov     byte [ebp - 4], al\n"
        "        movsx   eax, al\n"           # NOT redundant — different op
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # The movsx is preserved (different op from movzx)
    assert "movsx   eax, al" in out


def test_self_extension_movsx_after_movsx_word():
    """movsx eax, ax is redundant after movsx eax, word [m]."""
    asm = (
        "_f:\n"
        "        movsx   eax, word [eax]\n"
        "        mov     word [ebp - 4], ax\n"
        "        movsx   eax, ax\n"           # REDUNDANT
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "movsx   eax, ax" not in out


def test_uncollapse_cmp_when_reload_movsx_byte():
    """`cmp byte [m], 0; jz; movsx eax, byte [m]` is rewritten to
    `movsx eax, byte [m]; test eax, eax; jz`. Saves 2 bytes
    (cmp byte 4 → test 2; movsx position unchanged).
    """
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        cmp     byte [eax], 0\n"
        "        jz      .L_zero\n"
        "        movsx   eax, byte [eax]\n"
        "        ret\n"
        ".L_zero:\n"
        "        xor     eax, eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # The byte cmp is gone; replaced with movsx + test
    assert "cmp     byte [eax], 0" not in out
    assert "movsx   eax, byte [eax]" in out
    assert "test    eax, eax" in out


def test_uncollapse_cmp_when_reload_movzx_byte():
    """movzx version: only safe with ZF-only Jcc (since movzx's
    SF differs from byte cmp's SF for negative bytes).
    """
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        cmp     byte [eax], 0\n"
        "        jz      .L_zero\n"        # ZF-only Jcc
        "        movzx   eax, byte [eax]\n"
        "        ret\n"
        ".L_zero:\n"
        "        xor     eax, eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # The byte cmp is gone; replaced with movzx + test
    assert "cmp     byte [eax], 0" not in out
    assert "movzx   eax, byte [eax]" in out
    assert "test    eax, eax" in out


def test_uncollapse_cmp_when_reload_movzx_with_signed_jcc_skipped():
    """movzx with non-ZF Jcc (e.g., jl/jg) — SF differs, must NOT
    rewrite."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        cmp     byte [eax], 0\n"
        "        jl      .L_neg\n"         # SF-reading Jcc
        "        movzx   eax, byte [eax]\n"
        "        ret\n"
        ".L_neg:\n"
        "        xor     eax, eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # The original cmp byte is preserved
    assert "cmp     byte [eax], 0" in out


def test_uncollapse_cmp_when_reload_byte_nonzero_skipped():
    """Non-zero immediate operand for byte cmp — different signed
    interpretation than 32-bit cmp, must NOT rewrite."""
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        cmp     byte [eax], 5\n"   # non-zero
        "        je      .L_match\n"
        "        movsx   eax, byte [eax]\n"
        "        ret\n"
        ".L_match:\n"
        "        xor     eax, eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # Original cmp preserved
    assert "cmp     byte [eax], 5" in out


def test_trampoline_elimination_basic():
    """`jcc L1; ...; L1: jmp L3` — redirect jcc to L3 directly,
    drop L1: jmp L3 trampoline. Realistic CFG."""
    asm = (
        "_f:\n"
        ".L_loop:\n"
        "        mov     eax, [ebp - 4]\n"
        "        cmp     eax, 0\n"
        "        je      .L_target\n"     # redirect target
        "        add     eax, 1\n"
        "        mov     [ebp - 4], eax\n"
        "        jmp     .L_loop\n"
        ".L_target:\n"
        "        jmp     .L_loop\n"        # trampoline back to loop top
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # je should now go to .L_loop directly via trampoline elimination.
    assert opt.stats.get("trampoline_elimination", 0) >= 1


def test_trampoline_elimination_skips_when_fallthrough():
    """If L1 has fallthrough from prior code, the trampoline can't
    be safely eliminated (the fallthrough path would change)."""
    asm = (
        "_f:\n"
        "        cmp     eax, 0\n"
        "        je      .L_target\n"
        # No terminator before .L_target — fallthrough applies
        ".L_target:\n"
        "        jmp     .L_real_dest\n"
        ".L_real_dest:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # No trampoline elimination should fire
    assert opt.stats.get("trampoline_elimination", 0) == 0


def test_trampoline_elimination_skips_self_target():
    """No infinite-loop trampoline (L1: jmp L1)."""
    asm = (
        "_f:\n"
        "        ret\n"
        ".L_loop:\n"
        "        jmp     .L_loop\n"  # self-trampoline; bail
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("trampoline_elimination", 0) == 0


def test_trampoline_elimination_global_label_safe():
    """Global labels (no leading `.`) are not considered for
    elimination — they may be exported."""
    asm = (
        "_f:\n"
        "        ret\n"
        "_global_target:\n"
        "        jmp     .somewhere\n"
        ".somewhere:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # Global trampolines are NOT eliminated.
    assert opt.stats.get("trampoline_elimination", 0) == 0


def test_chain_binop_collapse_add():
    """`mov ecx, X; add ecx, Y; add eax, ecx` → `add eax, X; add eax, Y`."""
    asm = (
        "_f:\n"
        "        mov     ecx, [ebp - 4]\n"
        "        add     ecx, [ebp - 8]\n"
        "        add     eax, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("chain_binop_collapse", 0) == 1
    assert "mov     ecx" not in out
    assert "add     eax, [ebp - 4]" in out
    assert "add     eax, [ebp - 8]" in out


def test_chain_binop_collapse_skips_sub():
    """sub is NOT associative under this rewrite —
    `eax - (X - Y) ≠ (eax - X) - Y`. Don't fire."""
    asm = (
        "_f:\n"
        "        mov     ecx, [ebp - 4]\n"
        "        sub     ecx, [ebp - 8]\n"
        "        sub     eax, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("chain_binop_collapse", 0) == 0


def test_chain_binop_collapse_skips_mismatched_ops():
    """B and C must use the SAME op."""
    asm = (
        "_f:\n"
        "        mov     ecx, [ebp - 4]\n"
        "        add     ecx, [ebp - 8]\n"
        "        xor     eax, ecx\n"  # diff op
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("chain_binop_collapse", 0) == 0


def test_chain_binop_collapse_skips_y_references_eax():
    """If Y references EAX, the rewrite changes its observed value."""
    asm = (
        "_f:\n"
        "        mov     ecx, [ebp - 4]\n"
        "        add     ecx, [eax + 4]\n"  # Y references EAX
        "        add     eax, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("chain_binop_collapse", 0) == 0


def test_chain_binop_collapse_skips_y_references_ecx():
    """If Y references ECX, the original reads ECX with A's value
    but the rewrite would read with pre-A ECX. Don't fire."""
    asm = (
        "_f:\n"
        "        mov     ecx, [ebp - 4]\n"
        "        add     ecx, [ecx + 4]\n"  # Y references ECX
        "        add     eax, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("chain_binop_collapse", 0) == 0


def test_chain_binop_collapse_skips_ecx_alive_after():
    """If ECX is read after C, dropping the only definition is
    unsafe."""
    asm = (
        "_f:\n"
        "        mov     ecx, [ebp - 4]\n"
        "        add     ecx, [ebp - 8]\n"
        "        add     eax, ecx\n"
        "        mov     [ebp - 12], ecx\n"  # ECX read here
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("chain_binop_collapse", 0) == 0


def test_chain_binop_collapse_imul():
    """imul is associative — chain rewrite is valid."""
    asm = (
        "_f:\n"
        "        mov     ecx, [ebp - 4]\n"
        "        imul    ecx, [ebp - 8]\n"
        "        imul    eax, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("chain_binop_collapse", 0) == 1
    assert "imul    eax, [ebp - 4]" in out
    assert "imul    eax, [ebp - 8]" in out


def test_pop_cmp_chain_retarget_basic():
    """`push eax; mov eax, N; shl eax, K; pop ecx; cmp eax, ecx;
    je L` — drop push/pop, retarget chain to ECX. ZF symmetric so
    subsequent je is safe."""
    asm = (
        "_f:\n"
        "        push    eax\n"
        "        mov     eax, 5\n"
        "        shl     eax, 3\n"
        "        pop     ecx\n"
        "        cmp     eax, ecx\n"
        "        je      .L_eq\n"
        "        ret\n"
        ".L_eq:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("pop_cmp_chain_retarget", 0) == 1
    # Chain retargeted to ECX
    assert "mov     ecx, 5" in out
    assert "shl     ecx, 3" in out
    # push and pop dropped
    assert "push    eax" not in out
    assert "pop     ecx" not in out


def test_pop_cmp_chain_retarget_skips_non_zf_jcc():
    """`jl/jg/jb/...` consume SF/CF/OF which aren't symmetric under
    cmp operand swap. Don't fire."""
    asm = (
        "_f:\n"
        "        push    eax\n"
        "        mov     eax, 5\n"
        "        shl     eax, 3\n"
        "        pop     ecx\n"
        "        cmp     eax, ecx\n"
        "        jl      .L_lt\n"
        "        ret\n"
        ".L_lt:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("pop_cmp_chain_retarget", 0) == 0


def test_pop_cmp_chain_retarget_jne():
    """jne is also ZF-only — fires."""
    asm = (
        "_f:\n"
        "        push    eax\n"
        "        mov     eax, 100\n"
        "        sar     eax, 2\n"
        "        pop     ecx\n"
        "        cmp     eax, ecx\n"
        "        jne     .L_ne\n"
        "        ret\n"
        ".L_ne:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("pop_cmp_chain_retarget", 0) == 1


def test_pop_cmp_chain_retarget_skips_chain_with_ecx_read():
    """Chain mustn't read ECX (else retarget would self-reference)."""
    asm = (
        "_f:\n"
        "        push    eax\n"
        "        mov     eax, [ecx]\n"  # reads ECX
        "        shl     eax, 1\n"
        "        pop     ecx\n"
        "        cmp     eax, ecx\n"
        "        je      .L_eq\n"
        "        ret\n"
        ".L_eq:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("pop_cmp_chain_retarget", 0) == 0


def test_pop_cmp_chain_retarget_skips_non_fresh_chain_first():
    """Chain's first instr must be a fresh EAX write (not RMW).
    `add eax, X` reads EAX's prior value (the saved LHS), so post-
    retarget the chain would compute differently."""
    asm = (
        "_f:\n"
        "        push    eax\n"
        "        add     eax, 5\n"  # RMW — reads prior eax
        "        pop     ecx\n"
        "        cmp     eax, ecx\n"
        "        je      .L_eq\n"
        "        ret\n"
        ".L_eq:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("pop_cmp_chain_retarget", 0) == 0


def test_rmw_collapse_edx():
    """rmw_collapse fires for non-EAX working registers (EDX, ECX, ...).
    The 3-line load+op+store on any GP scratch reg collapses to
    `OP [m], src`."""
    asm = (
        "_f:\n"
        "        mov     edx, [ebp - 4]\n"
        "        and     edx, 4294967239\n"
        "        mov     [ebp - 4], edx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("rmw_collapse", 0) == 1
    assert "and     dword [ebp - 4], 4294967239" in out


def test_rmw_collapse_ecx():
    """ECX as working reg also fires."""
    asm = (
        "_f:\n"
        "        mov     ecx, [ebp - 8]\n"
        "        or      ecx, 16\n"
        "        mov     [ebp - 8], ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("rmw_collapse", 0) == 1
    assert "or      dword [ebp - 8], 16" in out


def test_rmw_collapse_edx_with_reg_src_eax():
    """EDX as working reg, EAX as source — EAX-source is allowed since
    working reg is EDX (not the same)."""
    asm = (
        "_f:\n"
        "        mov     edx, [ebp - 4]\n"
        "        or      edx, eax\n"
        "        mov     [ebp - 4], edx\n"
        "        xor     eax, eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("rmw_collapse", 0) == 1
    assert "or      dword [ebp - 4], eax" in out


def test_rmw_collapse_skips_self_referential_mem():
    """If memory operand references the working register (e.g.,
    `[edx + 4]`), the rewrite is unsafe — after dropping the load,
    EDX would have a different value."""
    asm = (
        "_f:\n"
        "        mov     edx, [edx + 4]\n"
        "        add     edx, 5\n"
        "        mov     [edx + 4], edx\n"
        "        xor     eax, eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("rmw_collapse", 0) == 0


def test_rmw_collapse_skips_when_src_is_working_reg():
    """`mov edx, [m]; add edx, edx; mov [m], edx` — src can't be edx
    itself (its value is the load result, which we drop)."""
    asm = (
        "_f:\n"
        "        mov     edx, [ebp - 4]\n"
        "        add     edx, edx\n"  # src is working reg — bail
        "        mov     [ebp - 4], edx\n"
        "        xor     eax, eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("rmw_collapse", 0) == 0


def test_mov_neg_one_to_or_basic():
    """`mov reg, -1` → `or reg, -1`. Saves 2 bytes (5-byte
    mov-imm32 → 3-byte or imm8 sign-extended). Flags must be
    dead after."""
    asm = (
        "_f:\n"
        "        mov     eax, -1\n"
        "        ret\n"  # ret is a flag fence
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("mov_neg_one_to_or", 0) == 1
    assert "or      eax, -1" in out


def test_mov_neg_one_to_or_unsigned_form():
    """The codegen emits `mov reg, 4294967295` for unsigned-cast
    -1. Same rewrite applies."""
    asm = (
        "_f:\n"
        "        mov     edx, 4294967295\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("mov_neg_one_to_or", 0) == 1
    assert "or      edx, -1" in out


def test_mov_neg_one_to_or_skips_if_flags_read():
    """If flags are read after the mov before being clobbered,
    the rewrite is unsafe — `or` clears OF/CF and sets SF/ZF.
    `mov` preserves flags."""
    asm = (
        "_f:\n"
        "        cmp     eax, ecx\n"  # flags set
        "        mov     edx, -1\n"   # mov preserves flags
        "        je      .L_eq\n"     # reads flags
        "        ret\n"
        ".L_eq:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("mov_neg_one_to_or", 0) == 0


def test_mov_neg_one_to_or_skips_non_neg_one_value():
    """Only -1 (or 4294967295) — not 0 or other constants."""
    asm = (
        "_f:\n"
        "        mov     eax, 5\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("mov_neg_one_to_or", 0) == 0
    asm2 = (
        "_f:\n"
        "        mov     eax, -2\n"
        "        ret\n"
    )
    opt2 = PeepholeOptimizer()
    out2 = opt2.optimize(asm2)
    assert opt2.stats.get("mov_neg_one_to_or", 0) == 0


def test_mov_neg_one_to_or_fires_with_arithmetic_after():
    """Flag-clobbering arithmetic after the mov is a fence.
    The new `or`'s flags are clobbered before being read."""
    asm = (
        "_f:\n"
        "        mov     eax, -1\n"
        "        add     eax, 5\n"  # clobbers flags
        "        je      .L_eq\n"   # reads flags from add, not mov
        "        ret\n"
        ".L_eq:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("mov_neg_one_to_or", 0) == 1


def test_bool_materialize_collapse_basic():
    """`xor eax, eax; jmp L_end; .L_true: mov eax, 1; .L_end: test
    eax, eax; jnz .L_target` collapses to direct jcc redirects."""
    asm = (
        "_main:\n"
        "        cmp     eax, 1\n"
        "        jne     .L3_or_true\n"
        "        xor     eax, eax\n"
        "        jmp     .L4_or_end\n"
        ".L3_or_true:\n"
        "        mov     eax, 1\n"
        ".L4_or_end:\n"
        "        test    eax, eax\n"
        "        jnz     .L_target\n"
        "        push    42\n"           # EAX clobbered after
        "        call    _foo\n"
        "        pop     ecx\n"
        "        xor     eax, eax\n"
        "        ret\n"
        ".L_target:\n"
        "        call    _abort\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("bool_materialize_collapse", 0) == 1
    # The xor/jmp/L_or_true/mov/L_or_end/test/jnz should all be gone
    assert ".L3_or_true:" not in out
    assert ".L4_or_end:" not in out
    assert "test    eax, eax" not in out
    # The original `jne .L3_or_true` should now go to .L_target
    assert "jne     .L_target" in out


def test_bool_materialize_collapse_skips_eax_live_after():
    """If EAX is alive after the consumer's not-taken path,
    don't collapse (we'd change EAX's value)."""
    asm = (
        "_main:\n"
        "        cmp     eax, 1\n"
        "        jne     .L3_or_true\n"
        "        xor     eax, eax\n"
        "        jmp     .L4_or_end\n"
        ".L3_or_true:\n"
        "        mov     eax, 1\n"
        ".L4_or_end:\n"
        "        test    eax, eax\n"
        "        jnz     .L_target\n"
        "        ret\n"  # EAX is the return value — alive
        ".L_target:\n"
        "        call    _abort\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("bool_materialize_collapse", 0) == 0


def test_bool_materialize_collapse_jz_consumer():
    """jz consumer also collapses, with a synthetic skip label and
    `jmp .L_target` inserted at the dropped block's location."""
    asm = (
        "_main:\n"
        "        cmp     eax, 1\n"
        "        jne     .L3_or_true\n"
        "        xor     eax, eax\n"
        "        jmp     .L4_or_end\n"
        ".L3_or_true:\n"
        "        mov     eax, 1\n"
        ".L4_or_end:\n"
        "        test    eax, eax\n"
        "        jz      .L_target\n"  # jz consumer
        "        push    42\n"
        "        call    _foo\n"
        "        pop     ecx\n"
        "        xor     eax, eax\n"
        "        ret\n"
        ".L_target:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("bool_materialize_collapse", 0) == 1
    # Block is dropped; replaced with jmp .L_target + a skip label.
    # The original jne path now jumps past the new jmp to fall through.
    assert ".L3_or_true:" not in out
    assert ".L4_or_end:" not in out
    assert "test    eax, eax" not in out
    # A `jmp .L_target` is inserted; the .bm_skip_N label may further
    # cascade with jcc_jmp_inversion. After cascade, the jne becomes
    # `je .L_target` (inverted to point at target, since jne to skip
    # followed by jmp to target inverts to je).
    assert "je      .L_target" in out


def test_bool_materialize_collapse_multiple_jccs_to_true():
    """Multiple jcc references to .L_or_true (typical short-circuit
    || pattern with multiple operands) all redirect to .L_target."""
    asm = (
        "_main:\n"
        "        cmp     eax, 1\n"
        "        jne     .L3_or_true\n"   # first short-circuit
        "        cmp     eax, 5\n"
        "        jne     .L3_or_true\n"   # second short-circuit
        "        xor     eax, eax\n"
        "        jmp     .L4_or_end\n"
        ".L3_or_true:\n"
        "        mov     eax, 1\n"
        ".L4_or_end:\n"
        "        test    eax, eax\n"
        "        jnz     .L_target\n"
        "        push    42\n"
        "        call    _foo\n"
        "        pop     ecx\n"
        "        xor     eax, eax\n"
        "        ret\n"
        ".L_target:\n"
        "        call    _abort\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("bool_materialize_collapse", 0) == 1
    # Both jne's should now point to .L_target
    assert out.count("jne     .L_target") == 2
    assert ".L3_or_true:" not in out


def test_bool_materialize_collapse_form_b_jz_consumer():
    """Form B: LL-cmp variant where mov-1 comes BEFORE xor.

    Pattern (with `xor edx, edx` LL-cmp tail):
        cmp ...; jne .L_false
        cmp ...; jne .L_false
        .L_true: mov eax, 1; jmp .L_end
        .L_false: xor eax, eax
        .L_end: xor edx, edx; test eax, eax; jz .L_target

    For jz: result==0 means a jcc fired (some condition failed).
    Each `jne .L_false` retargets to `jz`'s target directly.
    """
    asm = (
        "_main:\n"
        "        cmp     edx, ebx\n"
        "        jne     .L8_ll_cmp_false\n"
        "        cmp     eax, ecx\n"
        "        jne     .L8_ll_cmp_false\n"
        ".L7_ll_cmp_true:\n"
        "        mov     eax, 1\n"
        "        jmp     .L9_ll_cmp_end\n"
        ".L8_ll_cmp_false:\n"
        "        xor     eax, eax\n"
        ".L9_ll_cmp_end:\n"
        "        xor     edx, edx\n"
        "        test    eax, eax\n"
        "        jz      .L6_endif\n"
        "        call    _abort\n"
        ".L6_endif:\n"
        "        xor     eax, eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("bool_materialize_collapse", 0) == 1
    # Both jne's should now point at .L6_endif (jz's target).
    assert out.count("jne     .L6_endif") == 2
    # Materialize block gone.
    assert "mov     eax, 1" not in out
    assert ".L8_ll_cmp_false:" not in out
    assert ".L9_ll_cmp_end:" not in out
    # Optional .L_true label (unreferenced) also dropped.
    assert ".L7_ll_cmp_true:" not in out


def test_bool_materialize_collapse_form_b_jnz_consumer():
    """Form B with jnz consumer: result==1 means all conditions met.

    Insert `jmp .L_target` + skip label; retarget jccs to skip.
    """
    asm = (
        "_main:\n"
        "        cmp     eax, 1\n"
        "        jne     .L_false\n"
        "        cmp     eax, 2\n"
        "        jne     .L_false\n"
        "        mov     eax, 1\n"
        "        jmp     .L_end\n"
        ".L_false:\n"
        "        xor     eax, eax\n"
        ".L_end:\n"
        "        test    eax, eax\n"
        "        jnz     .L_target\n"
        "        xor     eax, eax\n"
        "        ret\n"
        ".L_target:\n"
        "        call    _abort\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("bool_materialize_collapse", 0) == 1
    # Original materialize block labels gone.
    assert ".L_false:" not in out
    assert ".L_end:" not in out
    # After cascade with jcc_jmp_inversion, the second jne+jmp pair
    # collapses to `je .L_target` (taking branch when condition met).
    assert "je      .L_target" in out
    # The first jne goes to the skip label (still present since it's
    # the target of the first short-circuit jcc).
    assert ".bm_skip_" in out


def test_bool_materialize_collapse_form_b_no_edx_clear():
    """Form B without the `xor edx, edx` LL-cmp tail.

    Pattern is otherwise identical to the LL form. Used when the
    short-circuit chain produces a regular int-typed boolean.
    """
    asm = (
        "_main:\n"
        "        cmp     eax, 0\n"
        "        je      .L_false\n"
        "        mov     eax, 1\n"
        "        jmp     .L_end\n"
        ".L_false:\n"
        "        xor     eax, eax\n"
        ".L_end:\n"
        "        test    eax, eax\n"
        "        jz      .L_target\n"
        "        call    _foo\n"
        "        xor     eax, eax\n"
        "        ret\n"
        ".L_target:\n"
        "        call    _abort\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert opt.stats.get("bool_materialize_collapse", 0) == 1
    assert "je      .L_target" in out
    assert ".L_false:" not in out


def test_bool_materialize_collapse_form_b_skips_eax_live():
    """Form B should NOT fire if EAX is live after the consumer."""
    asm = (
        "_main:\n"
        "        cmp     eax, 1\n"
        "        jne     .L_false\n"
        "        mov     eax, 1\n"
        "        jmp     .L_end\n"
        ".L_false:\n"
        "        xor     eax, eax\n"
        ".L_end:\n"
        "        test    eax, eax\n"
        "        jz      .L_target\n"
        "        ret\n"   # EAX is the return value!
        ".L_target:\n"
        "        call    _abort\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    # Should NOT fire — EAX is live (return value).
    assert opt.stats.get("bool_materialize_collapse", 0) == 0


# ── shl_add_reg_to_lea ───────────────────────────────────────────


def test_shl_add_reg_to_lea_basic():
    """`shl ecx, 2; add eax, ecx` → `lea eax, [eax + ecx*4]`.

    Saves 1 instruction (~2 bytes). Sister of shl_add_label_to_lea
    for the case where the second add operand is a register (not a
    label).
    """
    asm = (
        "_f:\n"
        "        mov     eax, [ebp + 8]\n"
        "        mov     ecx, [ebp - 4]\n"
        "        shl     ecx, 2\n"
        "        add     eax, ecx\n"
        "        push    eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "lea     eax, [eax + ecx*4]" in out
    assert "shl     ecx, 2" not in out
    assert "add     eax, ecx" not in out
    assert opt.stats.get("shl_add_reg_to_lea") == 1


def test_shl_add_reg_to_lea_scale_2():
    """N=1 produces SCALE=2."""
    asm = (
        "_f:\n"
        "        shl     ecx, 1\n"
        "        add     eax, ecx\n"
        "        push    eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "lea     eax, [eax + ecx*2]" in out


def test_shl_add_reg_to_lea_scale_8():
    """N=3 produces SCALE=8."""
    asm = (
        "_f:\n"
        "        shl     edx, 3\n"
        "        add     eax, edx\n"
        "        push    eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "lea     eax, [eax + edx*8]" in out


def test_shl_add_reg_to_lea_skips_same_reg():
    """`shl eax, N; add eax, eax` → DST == IDX. Result would be
    different (eax * (2^N + 1) vs eax * (scale + 1) where scale = 2^N).

    Specifically: `shl eax, 2; add eax, eax` = eax*5 (after shl,
    eax = eax*4; then eax = eax*4 + eax*4 = eax*8). Hmm wait that's
    eax*8 not eax*5... wait `add eax, eax` sums TWO copies of the
    SAME current eax = 2*eax. So `shl eax, 2; add eax, eax` =
    eax*4 then eax = (eax*4)*2 = eax*8. The lea form would be
    `lea eax, [eax + eax*4]` = 5*eax. So they differ — SKIP.
    """
    asm = (
        "_f:\n"
        "        shl     eax, 2\n"
        "        add     eax, eax\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("shl_add_reg_to_lea", 0) == 0


def test_shl_add_reg_to_lea_skips_non_matching_idx():
    """The shl IDX register must match the add's source register."""
    asm = (
        "_f:\n"
        "        shl     ecx, 2\n"
        "        add     eax, edx\n"   # source is edx, not ecx
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("shl_add_reg_to_lea", 0) == 0


def test_shl_add_reg_to_lea_skips_when_flags_read():
    """If flags after the add are read (e.g., by jz), skip — lea
    doesn't set flags but add does.
    """
    asm = (
        "_f:\n"
        "        shl     ecx, 2\n"
        "        add     eax, ecx\n"
        "        jz      .L\n"
        "        ret\n"
        ".L:\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("shl_add_reg_to_lea", 0) == 0


def test_shl_add_reg_to_lea_skips_invalid_count():
    """N must be 1, 2, or 3 (lea SCALE bits support only 1/2/4/8)."""
    asm = (
        "_f:\n"
        "        shl     ecx, 4\n"   # SCALE 16 — not encodable
        "        add     eax, ecx\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    opt.optimize(asm)
    assert opt.stats.get("shl_add_reg_to_lea", 0) == 0


def test_shl_add_reg_to_lea_followed_by_dec():
    """Real-world shape from torture: `shl ecx, 2; add eax, ecx;
    dec dword [eax]`. dec sets flags but doesn't read prior flags
    (the lea didn't run yet). After the rewrite, dec sees the same
    flag state it would after lea (= no change), then sets its own
    flags. Safe.
    """
    asm = (
        "_f:\n"
        "        shl     ecx, 2\n"
        "        add     eax, ecx\n"
        "        dec     dword [eax]\n"
        "        ret\n"
    )
    opt = PeepholeOptimizer()
    out = opt.optimize(asm)
    assert "lea     eax, [eax + ecx*4]" in out
    assert opt.stats.get("shl_add_reg_to_lea", 0) == 1
