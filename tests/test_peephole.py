"""Tests for the asm-level peephole optimizer."""

from uc386.peephole import PeepholeOptimizer, optimize


# ── P1: dead-after-terminator ────────────────────────────────────


def test_drops_dead_xor_after_jmp():
    asm = (
        "_main:\n"
        "        mov     eax, 0\n"
        "        jmp     .epilogue\n"
        "        xor     eax, eax\n"
        ".epilogue:\n"
        "        ret\n"
    )
    out = optimize(asm)
    assert "xor     eax, eax" not in out
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
    """The canonical binop right-operand pattern collapses."""
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
    # The 4 stack-machine lines collapse to one `mov ecx, 1`.
    assert "push    eax" not in out
    assert "mov     ecx, eax" not in out
    assert "pop     eax" not in out
    assert "mov     ecx, 1" in out
    # The cmp survives.
    assert "cmp     eax, ecx" in out
    assert opt.stats.get("binop_collapse", 0) == 1


def test_binop_collapse_with_memory_operand():
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
    assert "mov     ecx, [ebp - 8]" in out
    assert opt.stats.get("binop_collapse", 0) == 1


def test_binop_collapse_skips_esp_relative_source():
    """`push eax` shifts ESP, so a subsequent `[esp + N]` read targets
    a different byte after collapse. Don't fire."""
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
    # Pattern must NOT fire — the [esp + N] is push-relative.
    assert opt.stats.get("binop_collapse", 0) == 0
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
    # And the dead xor is gone from the output.
    assert "xor     eax, eax\n.epilogue:" not in asm


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
