"""Peephole optimizer for uc386 NASM assembly output.

Applies pattern-based and structural rewrites on the generated asm text
just before it's written. Runs to fixed point.

This is the inline phase of the optimizer. Once the pattern set has
crystallized, this module will be extracted to a separate `upeep386`
package mirroring the upeepz80/uc80 split.

Currently implements:
  - dead_after_terminator: drop instrs between an unconditional `jmp`/
    `ret` and the next label/directive/data line.
  - jmp_to_next_label: drop `jmp X` immediately followed by `X:`.
  - binop_collapse: replace the 4-line stack-machine right-operand
    transfer (`push eax; mov eax, src; mov ecx, eax; pop eax`) with
    `mov ecx, src` when src is a single-instruction load.
  - store_collapse: drop the push/pop pair around a store (`push eax;
    mov eax, src; pop ecx; mov [ecx], eax`) when src is a
    single-instruction load that doesn't read ECX.
  - leave_collapse: replace the function epilogue's `mov esp, ebp;
    pop ebp` with the equivalent single-byte `leave`. Saves 1 instr
    + 2 bytes per function epilogue.
  - imm_store_collapse: replace `mov eax, IMM; mov [addr], eax;
    mov eax, X` with `mov dword [addr], IMM; mov eax, X`. The
    immediate-store form is one NASM instruction; the EAX overwrite
    in the third line confirms EAX was dead.
  - setcc_jcc_collapse: replace the `setCC al; movzx eax, al; test
    eax, eax; jz/jnz LBL` boolean-materialize-then-branch sequence
    with a single conditional jump using the (possibly inverted) CC.
    Saves 3 instructions per `if`/`while`/`for` condition.
  - push_immediate: replace `mov eax, IMM_OR_LABEL; push eax` with
    `push IMM_OR_LABEL` when the next instruction overwrites EAX
    (the mov was just for the push). Common at every cdecl arg site.
  - ecx_binop_collapse: replace `mov ecx, <src>; OP eax, ecx` with
    `OP eax, <src>` for OP ∈ {add, sub, and, or, xor, cmp, test,
    imul, adc, sbb}. Saves the bytes of the prior `mov ecx, <src>`
    (5 for imm32, 3-7 for memory, 2 for register). Source can be
    any addressing mode the OP supports as a memory or immediate
    source operand. Witness: next instruction overwrites ECX (full
    mov / pop / xor / lea / call) or doesn't reference ECX/CX/CL/CH.
  - mov_zero_to_xor: replace `mov eax, 0` with `xor eax, eax`. Saves
    3 bytes (5-byte mov-imm32 → 2-byte xor reg, reg). Conservative:
    only fires when a forward scan finds a flag-clobbering instruction
    or a `ret` before any flag-reading instruction. Same pattern for
    other gp registers (ECX, EDX, EBX, ESI, EDI, EBP).
  - store_load_collapse: drop the redundant load after a store of
    the same register to the same address: `mov [X], R; mov R, [X]`
    → just the store. The register still holds its stored value, so
    the load is unnecessary. Saves the bytes of the load instruction
    (3 for [ebp-N], 6 for [_glob]).
  - right_operand_retarget: collapse the RHS-chain-then-copy pattern
    `push eax; <chain of EAX-writing ops>; mov ecx, eax; pop eax`
    by retargeting every instruction in the chain to write ECX
    instead. The `mov ecx, eax` becomes redundant and is dropped.
    Saves 2 bytes per binop with a non-trivial right-hand side
    (e.g. `a[i] cmp b[i]` where each side dereferences an array).
  - cmp_zero_to_test: replace `cmp reg, 0` with `test reg, reg`.
    Saves 1 byte per match (3-byte cmp-imm8-sign-ext → 2-byte test).
    Identical flag effects for ZF/SF/OF/CF/PF.
  - dead_mov_to_reg: drop `mov <reg>, <src>` when forward scan finds
    a full reg overwrite before any read. Common in postfix `i++`
    in for-loop steps where the value is discarded. Crosses labels
    and unconditional jumps under the codegen invariant (regs are
    always overwritten at the start of each labeled block).
  - prologue_to_enter: `push ebp; mov ebp, esp; sub esp, IMM`
    becomes `enter IMM, 0`. Saves 2-5 bytes per function (depending
    on whether IMM fits in imm8). Identical semantics — enter
    pushes EBP, sets EBP=ESP, allocates IMM stack bytes.
  - redundant_eax_load: drop `mov eax, M` when EAX already provably
    holds M's value (we just loaded it; intervening instructions
    don't write EAX or alias-modify M). Common in `int v = x; v * v`
    where the second `mov eax, [ebp+8]` is redundant.
  - label_offset_fold: collapse `mov reg, LABEL; add reg, IMM` (or
    `sub reg, IMM`) into `mov reg, LABEL ± IMM`. NASM resolves the
    `LABEL ± IMM` at assemble time, producing a single 5-byte
    `mov reg, imm32` and dropping the 3-byte add/sub. Saves 3 bytes
    per match. Common in pointer arithmetic on globals
    (`mov eax, _g; add eax, 8` for `g[2]`).
  - cmp_load_collapse: collapse `mov reg, [mem]; cmp reg, X` into
    `cmp dword [mem], X` when reg is dead after. Saves 2-3 bytes
    per match. Common in `if (var == X)` and `if (var)` (the
    latter via `cmp_zero_to_test`'s output).
  - rmw_collapse: collapse `mov reg, [mem]; OP reg, IMM; mov [mem],
    reg` into `OP dword [mem], IMM` when reg is dead after. OP is
    one of add/sub/and/or/xor. Saves 5 bytes per match. Common in
    compound assignments to memory operands (`x += 5`).
  - fst_fstp_collapse: collapse `fst <addr>; fstp st0` into a single
    `fstp <addr>`. Saves 2 bytes per match. Common in FPU-heavy
    code — uc386 codegen lowers `*p = expr` for float/double via
    fst+fstp-st0, which equals fstp-to-mem.
  - fpu_op_collapse: collapse `fld <addr>; foppc st1, st0` into a
    single memory-form FPU op `fop <addr>`. Same FPU stack/memory
    state, 2 bytes saved per match. Applies to faddp / fsubp /
    fmulp / fdivp (the fsubrp/fdivrp swapped variants get their own
    rewrites).
  - add_one_to_inc: `add reg, 1` → `inc reg` (and `sub reg, 1` →
    `dec reg`). Saves 2 bytes per register match, 1 byte per memory
    match (`add dword [mem], 1` → `inc dword [mem]`, etc., for byte/
    word/dword forms). inc/dec leave CF unchanged while add/sub set
    CF; the rewrite is safe only when CF is dead after (no
    `jc/jnc/ja/jb/...` reads it before being overwritten).
  - redundant_test_collapse: drop `test reg, reg` immediately after
    a flag-setting arithmetic op on the same reg. Common after
    `and eax, MASK; test eax, eax; jcc` — the AND already set ZF/SF
    based on its result, so the test is redundant. Saves 2 bytes.
  - push_memory: collapse `mov reg, [mem]; push reg` into a single
    `push dword [mem]`. NASM's `push r/m32` form is 3 bytes for ebp-
    relative addressing; the original mov+push is 4 bytes. Saves 1
    byte per match. Common in cdecl call-site arg setup, where the
    intermediate register is dead after the push.
  - disp_load_collapse: collapse `add REG, DISP; mov DST, [REG]`
    into `mov DST, [REG + DISP]` using x86's disp32 addressing.
    Saves 1-2 bytes per match. Common in `p->member` struct member
    access for non-zero offsets, where the codegen emits explicit
    `add reg, offset` before dereference.
  - index_load_collapse: collapse the array-index address-
    computation chain `shl IDX, N; add BASE, IDX; mov DST, [BASE]`
    into a single `mov DST, [BASE + IDX*SCALE]` using x86's SIB
    byte. Saves 4 bytes per match (shl + add gone, mov gains 1
    byte for the SIB byte). N ∈ {1, 2, 3} for scale 2/4/8. Common
    in `arr[i]` array indexing for int arrays (scale 4) and
    pointer arrays (scale 4).
  - compound_assign_collapse: collapse the codegen's compound-
    assignment frame `push dword [m]; <chain>; mov ecx, eax;
    pop eax; OP eax, ecx; mov [m], eax` into a single in-place
    `OP [m], eax`. The chain must not modify [m] or perform stack
    manipulation, and ECX must be dead after the store. Saves 8
    bytes per match (drops the push/pop framing + transfer + store).
    Common in `x += rhs` / `x -= rhs` etc. for slot-typed lvalues
    where the rhs computation can't be reduced via simpler passes.
  - redundant_xor_zero: drop `xor reg, reg` when reg is already
    zero from a recent `xor reg, reg` and nothing modified it
    since. Saves 2 bytes per match. Common after zero_init_collapse
    fires twice in a function prologue, leaving back-to-back
    `xor eax, eax` instructions.
  - zero_init_collapse: replace 1+ adjacent `mov <size> [m], 0`
    stores with `xor eax, eax` + per-store `mov [m], <eax/ax/al>`.
    Saves 2 bytes per dword-zero store (7 bytes → 3 bytes per store
    plus a single 2-byte xor) and 1 byte per byte-zero store. Common
    in function prologues with multiple `int x = 0;` initializers.
  - jcc_jmp_inversion: collapse `jcc L1; jmp L2; L1:` into
    `j!cc L2; L1:` (with the cc inverted). Saves 5 bytes per match —
    drops the unconditional jmp. Common in `if-else` and ternary
    lowering where one branch is just a fallthrough to L2 after
    redundant_eax_load eliminates the only instruction.
  - narrowing_load_test_collapse: collapse
    `movsx/movzx eax, byte [SRC]; test eax, eax` into
    `cmp byte [SRC], 0` (and likewise for word). The narrowing load
    only sets EAX; the test then checks for zero. Direct `cmp <size>
    [SRC], 0` produces the same flags (a zero-extended/sign-extended
    byte/word is zero iff the byte/word itself is zero). Saves 2
    bytes per match. Common in string/byte loops:
    `while (*p) p++;` → `... cmp byte [eax], 0; jz end ...`.

Patterns to add (see PEEPHOLE_PLAN.md for details): tail calls,
jump threading, multi-instruction right-operand retargeting.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# Instruction line: leading whitespace, mnemonic, operands.
# Examples:
#   "        jmp     .epilogue"
#   "        mov     eax, [ebp - 4]"
#   "        ret"
_INSTR_RE = re.compile(r"^(\s+)([a-zA-Z][a-zA-Z0-9]*)(?:\s+(.*?))?\s*$")

# Label line: identifier optionally prefixed by `.`, ending in `:`.
# Examples: `_main:`, `.L1_for_top:`, `__start:`
_LABEL_RE = re.compile(r"^(\.?[A-Za-z_][A-Za-z0-9_]*)\s*:\s*$")

# Top-level directives we treat as fences (end any dead-zone).
# `section .text` / `section .data` / `section .bss` / `bits 32` /
# `global _name` / `extern _name`.
_FENCE_PREFIXES = ("section", "bits", "global", "extern")


@dataclass
class Line:
    """One line of the asm output, classified."""
    raw: str
    kind: str  # "instr" | "label" | "directive" | "data" | "comment" | "blank"
    op: str = ""        # for instr: mnemonic (lowercased)
    operands: str = ""  # for instr: everything after mnemonic, raw
    label: str = ""     # for label: the name


def _classify(raw: str) -> Line:
    stripped = raw.strip()
    if not stripped:
        return Line(raw=raw, kind="blank")
    if stripped.startswith(";"):
        return Line(raw=raw, kind="comment")
    # Label
    m = _LABEL_RE.match(raw.lstrip())
    if m and not raw.lstrip().startswith(("section", "bits", "global", "extern")):
        # Make sure the colon is the WHOLE thing — not e.g. `mov eax, [foo]`
        # which doesn't match _LABEL_RE anyway.
        return Line(raw=raw, kind="label", label=m.group(1))
    # Indented mnemonic line
    m = _INSTR_RE.match(raw)
    if m:
        op = m.group(2).lower()
        operands = (m.group(3) or "").strip()
        # NASM directives like `db`, `dw`, `dd`, `dq`, `times`, `resb`,
        # `resw`, `resd`, `resq`, `equ` — these live in .data/.bss and
        # aren't real instructions. Keep them as "data" so dead-after-jmp
        # leaves them alone.
        if op in {"db", "dw", "dd", "dq", "dt", "times",
                  "resb", "resw", "resd", "resq", "equ"}:
            return Line(raw=raw, kind="data", op=op, operands=operands)
        # Section/bits/global/extern at indented positions
        if op in _FENCE_PREFIXES:
            return Line(raw=raw, kind="directive", op=op, operands=operands)
        return Line(raw=raw, kind="instr", op=op, operands=operands)
    # Unrecognized — keep verbatim. Common case: `_label equ value`,
    # or weird user-supplied lines.
    return Line(raw=raw, kind="directive")


def _is_unconditional_terminator(line: Line) -> bool:
    """An instruction whose execution never falls through to the next line."""
    if line.kind != "instr":
        return False
    # `jmp <anything>` (direct or indirect)
    if line.op == "jmp":
        return True
    # `ret` alone (no condition codes on x86 ret).
    if line.op == "ret":
        return True
    # `iret`, `retf`, `retn` — uncommon in our codegen but be safe.
    if line.op in {"iret", "iretd", "retf", "retn"}:
        return True
    return False


def _operands_split(operands: str) -> tuple[str, str] | None:
    """Split a two-operand `mov` into (dest, src). Returns None for
    anything that isn't a clean `dst, src` pair."""
    if "," not in operands:
        return None
    # NASM allows commas inside `[...]` (`[ebp - 4]` doesn't have one,
    # but `[ebp + ecx*4]` doesn't either; `[base + index, scale]`
    # syntax doesn't exist in NASM). Effective addresses use `+` / `-`
    # / `*`, never bare commas, so a top-level split on `,` is safe.
    dest, _, src = operands.partition(",")
    return dest.strip(), src.strip()


def _references_register(text: str, reg: str) -> bool:
    """Does `text` mention register `reg` (case-insensitive)? Looks for
    word-boundary matches so `ecx` doesn't match `cx` or `recx`."""
    return bool(re.search(rf"\b{reg}\b", text, re.IGNORECASE))


def _is_simple_eax_load(line: Line) -> bool:
    """Is this instruction a single `mov eax, <src>` where src is a
    valid mov source operand (immediate, memory ref, register, label)?
    The pattern needs this to know it can rewrite EAX→ECX safely."""
    if line.kind != "instr" or line.op != "mov":
        return False
    parts = _operands_split(line.operands)
    if parts is None:
        return False
    dest, _ = parts
    return dest.lower() == "eax"


def _is_imm_or_label_to_eax(line: Line) -> bool:
    """Is this `mov eax, <constant-expr>` — immediate or label-address,
    NOT a memory deref or register source? Used by both
    imm_store_collapse and push_immediate."""
    if line.kind != "instr" or line.op != "mov":
        return False
    parts = _operands_split(line.operands)
    if parts is None:
        return False
    dest, src = parts
    if dest.lower() != "eax":
        return False
    # Reject memory derefs.
    if "[" in src:
        return False
    # Reject bare register names.
    regs = {"eax", "ebx", "ecx", "edx", "esi", "edi", "ebp", "esp",
            "ax", "bx", "cx", "dx", "si", "di", "bp", "sp",
            "al", "bl", "cl", "dl", "ah", "bh", "ch", "dh"}
    if src.lower() in regs:
        return False
    return True


def _retarget_eax_to_ecx(line: Line) -> Line:
    """Rewrite `mov eax, src` to `mov ecx, src`. Caller verified the
    line shape via `_is_simple_eax_load`."""
    parts = _operands_split(line.operands)
    assert parts is not None
    _, src = parts
    new_raw = re.sub(r"\beax\b", "ecx", line.raw, count=1)
    new_operands = f"ecx, {src}"
    return Line(raw=new_raw, kind="instr", op="mov", operands=new_operands)


def _jmp_target(line: Line) -> str | None:
    """Return the target label of a direct `jmp <label>`, or None for
    indirect / unrecognized."""
    if line.kind != "instr" or line.op != "jmp":
        return None
    return _branch_target(line)


def _branch_target(line: Line) -> str | None:
    """Return the target label of any direct branch (jmp/jcc) — i.e.
    the operand is a literal label name. Returns None for indirect
    branches (`jmp eax`, `jmp [_tbl]`) or unparseable operands.
    """
    if line.kind != "instr":
        return None
    operand = line.operands.strip()
    if not operand or operand[0] in "[":
        return None
    if operand.lower() in {"eax", "ebx", "ecx", "edx", "esi", "edi",
                           "ebp", "esp"}:
        return None
    if re.fullmatch(r"\.?[A-Za-z_][A-Za-z0-9_]*", operand):
        return operand
    return None


class PeepholeOptimizer:
    """Multi-pass peephole optimizer.

    Usage:
        opt = PeepholeOptimizer()
        new_asm = opt.optimize(asm_text)
        for name, count in opt.stats.items(): ...
    """

    MAX_PASSES = 10

    def __init__(self) -> None:
        self.stats: dict[str, int] = {}

    def optimize(self, asm_text: str) -> str:
        # Preserve trailing newline behavior.
        trailing_nl = asm_text.endswith("\n")
        raw_lines = asm_text.splitlines()
        lines = [_classify(r) for r in raw_lines]

        for _ in range(self.MAX_PASSES):
            before = len(lines)
            lines = self._pass_dead_after_terminator(lines)
            lines = self._pass_jmp_to_next_label(lines)
            lines = self._pass_binop_collapse(lines)
            lines = self._pass_store_collapse(lines)
            lines = self._pass_leave_collapse(lines)
            lines = self._pass_imm_store_collapse(lines)
            lines = self._pass_setcc_jcc_collapse(lines)
            lines = self._pass_push_immediate(lines)
            lines = self._pass_imm_binop_collapse(lines)
            lines = self._pass_mov_zero_to_xor(lines)
            lines = self._pass_store_load_collapse(lines)
            lines = self._pass_right_operand_retarget(lines)
            # push_memory runs AFTER right_operand_retarget — that pass
            # consumes inner `push eax / chain / mov ecx,eax / pop eax`
            # save-restore frames, eliminating the inner mov+push
            # entirely (saves more than push_memory's 1-byte rewrite).
            # Run push_memory afterwards so it picks up only the
            # remaining mov+push pairs (outer save-pushes that don't
            # have a matching mov-ecx-eax pop pattern).
            lines = self._pass_push_memory(lines)
            lines = self._pass_cmp_zero_to_test(lines)
            lines = self._pass_dead_mov_to_reg(lines)
            lines = self._pass_prologue_to_enter(lines)
            lines = self._pass_label_offset_fold(lines)
            # cmp_load_collapse runs BEFORE redundant_eax_load: the
            # smarter redundant_eax_load (with jcc-target tracking)
            # can drop a mov that cmp_load_collapse would've fused
            # with its cmp. Running cmp_load_collapse first lets both
            # passes fire on independent patterns.
            lines = self._pass_cmp_load_collapse(lines)
            lines = self._pass_redundant_eax_load(lines)
            lines = self._pass_redundant_ecx_load(lines)
            lines = self._pass_rmw_collapse(lines)
            lines = self._pass_fst_fstp_collapse(lines)
            lines = self._pass_fpu_op_collapse(lines)
            lines = self._pass_add_one_to_inc(lines)
            lines = self._pass_redundant_test_collapse(lines)
            lines = self._pass_narrowing_load_test_collapse(lines)
            lines = self._pass_jcc_jmp_inversion(lines)
            lines = self._pass_zero_init_collapse(lines)
            lines = self._pass_redundant_xor_zero(lines)
            lines = self._pass_compound_assign_collapse(lines)
            lines = self._pass_index_load_collapse(lines)
            lines = self._pass_disp_load_collapse(lines)
            lines = self._pass_push_disp_collapse(lines)
            lines = self._pass_push_index_collapse(lines)
            lines = self._pass_self_mov_elimination(lines)
            lines = self._pass_transfer_pop_collapse(lines)
            lines = self._pass_dup_push_pop_self_op(lines)
            lines = self._pass_push_pop_op_to_memop(lines)
            lines = self._pass_label_load_collapse(lines)
            lines = self._pass_label_push_collapse(lines)
            lines = self._pass_label_store_collapse(lines)
            lines = self._pass_lea_load_collapse(lines)
            lines = self._pass_lea_offset_fold(lines)
            lines = self._pass_lea_forward_to_reg(lines)
            lines = self._pass_lea_store_collapse(lines)
            lines = self._pass_dead_stack_store(lines)
            lines = self._pass_value_forward_to_reg(lines)
            lines = self._pass_byte_stores_to_dword(lines)
            lines = self._pass_pop_index_push_collapse(lines)
            lines = self._pass_pop_index_load_collapse(lines)
            lines = self._pass_push_pop_to_mov(lines)
            lines = self._pass_sib_const_index_fold(lines)
            lines = self._pass_push_const_index_fold(lines)
            after = len(lines)
            if after == before:
                # All passes only delete or replace-with-fewer; if the
                # length didn't change, no rewrites fired this round.
                break

        out = "\n".join(line.raw for line in lines)
        if trailing_nl:
            out += "\n"
        return out

    # ── Patterns ──────────────────────────────────────────────────

    def _pass_dead_after_terminator(self, lines: list[Line]) -> list[Line]:
        """P1: Drop instruction lines (and comments/blanks adjacent to
        them) between an unconditional terminator and the next label /
        directive / data line.

        Labels, directives, data lines, and the terminator itself are
        preserved — only instructions in the dead zone get dropped, plus
        comments and blanks that have no live neighbor on either side.
        Conservative: we keep comments/blanks if any live instr exists
        before the next fence in the dead zone (none can, by definition,
        but the rule is: "comment/blank lines between dead code stay
        with the code" — so they're dropped too since the surrounding
        code is dead).
        """
        out: list[Line] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            out.append(line)
            if not _is_unconditional_terminator(line):
                i += 1
                continue
            # Found a terminator. Skip ahead past dead instructions.
            j = i + 1
            killed = 0
            while j < len(lines):
                nxt = lines[j]
                if nxt.kind in ("label", "directive", "data"):
                    break
                if nxt.kind == "instr":
                    killed += 1
                    j += 1
                    continue
                # Blank / comment in the dead zone — drop too, since
                # they were attached to the dead instructions.
                j += 1
            if killed:
                self.stats["dead_after_terminator"] = (
                    self.stats.get("dead_after_terminator", 0) + killed
                )
            i = j
        return out

    def _pass_binop_collapse(self, lines: list[Line]) -> list[Line]:
        """Collapse the stack-machine right-operand transfer pattern.

        Match (4 consecutive instr lines, blanks/comments tolerated):
            push    eax
            mov     eax, <src>           ; single-instruction right
            mov     ecx, eax
            pop     eax
        Replace with:
            mov     ecx, <src>

        The push saves the left operand; the right is loaded into EAX,
        transferred to ECX as the binop's second operand, and EAX is
        restored. After collapse, EAX is unchanged (no clobber → no
        save needed), and the right lands directly in ECX.

        Safe because `mov ecx, src` reads `src` with the same EAX/ECX
        values as the original middle instruction did (push doesn't
        change EAX, and the original middle hadn't touched ECX yet).
        """
        out: list[Line] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            # Look for `push eax`.
            if (line.kind == "instr" and line.op == "push"
                    and line.operands.strip().lower() == "eax"):
                # Need three more instr lines below: mov eax, src;
                # mov ecx, eax; pop eax.
                instrs = self._next_n_instrs(lines, i + 1, 3)
                if instrs is not None and self._matches_binop_collapse(instrs):
                    [(_, c), (_, d), (_, e)] = instrs
                    # Keep original blanks/comments between push and the
                    # first instr (none in our codegen, but be safe).
                    # Build the replacement: a single retargeted line.
                    new_line = _retarget_eax_to_ecx(c)
                    # Skip from `push eax` (i) through `pop eax` (e_idx).
                    e_idx = instrs[2][0]
                    out.append(new_line)
                    self.stats["binop_collapse"] = (
                        self.stats.get("binop_collapse", 0) + 1
                    )
                    i = e_idx + 1
                    continue
            out.append(line)
            i += 1
        return out

    def _pass_store_collapse(self, lines: list[Line]) -> list[Line]:
        """Drop the push/pop pair around a store-through-pointer.

        Match (4 consecutive instr lines):
            push    eax
            mov     eax, <src>           ; single-instruction value
            pop     ecx
            mov     [ecx], eax
        Replace with:
            mov     ecx, eax
            mov     eax, <src>
            mov     [ecx], eax

        Net: drop push + pop, add `mov ecx, eax`. Saves 1 instruction
        plus 2 stack ops. Conservative compared to the full retarget
        variant; that one rewrites the previous `mov eax, addr` line
        too. Defer.

        Safe because `<src>` is verified not to read ECX (would
        observe the new save value rather than the old caller-state).
        """
        out: list[Line] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if (line.kind == "instr" and line.op == "push"
                    and line.operands.strip().lower() == "eax"):
                instrs = self._next_n_instrs(lines, i + 1, 3)
                if instrs is not None and self._matches_store_collapse(instrs):
                    [(c_idx, c), (d_idx, _d), (e_idx, e)] = instrs
                    # Replace `push eax` with `mov ecx, eax`.
                    push_line = line
                    new_push = Line(
                        raw=push_line.raw.replace(
                            "push    eax", "mov     ecx, eax", 1
                        ).replace("push eax", "mov ecx, eax", 1),
                        kind="instr", op="mov", operands="ecx, eax",
                    )
                    out.append(new_push)
                    out.append(c)
                    out.append(e)
                    self.stats["store_collapse"] = (
                        self.stats.get("store_collapse", 0) + 1
                    )
                    i = e_idx + 1
                    continue
            out.append(line)
            i += 1
        return out

    # ── Helpers for multi-line patterns ───────────────────────────

    def _next_n_instrs(self, lines: list[Line], start: int,
                       n: int) -> list[tuple[int, Line]] | None:
        """Return the next N instruction lines starting at index
        `start`, as `[(idx, line), ...]`. Skip blanks/comments. Return
        None if we hit a label/directive/data before finding N instrs."""
        out: list[tuple[int, Line]] = []
        j = start
        while j < len(lines) and len(out) < n:
            ln = lines[j]
            if ln.kind == "instr":
                out.append((j, ln))
            elif ln.kind in ("label", "directive", "data"):
                return None
            j += 1
        if len(out) < n:
            return None
        return out

    @staticmethod
    def _matches_binop_collapse(
        instrs: list[tuple[int, Line]],
    ) -> bool:
        """Verify the 3-instruction tail of the binop pattern:
        `mov eax, <src>; mov ecx, eax; pop eax`."""
        c_line = instrs[0][1]
        d_line = instrs[1][1]
        e_line = instrs[2][1]
        if not _is_simple_eax_load(c_line):
            return False
        # `push eax` decreases ESP by 4. If src is ESP-relative, the
        # original `mov eax, [esp + N]` reads from a different location
        # than the post-collapse `mov ecx, [esp + N]` would. Skip.
        parts = _operands_split(c_line.operands)
        assert parts is not None
        _, src = parts
        if _references_register(src, "esp"):
            return False
        if not (d_line.op == "mov"
                and d_line.operands.replace(" ", "").lower() == "ecx,eax"):
            return False
        if not (e_line.op == "pop"
                and e_line.operands.strip().lower() == "eax"):
            return False
        return True

    @staticmethod
    def _matches_store_collapse(
        instrs: list[tuple[int, Line]],
    ) -> bool:
        """Verify the 3-instruction tail of the store pattern:
        `mov eax, <src>; pop ecx; mov [ecx], eax` where src doesn't
        read ECX or ESP."""
        c_line = instrs[0][1]
        d_line = instrs[1][1]
        e_line = instrs[2][1]
        if not _is_simple_eax_load(c_line):
            return False
        parts = _operands_split(c_line.operands)
        assert parts is not None
        _, src = parts
        # src must not read ECX (would alias the saved address) or ESP
        # (the push shifted ESP, so an `[esp + N]` operand reads a
        # different byte than it would post-collapse).
        if _references_register(src, "ecx"):
            return False
        if _references_register(src, "esp"):
            return False
        if not (d_line.op == "pop"
                and d_line.operands.strip().lower() == "ecx"):
            return False
        if not (e_line.op == "mov"
                and e_line.operands.replace(" ", "").lower() == "[ecx],eax"):
            return False
        return True

    # Inverse table for `j<cc>` mnemonics. Mirrors `set<cc>`'s opposite
    # condition. Used by setcc_jcc_collapse.
    _CC_INVERSE: dict[str, str] = {
        "e": "ne", "ne": "e",
        "z": "nz", "nz": "z",   # z is alias of e
        "l": "ge", "ge": "l",
        "le": "g", "g": "le",
        "b": "ae", "ae": "b",
        "be": "a", "a": "be",
        "c": "nc", "nc": "c",   # c is alias of b
        "s": "ns", "ns": "s",
        "o": "no", "no": "o",
        "p": "np", "np": "p",
        "pe": "po", "po": "pe",
    }

    def _pass_setcc_jcc_collapse(self, lines: list[Line]) -> list[Line]:
        """Replace the setCC + movzx + test + jz/jnz boolean-materialize
        chain with a single conditional jump.

        Match (4 consecutive instr lines):
            setCC   al                ; or any sub-byte register
            movzx   eax, al           ; widen
            test    eax, eax
            j[n]z   LBL

        Replace with:
            j<CC-or-inverse>  LBL

        Mapping: `jz LBL` after `setCC al` jumps when AL was 0 — i.e.,
        when the inverted CC was true. So emit `j<inverse-of-CC> LBL`.
        `jnz LBL` jumps when AL was 1 — emit `j<CC> LBL` directly.

        The setCC's destination must be the low byte of EAX (AL),
        the movzx must widen AL into EAX, and the test must be on
        EAX. We don't try to handle other registers (BL, CL, etc.) —
        the codegen always uses AL → EAX in this pattern.
        """
        out: list[Line] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if (line.kind == "instr"
                    and line.op.startswith("set")
                    and line.op[3:] in self._CC_INVERSE):
                # setCC. Check operand is `al`.
                if line.operands.strip().lower() != "al":
                    out.append(line)
                    i += 1
                    continue
                instrs = self._next_n_instrs(lines, i + 1, 3)
                if instrs is None:
                    out.append(line)
                    i += 1
                    continue
                b_line = instrs[0][1]
                c_line = instrs[1][1]
                d_line = instrs[2][1]
                # b: `movzx eax, al`
                if not (b_line.op == "movzx"
                        and b_line.operands.replace(" ", "").lower()
                            == "eax,al"):
                    out.append(line)
                    i += 1
                    continue
                # c: `test eax, eax`
                if not (c_line.op == "test"
                        and c_line.operands.replace(" ", "").lower()
                            == "eax,eax"):
                    out.append(line)
                    i += 1
                    continue
                # d: `jz LBL` or `jnz LBL`. Resolve target label.
                if d_line.op not in ("jz", "je", "jnz", "jne"):
                    out.append(line)
                    i += 1
                    continue
                target = d_line.operands.strip()
                if not target:
                    out.append(line)
                    i += 1
                    continue
                # Map: setCC + jz → j<inverted CC>; setCC + jnz → j<CC>
                cc = line.op[3:]
                jz_like = d_line.op in ("jz", "je")
                final_cc = self._CC_INVERSE[cc] if jz_like else cc
                leading_ws = re.match(r"^\s*", line.raw).group(0)
                new_raw = f"{leading_ws}j{final_cc:<7}{target}"
                new_line = Line(raw=new_raw, kind="instr",
                                op=f"j{final_cc}", operands=target)
                out.append(new_line)
                self.stats["setcc_jcc_collapse"] = (
                    self.stats.get("setcc_jcc_collapse", 0) + 1
                )
                # Skip past lines i (setCC), b, c, d.
                d_idx = instrs[2][0]
                i = d_idx + 1
                continue
            out.append(line)
            i += 1
        return out

    def _pass_push_immediate(self, lines: list[Line]) -> list[Line]:
        """Replace `mov <reg>, IMM_OR_LABEL; ... ; push <reg>;
        <reg-write>` with `... ; push IMM_OR_LABEL; <reg-write>`.

        The codegen emits `mov reg, X; push reg` for every cdecl arg
        push. NASM's `push imm8` / `push imm32` instructions skip the
        register round-trip entirely. The instruction after `push reg`
        must overwrite the register for the rewrite to be safe.

        Up to 4 "reg-neutral" instructions may sit between the mov
        and the push — they must not read OR write the register's
        family (EAX/AX/AL/AH for EAX, etc.). This catches the LL-push
        pattern `mov eax, lo; mov edx, hi; push edx; push eax` which
        becomes `mov edx, hi; push edx; push lo` (and recurses on
        the EDX form to give `push hi; push lo`).

        Works for EAX, EBX, ECX, EDX, ESI, EDI, EBP. ESP excluded
        (used as stack pointer; rewrites would break alignment).
        """
        SUPPORTED_REGS = {"eax", "ebx", "ecx", "edx", "esi", "edi", "ebp"}

        def overwrites_reg(line: Line, reg32: str) -> bool:
            return PeepholeOptimizer._is_pure_reg_write(line, reg32)

        def is_reg_neutral(line: Line, reg32: str) -> bool:
            """Doesn't read or write the register family. May touch
            other state."""
            if line.kind != "instr":
                return False
            if line.op == "call":
                # Direct calls clobber EAX/ECX/EDX. For other regs
                # they're preserved. But indirect calls or any call
                # might invalidate analysis; conservative: block.
                return False
            if line.op in {"ret", "iret", "iretd", "retf", "retn",
                            "jmp"} or line.op.startswith("j"):
                return False
            if not PeepholeOptimizer._has_explicit_operands_only(line):
                return False
            if PeepholeOptimizer._references_reg_family(
                    line.operands, reg32):
                return False
            return True

        def is_imm_or_label_to_reg(line: Line, reg32: str) -> bool:
            if line.kind != "instr" or line.op != "mov":
                return False
            parts = _operands_split(line.operands)
            if parts is None:
                return False
            dest, src = parts
            if dest.lower() != reg32:
                return False
            if "[" in src:
                return False
            regs = {"eax", "ebx", "ecx", "edx", "esi", "edi",
                    "ebp", "esp",
                    "ax", "bx", "cx", "dx", "si", "di", "bp", "sp",
                    "al", "bl", "cl", "dl", "ah", "bh", "ch", "dh"}
            if src.lower() in regs:
                return False
            return True

        out = list(lines)
        i = 0
        while i < len(out):
            line = out[i]
            if (line.kind == "instr" and line.op == "mov"):
                parts = _operands_split(line.operands)
                if parts is not None:
                    dest, src = parts
                    reg32 = dest.lower()
                    if (reg32 in SUPPORTED_REGS
                            and is_imm_or_label_to_reg(line, reg32)):
                        # Find the matching `push <reg>` within the
                        # next 5 instructions, where each intermediate
                        # is reg-neutral.
                        push_idx: int | None = None
                        witness_idx: int | None = None
                        neutrals = 0
                        for j_offset, ln in enumerate(
                                out[i + 1:], start=i + 1):
                            if ln.kind in ("blank", "comment"):
                                continue
                            if ln.kind != "instr":
                                break
                            if (ln.op == "push"
                                    and ln.operands.strip().lower()
                                    == reg32):
                                push_idx = j_offset
                                # Find the witness — scan past
                                # reg-neutral instrs (up to 10).
                                w_neutrals = 0
                                for k_offset, kln in enumerate(
                                        out[j_offset + 1:],
                                        start=j_offset + 1):
                                    if kln.kind in ("blank", "comment"):
                                        continue
                                    if kln.kind != "instr":
                                        break
                                    if overwrites_reg(kln, reg32):
                                        witness_idx = k_offset
                                        break
                                    if not is_reg_neutral(kln, reg32):
                                        break
                                    w_neutrals += 1
                                    if w_neutrals > 10:
                                        break
                                break
                            if not is_reg_neutral(ln, reg32):
                                break
                            neutrals += 1
                            if neutrals > 10:
                                break
                        if (push_idx is not None
                                and witness_idx is not None):
                            push_line = out[push_idx]
                            leading_ws = re.match(
                                r"^\s*", push_line.raw,
                            ).group(0)
                            new_raw = f"{leading_ws}push    {src}"
                            out[push_idx] = Line(
                                raw=new_raw, kind="instr",
                                op="push", operands=src,
                            )
                            out.pop(i)
                            self.stats["push_immediate"] = (
                                self.stats.get("push_immediate", 0) + 1
                            )
                            continue
            i += 1
        return out

    def _pass_push_memory(self, lines: list[Line]) -> list[Line]:
        """Collapse ``mov reg, [mem]; push reg`` into a single
        ``push dword [mem]``.

        NASM's ``push r/m32`` form is 3 bytes for ebp-relative
        addressing (``FF /6 modrm``); the original mov+push is 4
        bytes (``mov reg, [mem]`` is 3 bytes + ``push reg`` is 1
        byte). Saves 1 byte per match.

        Common in cdecl call-site arg setup, where the intermediate
        register is dead after the push (the caller's call instruction
        doesn't read its arg-setup registers).

        Conditions:
        - Line A: ``mov reg, [mem]`` (memory source, register dest).
        - Line B: ``push reg`` (same reg).
        - reg is dead after the push (CFG-aware via `_reg_dead_after`).

        Restricted to EAX/EBX/ECX/EDX/ESI/EDI/EBP — avoids ESP
        (which would corrupt the stack) and segment registers.

        ESP-relative memory operands (``[esp + N]``) are skipped:
        ``push`` decrements ESP, so the memory operand's address
        would shift between mov and push (well, it's stable in
        mov-then-push-with-direct-mem because the memory access
        happens before ESP changes — but we play conservative).
        """
        SUPPORTED_REGS = {
            "eax", "ebx", "ecx", "edx", "esi", "edi", "ebp",
        }
        out: list[Line] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if (
                i + 1 < len(lines)
                and line.kind == "instr"
                and line.op == "mov"
            ):
                parts = _operands_split(line.operands)
                if parts is not None:
                    dest, src = parts
                    dest_low = dest.strip().lower()
                    src_norm = src.strip()
                    if (
                        dest_low in SUPPORTED_REGS
                        and src_norm.startswith("[")
                        and src_norm.endswith("]")
                        and "esp" not in src_norm.lower()
                    ):
                        nxt = lines[i + 1]
                        if (
                            nxt.kind == "instr"
                            and nxt.op == "push"
                            and nxt.operands.strip().lower()
                            == dest_low
                            and self._reg_dead_after(
                                lines, i + 2, dest_low
                            )
                        ):
                            indent = self._extract_indent(nxt.raw)
                            new_operands = f"dword {src_norm}"
                            new_raw = (
                                f"{indent}push    {new_operands}"
                            )
                            new_line = Line(
                                raw=new_raw,
                                kind="instr",
                                op="push",
                                operands=new_operands,
                            )
                            out.append(new_line)
                            self.stats["push_memory"] = (
                                self.stats.get("push_memory", 0) + 1
                            )
                            i += 2
                            continue
            out.append(line)
            i += 1
        return out

    def _pass_imm_store_collapse(self, lines: list[Line]) -> list[Line]:
        """Collapse `mov eax, IMM; mov [addr], eax` to `mov dword
        [addr], IMM`, when EAX is provably dead after the store.

        The fast path uses an immediate witness (the very next instr
        is a fresh EAX overwrite); a CFG-aware fallback uses
        ``_reg_dead_after`` for cases where the witness is past a
        label boundary (basic block boundary).

        NASM accepts `mov dword [addr], IMM` directly.

        Conditions:
        - Line A: `mov eax, src` where src is a constant expression
          (no `[` — i.e., not a memory load, not a register-to-EAX
          where the register might be live).
        - Line B: `mov [addr], eax` — a 4-byte store from EAX.
        - EAX must be dead after Line B (immediate witness OR CFG-
          aware liveness).
        """
        out_list: list[Line] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if (line.kind == "instr" and line.op == "mov"
                    and self._is_imm_to_eax(line)):
                # Try the fast path first: immediate witness in the
                # next 2 instructions (B then C-as-witness).
                fast_instrs = self._next_n_instrs(lines, i + 1, 2)
                fired_fast = False
                b_idx_fast = None
                if fast_instrs is not None and self._matches_imm_store(
                    fast_instrs, line
                ):
                    [(b_idx_fast, _b), (_c_idx, _c)] = fast_instrs
                    fired_fast = True
                if fired_fast:
                    [(b_idx, b_line), (_c_idx, _c)] = fast_instrs
                    parts_a = _operands_split(line.operands)
                    assert parts_a is not None
                    _, src_imm = parts_a
                    parts_b = _operands_split(b_line.operands)
                    assert parts_b is not None
                    addr, _ = parts_b
                    leading_ws = re.match(r"^\s*", line.raw).group(0)
                    new_raw = f"{leading_ws}mov     dword {addr}, {src_imm}"
                    new_line = Line(
                        raw=new_raw, kind="instr", op="mov",
                        operands=f"dword {addr}, {src_imm}",
                    )
                    out_list.append(new_line)
                    self.stats["imm_store_collapse"] = (
                        self.stats.get("imm_store_collapse", 0) + 1
                    )
                    i = b_idx + 1
                    continue
                # Slow path: CFG-aware liveness — check Line B (next
                # instr is `mov [m], eax`) and EAX dead after.
                next_instrs = self._next_n_instrs(lines, i + 1, 1)
                if next_instrs is not None:
                    [(b_idx, b_line)] = next_instrs
                    if (
                        b_line.op == "mov"
                        and self._is_eax_to_mem_store(b_line)
                        and self._reg_dead_after(
                            lines, b_idx + 1, "eax"
                        )
                    ):
                        parts_a = _operands_split(line.operands)
                        assert parts_a is not None
                        _, src_imm = parts_a
                        addr = self._mem_dest(b_line)
                        if addr is not None:
                            leading_ws = re.match(
                                r"^\s*", line.raw
                            ).group(0)
                            new_raw = (
                                f"{leading_ws}mov     dword "
                                f"{addr}, {src_imm}"
                            )
                            new_line = Line(
                                raw=new_raw, kind="instr", op="mov",
                                operands=(
                                    f"dword {addr}, {src_imm}"
                                ),
                            )
                            out_list.append(new_line)
                            self.stats[
                                "imm_store_collapse"
                            ] = (
                                self.stats.get(
                                    "imm_store_collapse", 0
                                ) + 1
                            )
                            i = b_idx + 1
                            continue
            out_list.append(line)
            i += 1
        return out_list

    @staticmethod
    def _is_imm_to_eax(line: Line) -> bool:
        """Is this a `mov eax, <constant-expr>` (immediate or label,
        but not a memory deref or register source)?"""
        return _is_imm_or_label_to_eax(line)

    @staticmethod
    def _matches_imm_store(
        instrs: list[tuple[int, Line]],
        a_line: Line,
    ) -> bool:
        """Verify line B (`mov [addr], eax`) and line C (`mov eax, X`).

        Line C must be a *fresh* EAX overwrite — its source operand
        must not reference EAX/AX/AL/AH. Otherwise the instruction is
        a self-RMW (e.g. ``mov eax, [eax]``) and depends on EAX's
        prior value. Dropping line A (``mov eax, IMM``) in that case
        would change line C's effective behavior because EAX would no
        longer hold IMM.
        """
        b_line = instrs[0][1]
        c_line = instrs[1][1]
        if b_line.op != "mov":
            return False
        parts_b = _operands_split(b_line.operands)
        if parts_b is None:
            return False
        b_dest, b_src = parts_b
        # Destination must be a memory ref.
        if not b_dest.startswith("["):
            return False
        # Source must be EAX (not a sub-register — those need narrower
        # store widths and a different rewrite).
        if b_src.strip().lower() != "eax":
            return False
        # Line C must be a fresh EAX overwrite (any of mov/lea/movsx/
        # movzx/pop/xor-self) whose source is EAX-independent.
        if not PeepholeOptimizer._is_fresh_eax_write(c_line):
            return False
        return True

    def _pass_leave_collapse(self, lines: list[Line]) -> list[Line]:
        """Replace the function epilogue's `mov esp, ebp; pop ebp` with
        the single-byte `leave` instruction. NASM emits both encodings
        equivalently; `leave` is 1 byte (0xC9) vs the 3-byte combination
        (0x89 0xEC 0x5D)."""
        out: list[Line] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if (line.kind == "instr"
                    and line.op == "mov"
                    and line.operands.replace(" ", "").lower() == "esp,ebp"):
                # Look at the next non-blank/non-comment instr.
                j = i + 1
                while j < len(lines) and lines[j].kind in ("blank", "comment"):
                    j += 1
                if (j < len(lines)
                        and lines[j].kind == "instr"
                        and lines[j].op == "pop"
                        and lines[j].operands.strip().lower() == "ebp"):
                    # Build a `leave` line, preserving the original
                    # leading whitespace from the `mov esp, ebp` line.
                    leading_ws = re.match(r"^\s*", line.raw).group(0)
                    leave_raw = f"{leading_ws}leave"
                    out.append(Line(raw=leave_raw, kind="instr",
                                    op="leave", operands=""))
                    self.stats["leave_collapse"] = (
                        self.stats.get("leave_collapse", 0) + 1
                    )
                    i = j + 1
                    continue
            out.append(line)
            i += 1
        return out

    # Binops that accept `OP eax, imm32` directly (no ecx round-trip).
    # `imul eax, ecx` becomes `imul eax, eax, IMM` — NASM accepts both
    # `imul eax, IMM` (which encodes as `imul eax, eax, IMM`) and the
    # explicit three-operand form. We use the two-operand form below
    # so the rewrite is uniform across all ops in this set.
    _IMM_BINOP_OPS: frozenset[str] = frozenset({
        "add", "sub", "and", "or", "xor", "cmp", "test",
        "imul", "adc", "sbb",
    })

    def _pass_imm_binop_collapse(self, lines: list[Line]) -> list[Line]:
        """Collapse `mov ecx, <src>; OP eax, ecx` to `OP eax, <src>`.

        Match (2 consecutive instr lines):
            mov     ecx, <src>           ; immediate, label, mem, or reg
            <OP>    eax, ecx

        Replace with:
            <OP>    eax, <src>

        Witness: the instruction after the OP must overwrite ECX
        (full mov / pop / xor reg, reg / lea / call) OR not reference
        ECX (or CX/CL/CH aliases). The codegen always reloads ECX
        before each use, so this fires for every `eax OP <something>`
        shape with a transferred ECX.

        Sources we support:
        - Immediate / label: `mov ecx, 50` / `mov ecx, _glob`.
        - Memory: `mov ecx, [ebp - 4]` / `mov ecx, [_glob]`.
        - Register: `mov ecx, ebx`.

        We don't support sources that contain ECX/CX/CL/CH (would
        self-reference after collapse) or ESP-relative memory if the
        OP itself reads ESP (none of the binops here do, but be
        defensive — only forbid ESP-relative if the OP also reads
        flags, which isn't a real case here).
        """
        out: list[Line] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if (line.kind == "instr"
                    and line.op == "mov"
                    and self._is_simple_to_ecx(line)):
                # Find the next instruction line (the binop).
                instrs = self._next_n_instrs(lines, i + 1, 1)
                if instrs is not None:
                    [(b_idx, b_line)] = instrs
                    if (b_line.op in self._IMM_BINOP_OPS
                            and self._is_eax_ecx_binop(b_line)
                            and self._reg_dead_after(
                                lines, b_idx + 1, "ecx"
                            )):
                        # Build the rewritten op line.
                        parts_a = _operands_split(line.operands)
                        assert parts_a is not None
                        _, src_imm = parts_a
                        leading_ws = re.match(r"^\s*", b_line.raw).group(0)
                        new_raw = (
                            f"{leading_ws}{b_line.op:<8}eax, {src_imm}"
                        )
                        new_line = Line(
                            raw=new_raw, kind="instr", op=b_line.op,
                            operands=f"eax, {src_imm}",
                        )
                        out.append(new_line)
                        self.stats["imm_binop_collapse"] = (
                            self.stats.get("imm_binop_collapse", 0) + 1
                        )
                        i = b_idx + 1
                        continue
            out.append(line)
            i += 1
        return out

    @staticmethod
    def _is_simple_to_ecx(line: Line) -> bool:
        """`mov ecx, <src>` where the source is an immediate, label,
        memory operand, or register that doesn't self-reference ECX.

        Excludes:
        - Sources containing ECX / CX / CL / CH (post-rewrite would
          read the OP's destination operand, changing semantics).
        """
        if line.kind != "instr" or line.op != "mov":
            return False
        parts = _operands_split(line.operands)
        if parts is None:
            return False
        dest, src = parts
        if dest.lower() != "ecx":
            return False
        # No ECX self-reference in the source.
        for alias in ("ecx", "cx", "cl", "ch"):
            if _references_register(src, alias):
                return False
        return True

    @staticmethod
    def _is_eax_ecx_binop(line: Line) -> bool:
        """`<OP> eax, ecx` shape — the only one we can fold a preceding
        `mov ecx, IMM` into."""
        if line.kind != "instr":
            return False
        parts = _operands_split(line.operands)
        if parts is None:
            return False
        dest, src = parts
        return dest.lower() == "eax" and src.lower() == "ecx"

    @staticmethod
    def _ecx_dead_after(line: Line) -> bool:
        """Witness that ECX is dead at this point. Conservative: either
        the instruction overwrites ECX entirely, or it doesn't reference
        ECX at all. Anything that reads ECX (e.g. `cmp ebx, ecx`,
        `mov [ecx], eax`, `add ebx, ecx`) defeats the rewrite."""
        if line.kind != "instr":
            # A label / directive / data line means we can't see what
            # comes after — be conservative and assume ECX may be live
            # (some other path could reach the label).
            return False
        # Full ECX overwrite: mov/pop/lea ecx, ...; xor ecx, ecx.
        if line.op == "mov":
            parts = _operands_split(line.operands)
            if parts and parts[0].lower() == "ecx":
                return True
        if line.op == "pop" and line.operands.strip().lower() == "ecx":
            return True
        if line.op == "lea":
            parts = _operands_split(line.operands)
            if parts and parts[0].lower() == "ecx":
                return True
        if line.op == "xor":
            parts = _operands_split(line.operands)
            if (parts and parts[0].lower() == "ecx"
                    and parts[1].strip().lower() == "ecx"):
                return True
        # `call X` clobbers caller-saved (EAX/ECX/EDX) per cdecl.
        if line.op == "call":
            return True
        # `ret` ends the function — ECX no longer matters.
        if line.op == "ret":
            return True
        # Otherwise: instruction must not reference ECX or any of its
        # sub-registers (CX, CL, CH). The original sequence wrote the
        # IMM into all bits of ECX before the binop, so a downstream
        # read of CL etc. saw IMM-low-bits — but post-collapse, ECX
        # is whatever the prior code left it. Different value → unsafe.
        for alias in ("ecx", "cx", "cl", "ch"):
            if _references_register(line.operands, alias):
                return False
        return True

    # Instructions that clobber EFLAGS (any of CF/PF/AF/ZF/SF/OF).
    # Source: Intel SDM Volume 2 — instructions whose "Flags Affected"
    # section lists at least one flag as written. We use this to know
    # when a prior `xor eax, eax`'s flag write becomes irrelevant.
    _FLAG_CLOBBERING_OPS: frozenset[str] = frozenset({
        "add", "sub", "cmp", "inc", "dec", "and", "or", "xor", "neg",
        "test", "shl", "shr", "sar", "rol", "ror", "rcl", "rcr",
        "shld", "shrd", "imul", "mul", "idiv", "div", "adc", "sbb",
        "bsr", "bsf", "bt", "btc", "btr", "bts", "popf", "popfd",
        "sahf", "stc", "clc", "cmc", "std", "cld", "sti", "cli",
    })

    # Instructions that READ EFLAGS — would observe our `xor`'s
    # spurious flag write. Source: Intel SDM Vol 2 "Flags Tested".
    _FLAG_READING_OPS: frozenset[str] = frozenset({
        # Conditional branches: jcc family
        "ja", "jae", "jb", "jbe", "jc", "jcxz", "je", "jecxz", "jg",
        "jge", "jl", "jle", "jna", "jnae", "jnb", "jnbe", "jnc",
        "jne", "jng", "jnge", "jnl", "jnle", "jno", "jnp", "jns",
        "jnz", "jo", "jp", "jpe", "jpo", "js", "jz",
        # Conditional sets
        "seta", "setae", "setb", "setbe", "setc", "sete", "setg",
        "setge", "setl", "setle", "setna", "setnae", "setnb",
        "setnbe", "setnc", "setne", "setng", "setnge", "setnl",
        "setnle", "setno", "setnp", "setns", "setnz", "seto", "setp",
        "setpe", "setpo", "sets", "setz",
        # Conditional moves
        "cmova", "cmovae", "cmovb", "cmovbe", "cmovc", "cmove",
        "cmovg", "cmovge", "cmovl", "cmovle", "cmovna", "cmovnae",
        "cmovnb", "cmovnbe", "cmovnc", "cmovne", "cmovng", "cmovnge",
        "cmovnl", "cmovnle", "cmovno", "cmovnp", "cmovns", "cmovnz",
        "cmovo", "cmovp", "cmovpe", "cmovpo", "cmovs", "cmovz",
        # Loop-with-condition
        "loope", "loopne", "loopz", "loopnz",
        # Read CF
        "adc", "sbb", "rcl", "rcr",
        # Save flags
        "pushf", "pushfd", "lahf", "into",
    })

    def _pass_mov_zero_to_xor(self, lines: list[Line]) -> list[Line]:
        """Replace `mov reg, 0` with `xor reg, reg` for any 32-bit
        general-purpose register. Saves 3 bytes per match (5-byte
        mov-imm32 → 2-byte xor reg, reg).

        Conservative flag analysis: scan forward up to ~10 instructions
        from the mov. If a flag-CLOBBERING op appears before any
        flag-READING op, the rewrite is safe. If a flag-reading op
        appears first, OR we hit a label / unconditional jump before
        either, skip — the new flags from xor might be observed.

        Special case: `ret` is treated as safe (function return doesn't
        propagate flags semantically — flags are caller-saved by
        convention; the System V i386 ABI doesn't preserve them
        across calls anyway).
        """
        out: list[Line] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            reg = self._mov_reg_zero(line)
            if reg is not None and self._flags_safe_after(lines, i + 1):
                leading_ws = re.match(r"^\s*", line.raw).group(0)
                new_raw = f"{leading_ws}xor     {reg}, {reg}"
                new_line = Line(
                    raw=new_raw, kind="instr", op="xor",
                    operands=f"{reg}, {reg}",
                )
                out.append(new_line)
                self.stats["mov_zero_to_xor"] = (
                    self.stats.get("mov_zero_to_xor", 0) + 1
                )
                i += 1
                continue
            out.append(line)
            i += 1
        return out

    @staticmethod
    def _mov_reg_zero(line: Line) -> str | None:
        """Return the register name if this is `mov <gp32-reg>, 0`,
        else None."""
        if line.kind != "instr" or line.op != "mov":
            return None
        parts = _operands_split(line.operands)
        if parts is None:
            return None
        dest, src = parts
        if src.strip() != "0":
            return None
        dest_lower = dest.lower()
        if dest_lower in {"eax", "ebx", "ecx", "edx",
                          "esi", "edi", "ebp"}:
            return dest_lower
        return None

    def _flags_safe_after(self, lines: list[Line], start_idx: int) -> bool:
        """Scan forward up to ~20 instructions. Return True if a
        flag-CLOBBERING op or function exit appears before any
        flag-READING op.

        Cross-label scanning relies on a uc386 codegen invariant:
        every entry to a labeled block re-establishes flag state via
        an explicit `cmp` / `test` / arithmetic op before any
        conditional branch. The optimizer's input is always uc386
        output (libc is appended later), so this invariant holds.
        """
        scanned = 0
        j = start_idx
        while j < len(lines) and scanned < 20:
            ln = lines[j]
            if ln.kind in ("blank", "comment", "label"):
                # Label crossings are safe under the codegen invariant.
                j += 1
                continue
            if ln.kind in ("directive", "data"):
                # End of function (or section change). Treat as safe —
                # we're past any flag-reading code in this function.
                return True
            if ln.kind != "instr":
                return False
            scanned += 1
            if ln.op == "ret" or ln.op in {"iret", "iretd", "retf", "retn"}:
                return True
            if ln.op == "jmp":
                # Codegen jmps lead to labels whose first instruction
                # re-establishes flag state. Keep scanning past.
                j += 1
                continue
            if ln.op == "call":
                # Calls clobber flags via the callee. Safe.
                return True
            if ln.op in self._FLAG_READING_OPS:
                return False
            if ln.op in self._FLAG_CLOBBERING_OPS:
                return True
            # Otherwise: flag-neutral instr (mov / lea / push / pop /
            # nop / xchg / etc.) — keep scanning.
            j += 1
        return False

    def _pass_store_load_collapse(self, lines: list[Line]) -> list[Line]:
        """Drop the redundant load after a store-then-load of the
        same register to the same address.

        Match (2 consecutive instr lines):
            mov     [<addr>], <reg>
            mov     <reg>, [<addr>]

        Replace with just the store. The register still holds its
        stored value, so the load is a no-op.

        Address comparison is textual (after stripping whitespace),
        which is correct: NASM's `[ebp - 4]` always renders the same
        way for the same operand. The codegen always emits stores
        and loads with matching syntax.

        Skip when the address is ESP-relative — pushes/pops that
        might come between... wait, the pattern requires the two
        instructions to be CONSECUTIVE (next_n_instrs already skips
        blanks/comments only, not instrs), so there's nothing
        between them. ESP-relative is safe.
        """
        out: list[Line] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            store_info = self._mov_reg_to_mem(line)
            if store_info is not None:
                addr, reg = store_info
                instrs = self._next_n_instrs(lines, i + 1, 1)
                if instrs is not None:
                    [(b_idx, b_line)] = instrs
                    load_info = self._mov_mem_to_reg(b_line)
                    if (load_info is not None
                            and load_info == (reg, addr)):
                        out.append(line)
                        self.stats["store_load_collapse"] = (
                            self.stats.get("store_load_collapse", 0) + 1
                        )
                        i = b_idx + 1
                        continue
            out.append(line)
            i += 1
        return out

    @staticmethod
    def _mov_reg_to_mem(line: Line) -> tuple[str, str] | None:
        """Return (addr, reg) if this is `mov [<addr>], <reg>`, else
        None. Reg must be a 32-bit gp register."""
        if line.kind != "instr" or line.op != "mov":
            return None
        parts = _operands_split(line.operands)
        if parts is None:
            return None
        dest, src = parts
        if not dest.startswith("["):
            return None
        # Source must be a 32-bit gp register (not a sub-byte alias —
        # sub-byte stores have different widths).
        src_lower = src.lower()
        if src_lower not in {"eax", "ebx", "ecx", "edx",
                              "esi", "edi", "ebp", "esp"}:
            return None
        # Normalize address whitespace for matching.
        return (re.sub(r"\s+", "", dest), src_lower)

    @staticmethod
    def _mov_mem_to_reg(line: Line) -> tuple[str, str] | None:
        """Return (reg, addr) if this is `mov <reg>, [<addr>]`, else
        None."""
        if line.kind != "instr" or line.op != "mov":
            return None
        parts = _operands_split(line.operands)
        if parts is None:
            return None
        dest, src = parts
        dest_lower = dest.lower()
        if dest_lower not in {"eax", "ebx", "ecx", "edx",
                               "esi", "edi", "ebp", "esp"}:
            return None
        if not src.startswith("["):
            return None
        return (dest_lower, re.sub(r"\s+", "", src))

    def _pass_right_operand_retarget(self, lines: list[Line]) -> list[Line]:
        """Retarget the binop RHS chain from EAX to ECX, then drop
        the entire `push eax / pop eax / mov ecx, eax` save/restore
        scaffold — EAX is preserved naturally because the retargeted
        chain only writes ECX.

        Match (looking back from `mov ecx, eax; pop eax`):
            push    eax                   ← LHS save (chain start marker)
            <fresh EAX write>             ← chain instr 1: mov/lea/pop/xor
            <EAX read-or-RMW>*            ← chain instrs 2..N: any op
                                            with dest=eax that doesn't
                                            read ECX/CX/CL/CH.
            mov     ecx, eax              ← redundant copy
            pop     eax                   ← restore LHS (also redundant)

        Replace with:
            <retargeted chain instr 1>    ← dest changed to ECX
            <retargeted chain instrs 2..N> ← dest + [eax]→[ecx] in src

        Saves 4 bytes per binop with non-trivial RHS: 1 (push eax) +
        2 (mov ecx, eax) + 1 (pop eax) = 4. The chain's running value
        now lives in ECX from the start; EAX's LHS value is preserved
        across the chain because no chain instruction touches EAX.

        Conditions:
        - All instructions in the chain must write EAX as their dest.
        - Source operands must not reference ECX/CX/CL/CH (else the
          retargeted version would self-reference).
        - The chain's first instruction must be a "fresh write"
          (mov / lea / pop / xor reg+reg / call) — otherwise the
          chain depends on EAX's prior value (the LHS), which post-
          retarget wouldn't be in ECX.
        - Bound the backward scan at 12 instructions (any reasonable
          RHS chain).
        """
        out = list(lines)
        i = 0
        while i < len(out):
            line = out[i]
            if not (line.kind == "instr"
                    and line.op == "mov"
                    and line.operands.replace(" ", "").lower()
                        == "ecx,eax"):
                i += 1
                continue
            # Next instr must be `pop eax`.
            instrs_after = self._next_n_instrs(out, i + 1, 1)
            if (instrs_after is None
                    or instrs_after[0][1].op != "pop"
                    or instrs_after[0][1].operands.strip().lower()
                        != "eax"):
                i += 1
                continue
            pop_idx = instrs_after[0][0]
            # Backward scan to find the chain.
            chain_info = self._find_rhs_chain(out, i)
            if chain_info is None:
                i += 1
                continue
            push_idx, chain_indices = chain_info
            # Retarget every chain instruction (dest + src EAX → ECX
            # for all aliases).
            for k in chain_indices:
                out[k] = self._retarget_instr_eax_to_ecx(out[k])
            # Drop in this order (back to front so indices stay valid):
            # pop_idx > i > push_idx (always true).
            out.pop(pop_idx)
            out.pop(i)
            out.pop(push_idx)
            self.stats["right_operand_retarget"] = (
                self.stats.get("right_operand_retarget", 0) + 1
            )
            # Next iteration: start scanning from where push was
            # (now occupied by the first chain instruction).
            i = push_idx
            continue
        return out

    @staticmethod
    def _find_rhs_chain(
        out: list[Line], mov_ecx_idx: int,
    ) -> tuple[int, list[int]] | None:
        """Backward-scan from mov_ecx_idx looking for a chain of
        EAX-writing instructions terminated by `push eax`. Return
        (push_eax_index, chain_indices_in_source_order), or None."""
        chain: list[int] = []
        j = mov_ecx_idx - 1
        bound = max(0, mov_ecx_idx - 12)
        while j >= bound:
            ln = out[j]
            if ln.kind in ("blank", "comment"):
                j -= 1
                continue
            if ln.kind != "instr":
                return None  # label/directive — basic block boundary
            # `push eax` is the chain-start marker.
            if (ln.op == "push"
                    and ln.operands.strip().lower() == "eax"):
                if not chain:
                    return None  # empty chain — nothing to retarget
                # Chain must start with a fresh EAX write (the first
                # chain instr in source order is at chain[-1] since
                # we built it backward).
                first_ln = out[chain[-1]]
                if not PeepholeOptimizer._is_fresh_eax_write(first_ln):
                    return None
                return (j, list(reversed(chain)))
            # Otherwise this should be a chain instruction.
            if not PeepholeOptimizer._is_chain_eax_writer(ln):
                return None
            chain.append(j)
            j -= 1
        return None

    @staticmethod
    def _is_chain_eax_writer(line: Line) -> bool:
        """Is this an instruction that writes EAX as its destination
        and doesn't read ECX/CX/CL/CH? Eligible for retargeting to
        ECX. Also rejects ESP-relative sources because the surrounding
        push/pop that bracket the chain affect ESP — dropping them
        changes [esp + N] semantics."""
        if line.kind != "instr":
            return False
        parts = _operands_split(line.operands)
        if parts is None:
            # Single-operand or no-operand — must inspect by op.
            # neg/not/imul (one-op form) have a single operand.
            if line.op in {"neg", "not", "inc", "dec",
                           "mul", "imul", "div", "idiv"}:
                operand = line.operands.strip().lower()
                if operand == "eax":
                    # Unary on EAX — writes EAX.
                    if PeepholeOptimizer._references_ecx(operand):
                        return False
                    return True
            return False
        dest, src = parts
        dest_lower = dest.lower()
        if dest_lower != "eax":
            return False
        # Source mustn't reference ECX. (Sources that reference [eax]
        # are fine — they'll be rewritten in retarget.)
        if PeepholeOptimizer._references_ecx(src):
            return False
        # Source mustn't be ESP-relative. The retarget drops the
        # push/pop scaffold that surrounds the chain; if the chain
        # reads `[esp + N]`, dropping the push (which decremented
        # ESP by 4) changes which byte is read. Conservative skip.
        if _references_register(src, "esp"):
            return False
        # cmp/test read EAX without writing — exclude.
        if line.op in {"cmp", "test"}:
            return False
        # `xchg eax, X` swaps — read+write both. Skip.
        if line.op == "xchg":
            return False
        # `cdq` writes EDX based on EAX's sign. EAX unchanged. Skip.
        if line.op == "cdq":
            return False
        return True

    @staticmethod
    def _is_fresh_eax_write(line: Line) -> bool:
        """Does this instruction write EAX without reading it first
        (a fresh write rather than a read-modify-write)? The source
        operand must NOT reference EAX/AX/AL/AH — otherwise the
        instruction depends on EAX's prior value and isn't fresh."""
        if line.kind != "instr":
            return False
        if line.op in {"mov", "lea", "movsx", "movzx"}:
            parts = _operands_split(line.operands)
            if parts and parts[0].lower() == "eax":
                src = parts[1]
                if PeepholeOptimizer._references_eax_low_high(src):
                    return False
                return True
            return False
        if line.op == "pop" and line.operands.strip().lower() == "eax":
            return True
        if line.op == "xor":
            parts = _operands_split(line.operands)
            if (parts and parts[0].lower() == "eax"
                    and parts[1].strip().lower() == "eax"):
                return True
            return False
        if line.op == "call":
            # Direct call clobbers EAX with the callee's return value.
            # No EAX read at the call site (cdecl args are on stack).
            # Indirect call `call eax` reads EAX — exclude.
            operand = line.operands.strip().lower()
            if operand in {"eax", "ax", "al", "ah"}:
                return False
            if "[" in operand and "eax" in operand:
                # `call [eax + N]` reads EAX as part of the address.
                return False
            return True
        return False

    @staticmethod
    def _references_ecx(text: str) -> bool:
        """Does `text` reference ECX or any of its aliases (CX/CL/CH)?"""
        for alias in ("ecx", "cx", "cl", "ch"):
            if _references_register(text, alias):
                return True
        return False

    @staticmethod
    def _references_eax_low_high(text: str) -> bool:
        """Does `text` reference EAX or any of its aliases (AX/AL/AH)?"""
        for alias in ("eax", "ax", "al", "ah"):
            if _references_register(text, alias):
                return True
        return False

    # Mapping from EAX-family register names to ECX-family equivalents.
    # Order matters for the substitution: `eax` first so we don't
    # accidentally match the `ax` inside `eax`. (Word-boundary `\b`
    # would also handle this, but explicit ordering is safer.)
    _EAX_TO_ECX_ALIASES: list[tuple[str, str]] = [
        ("eax", "ecx"),
        ("ax", "cx"),
        ("al", "cl"),
        ("ah", "ch"),
    ]

    @staticmethod
    def _retarget_instr_eax_to_ecx(line: Line) -> Line:
        """Rewrite every EAX-family register reference in operands and
        raw line to the corresponding ECX-family register (eax→ecx,
        ax→cx, al→cl, ah→ch). The chain's running value lives in
        ECX after retarget; any `[eax]` becomes `[ecx]` for the same
        reason."""
        new_operands = line.operands
        new_raw = line.raw
        for old, new in PeepholeOptimizer._EAX_TO_ECX_ALIASES:
            new_operands = re.sub(
                rf"\b{old}\b", new, new_operands, flags=re.IGNORECASE
            )
            new_raw = re.sub(
                rf"\b{old}\b", new, new_raw, flags=re.IGNORECASE
            )
        return Line(
            raw=new_raw, kind=line.kind, op=line.op,
            operands=new_operands,
        )

    def _pass_cmp_zero_to_test(self, lines: list[Line]) -> list[Line]:
        """Replace `cmp <reg>, 0` with `test <reg>, <reg>`.

        Both forms set flags identically for the comparison-with-zero
        case:
        - ZF: 1 iff reg == 0
        - SF: 1 iff reg's high bit is set (signed-negative)
        - OF: 0 (no overflow possible)
        - CF: 0 (no borrow possible)
        - PF: parity of reg's low byte

        Subsequent JCCs (je/jne/jl/jge/js/jns/etc.) all see the same
        flag state, so the rewrite preserves semantics.

        Saves 1 byte per match: `cmp eax, 0` is 3 bytes (with imm8
        sign-extension); `test eax, eax` is 2 bytes.
        """
        out: list[Line] = []
        for line in lines:
            if (line.kind == "instr" and line.op == "cmp"):
                parts = _operands_split(line.operands)
                if parts is not None:
                    dest, src = parts
                    if (src.strip() == "0"
                            and dest.lower() in {"eax", "ebx", "ecx",
                                                 "edx", "esi", "edi",
                                                 "ebp", "esp"}):
                        leading_ws = re.match(r"^\s*", line.raw).group(0)
                        new_raw = (
                            f"{leading_ws}test    {dest}, {dest}"
                        )
                        out.append(Line(
                            raw=new_raw, kind="instr", op="test",
                            operands=f"{dest}, {dest}",
                        ))
                        self.stats["cmp_zero_to_test"] = (
                            self.stats.get("cmp_zero_to_test", 0) + 1
                        )
                        continue
            out.append(line)
        return out

    # Mapping from each 32-bit gp reg to its sub-register aliases
    # (16-bit and 8-bit). Used by dead-mov / liveness analyses.
    _REG_FAMILY: dict[str, frozenset[str]] = {
        "eax": frozenset({"eax", "ax", "al", "ah"}),
        "ebx": frozenset({"ebx", "bx", "bl", "bh"}),
        "ecx": frozenset({"ecx", "cx", "cl", "ch"}),
        "edx": frozenset({"edx", "dx", "dl", "dh"}),
        "esi": frozenset({"esi", "si"}),
        "edi": frozenset({"edi", "di"}),
        "ebp": frozenset({"ebp", "bp"}),
        "esp": frozenset({"esp", "sp"}),
    }

    @staticmethod
    def _references_reg_family(text: str, reg32: str) -> bool:
        """Does `text` reference reg32 or any of its sub-aliases?"""
        for alias in PeepholeOptimizer._REG_FAMILY[reg32]:
            if _references_register(text, alias):
                return True
        return False

    @staticmethod
    def _is_pure_reg_write(line: Line, reg32: str) -> bool:
        """Does this instruction write reg32 in full WITHOUT reading
        any of its sub-aliases first? Pure writes:
        - mov reg, src where src doesn't reference reg-family
        - lea reg, [...] where the address doesn't reference reg-family
        - pop reg
        - xor reg, reg
        - movsx reg, src / movzx reg, src where src doesn't reference reg-family
        - call <label> (clobbers EAX/ECX/EDX in cdecl, doesn't read reg)
          for reg ∈ {eax, ecx, edx}
        """
        if line.kind != "instr":
            return False
        family = PeepholeOptimizer._REG_FAMILY[reg32]
        if line.op in {"mov", "lea", "movsx", "movzx"}:
            parts = _operands_split(line.operands)
            if not parts:
                return False
            dest, src = parts
            if dest.lower() != reg32:
                return False
            return not PeepholeOptimizer._references_reg_family(src, reg32)
        if line.op == "pop":
            if line.operands.strip().lower() == reg32:
                return True
            return False
        if line.op == "xor":
            parts = _operands_split(line.operands)
            if (parts and parts[0].lower() == reg32
                    and parts[1].strip().lower() == reg32):
                return True
            return False
        if line.op == "call":
            # Direct call clobbers caller-saved EAX/ECX/EDX.
            # Indirect call via reg-family reads it — exclude.
            operand = line.operands.strip().lower()
            if operand in family:
                return False
            if "[" in operand:
                # `call [reg + N]` reads reg.
                if PeepholeOptimizer._references_reg_family(operand, reg32):
                    return False
            if reg32 in {"eax", "ecx", "edx"}:
                return True
        # `cdq` sign-extends EAX into EDX — pure write to EDX (doesn't
        # read EDX's prior value, only EAX's).
        if line.op == "cdq" and reg32 == "edx":
            return True
        return False

    # Instructions known to reference registers IMPLICITLY (beyond
    # what's in the operand string). Used to know when our reg-
    # tracking peepholes need to bail. Examples: cdq reads EAX, idiv
    # reads EDX:EAX, lodsd reads ESI, rep* reads ECX.
    _IMPLICIT_REG_USERS: frozenset[str] = frozenset({
        # Sign/zero extension family
        "cdq", "cwde", "cwd", "cbw", "cqo", "cdqe",
        # One-operand mul/div implicitly read/write EDX:EAX
        "mul", "div", "idiv",  # (one-operand form)
        # String operations: read/write ESI, EDI, ECX
        "movsb", "movsw", "movsd",  # NB: movsd here is the
                                       # string instr, not SSE.
                                       # NASM uses `movsd` for both
                                       # but in 32-bit mode without
                                       # operands it's the string op.
        "lodsb", "lodsw", "lodsd",
        "stosb", "stosw", "stosd",
        "scasb", "scasw", "scasd",
        "cmpsb", "cmpsw", "cmpsd",
        # Repeat prefixes (read ECX)
        "rep", "repe", "repne", "repnz", "repz",
        # Loop instructions
        "loop", "loope", "loopne", "loopnz", "loopz",
        # XLAT reads AL, EBX
        "xlat", "xlatb",
        # Flag ops
        "pushf", "pushfd", "popf", "popfd", "lahf", "sahf",
        # bswap reads+writes a reg (single operand, but the operand
        # form is fine — we'll detect via the operand string)
        # enter/leave touch ESP/EBP
        "enter",  # leave is in the explicit-fine list
        # FPU instructions (don't touch gp regs by themselves)
        # but `fnstsw ax` writes AX
        "fnstsw", "fstsw",
        # int / iret / sysenter / sysexit — system calls
        "int", "into", "int1", "int3",
    })

    @staticmethod
    def _has_explicit_operands_only(line: Line) -> bool:
        """Does this instruction's behavior depend ONLY on its
        explicit operands (and not on implicit register references)?
        Used to know when scanning past an instruction without
        finding the tracked reg in the operand string is safe."""
        if line.op in PeepholeOptimizer._IMPLICIT_REG_USERS:
            return False
        # `fistp` and friends? FPU instructions don't touch gp regs.
        # An instruction starting with `f` (FPU) is fine.
        if line.op.startswith("f"):
            # Excludes the few `f*` we listed in _IMPLICIT_REG_USERS
            # like fnstsw — those are caught above.
            return True
        return True

    def _reg_dead_after(
        self, lines: list[Line], start_idx: int, reg32: str,
        visited_labels: frozenset[str] = frozenset(),
        depth: int = 0,
    ) -> bool:
        """Forward scan from start_idx. Return True if `reg32` is
        provably dead — a pure-write happens before any read.

        CFG-aware: at an unconditional `jmp X`, jumps to X (does not
        continue text-fallthrough). At a label by text-fallthrough,
        crosses transparently (the label may have other predecessors
        but they all reach the same first instruction, so dead-ness
        is determined by that instruction).

        The codegen invariant — registers are not assumed to carry
        values across labels — means cross-label scanning is safe
        for the reg liveness question.
        """
        if depth > 5:
            return False
        family = self._REG_FAMILY[reg32]
        scanned = 0
        j = start_idx
        while j < len(lines) and scanned < 20:
            ln = lines[j]
            if ln.kind in ("blank", "comment", "label"):
                # Codegen invariant: regs are not assumed live at any
                # label by predecessor jumps — any user of reg loads it
                # fresh. So crossing labels by text-fallthrough is OK.
                j += 1
                continue
            if ln.kind in ("directive", "data"):
                # End of function (or section change). The function's
                # return path was reached without finding a use.
                if reg32 == "ecx":
                    return True
                return False
            if ln.kind != "instr":
                return False
            scanned += 1
            # Check pure-write FIRST (before checking reads), since
            # mov reg, src can be a pure write OR self-RMW.
            if PeepholeOptimizer._is_pure_reg_write(ln, reg32):
                return True
            # Function exit terminators.
            if ln.op in {"ret", "iret", "iretd", "retf", "retn"}:
                # EAX/EDX may be the return value (live).
                # EBX/ESI/EDI/EBP are callee-saved (caller expects
                # them preserved — dropping a write inside the function
                # leaves them at the caller's value, which is fine).
                # ECX is caller-saved scratch.
                if reg32 in ("eax", "edx"):
                    return False
                return True
            # If this instruction references the reg family in any
            # operand, treat as a read.
            if self._references_reg_family(ln.operands, reg32):
                return False
            # `jmp <label>` (direct, unconditional): chase the target.
            # DO NOT fall through to text-next — that's unreachable
            # from this CFG path.
            if ln.op == "jmp":
                target = _jmp_target(ln)
                if target is None:
                    return False  # indirect jmp — bail
                if target in visited_labels:
                    return False  # loop — bail
                target_idx = self._find_label_idx(lines, target)
                if target_idx is None:
                    return False
                # Tail-call into the scan at the label's first
                # post-label instruction. Tail-call style avoids
                # unbounded recursion when chains of jmps are short.
                return self._reg_dead_after(
                    lines, target_idx + 1, reg32,
                    visited_labels | frozenset({target}),
                    depth + 1,
                )
            # Conditional jumps split control flow. Both successors
            # must reach a pure-write before any read. Recurse on the
            # target path; continue scanning the fallthrough.
            if ln.op.startswith("j") and ln.op != "jmp":
                target = _branch_target(ln)
                if target is None:
                    return False  # indirect/unparseable — bail
                if target in visited_labels:
                    return False  # loop — bail
                target_idx = self._find_label_idx(lines, target)
                if target_idx is None:
                    return False
                # Target path: must be dead.
                target_dead = self._reg_dead_after(
                    lines, target_idx + 1, reg32,
                    visited_labels | frozenset({target}),
                    depth + 1,
                )
                if not target_dead:
                    return False
                # Fallthrough path: keep scanning.
                j += 1
                continue
            # `call <label>` clobbers caller-saved regs (EAX/ECX/EDX).
            if ln.op == "call":
                operand = ln.operands.strip().lower()
                if operand in family:
                    # Indirect via reg — reads it.
                    return False
                if "[" in operand and self._references_reg_family(
                        operand, reg32):
                    return False
                if reg32 in {"eax", "ecx", "edx"}:
                    return True
                # Reg preserved across cdecl call. Continue.
                j += 1
                continue
            # If this instruction can have implicit register reads
            # (e.g. cdq, idiv, lodsd, rep movsd) — bail conservatively.
            if not PeepholeOptimizer._has_explicit_operands_only(ln):
                return False
            # Otherwise: explicit-operand instr that doesn't reference
            # reg, didn't write it purely. Just move on.
            j += 1
        return False

    @staticmethod
    def _find_label_idx(lines: list[Line], label: str) -> int | None:
        """Find the index of label `label` in `lines`. Returns the
        index of the label line (the next instr is at idx+1)."""
        for k, ln in enumerate(lines):
            if ln.kind == "label" and ln.label == label:
                return k
        return None

    def _pass_dead_mov_to_reg(self, lines: list[Line]) -> list[Line]:
        """Drop `mov eax, <src>` (or `lea eax, ...`) when the loaded
        value is provably dead — a pure overwrite of EAX follows
        before any read.

        Common in postfix `i++` in for-loop step where the value is
        discarded:
            mov eax, [ebp - 4]   ← this load
            inc dword [ebp - 4]
            jmp .L1_for_top      ← .L1_for_top: mov eax, [...] (overwrites)

        Restricted to EAX because uc386 has a static-link convention
        (nested-function trampolines) where ECX is an implicit input
        to certain calls — dropping a `mov ecx, X` before such a
        call would break the call. EBX/ESI/EDI/EBP are callee-saved
        and rarely have dead writes worth dropping.

        Conditions:
        - Source must not reference EAX/AX/AL/AH (would be self-RMW).
        - Forward scan finds a pure-write to EAX before any read.
        - At a `ret`, EAX is live (return value).
        """
        out: list[Line] = []
        for i, line in enumerate(lines):
            if line.kind == "instr" and line.op in {"mov", "lea",
                                                     "movsx", "movzx"}:
                parts = _operands_split(line.operands)
                if parts is not None:
                    dest, src = parts
                    dest_lower = dest.lower()
                    if dest_lower == "eax":
                        # When src references eax-family, this is a
                        # self-RMW. If src is a memory deref (e.g.,
                        # ``mov eax, [eax]``), the read could have
                        # observable effects (page faults, MMIO) —
                        # skip dropping. If src is a sub-register
                        # (e.g., ``movsx eax, al``), no memory access,
                        # safe to drop when EAX is dead.
                        src_is_self = self._references_reg_family(
                            src, "eax"
                        )
                        src_is_mem = "[" in src
                        if not (src_is_self and src_is_mem):
                            if self._reg_dead_after(lines, i + 1,
                                                     "eax"):
                                self.stats["dead_mov_to_reg"] = (
                                    self.stats.get("dead_mov_to_reg", 0)
                                    + 1
                                )
                                continue
            out.append(line)
        return out

    def _pass_prologue_to_enter(self, lines: list[Line]) -> list[Line]:
        """P-prologue-to-enter: collapse the standard cdecl prologue
        ``push ebp; mov ebp, esp; sub esp, IMM`` to a single
        ``enter IMM, 0`` instruction. Saves 2 bytes when IMM fits in
        imm8, 5 bytes when IMM is imm32. Identical semantics — Intel
        defines ``enter IMM, 0`` to be exactly the three-instruction
        sequence (no extra register touches at level=0).

        Constraints:
        - Three consecutive ``instr`` lines (no intervening comments,
          blanks, or labels).
        - IMM must be a literal in (0, 65535] (the imm16 width of the
          first ``enter`` operand).
        """
        out: list[Line] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if (
                i + 2 < len(lines)
                and line.kind == "instr"
                and line.op == "push"
                and line.operands.strip().lower() == "ebp"
                and lines[i + 1].kind == "instr"
                and self._is_mov_ebp_esp(lines[i + 1])
                and lines[i + 2].kind == "instr"
                and lines[i + 2].op == "sub"
            ):
                imm = self._extract_sub_esp_imm(lines[i + 2])
                if imm is not None and 0 < imm <= 65535:
                    indent = self._extract_indent(line.raw)
                    new_raw = f"{indent}enter   {imm}, 0"
                    new_line = Line(
                        raw=new_raw,
                        kind="instr",
                        op="enter",
                        operands=f"{imm}, 0",
                    )
                    out.append(new_line)
                    self.stats["prologue_to_enter"] = (
                        self.stats.get("prologue_to_enter", 0) + 1
                    )
                    i += 3
                    continue
            out.append(line)
            i += 1
        return out

    @staticmethod
    def _is_mov_ebp_esp(line: Line) -> bool:
        if line.op != "mov":
            return False
        parts = _operands_split(line.operands)
        if parts is None:
            return False
        dest, src = parts
        return dest.strip().lower() == "ebp" and src.strip().lower() == "esp"

    @staticmethod
    def _extract_sub_esp_imm(line: Line) -> int | None:
        if line.op != "sub":
            return None
        parts = _operands_split(line.operands)
        if parts is None:
            return None
        dest, src = parts
        if dest.strip().lower() != "esp":
            return None
        s = src.strip()
        # Reject memory derefs and bare register names.
        if "[" in s:
            return None
        try:
            if s.lower().startswith("0x"):
                return int(s, 16)
            return int(s)
        except ValueError:
            return None

    @staticmethod
    def _extract_indent(raw: str) -> str:
        n = 0
        while n < len(raw) and raw[n] in " \t":
            n += 1
        return raw[:n]

    def _pass_redundant_eax_load(self, lines: list[Line]) -> list[Line]:
        """Drop ``mov eax, M`` when EAX already provably holds M.
        See ``_run_redundant_reg_load`` for the algorithm."""
        return self._run_redundant_reg_load(
            lines, "eax", {"al", "ah", "ax"}, "redundant_eax_load",
        )

    def _pass_redundant_ecx_load(self, lines: list[Line]) -> list[Line]:
        """Drop ``mov ecx, M`` when ECX already provably holds M.
        Sister of ``_pass_redundant_eax_load``. Common after
        ``index_load_collapse`` / ``push_index_collapse`` leaves
        a ``mov ecx, [m]`` whose value is still in ECX from the
        prior load."""
        return self._run_redundant_reg_load(
            lines, "ecx", {"cl", "ch", "cx"}, "redundant_ecx_load",
        )

    def _run_redundant_reg_load(
        self, lines: list[Line], reg32: str,
        sub_regs: set[str], stat_key: str,
    ) -> list[Line]:
        """Drop ``mov REG, M`` when REG already provably holds M's
        value — we loaded it earlier in the basic block and nothing
        between then and now wrote REG or touched M.

        Tracking is conservative: we follow REG's "current memory
        source" through a single straight-line block. Any of these
        invalidate the tracker:
        - Any write to REG (or its sub-regs) by any instruction
        - Any memory-write whose destination might alias M
        - Calls and unconditional control flow (jmp/ret/etc.)
        - Any instruction in ``_IMPLICIT_REG_USERS`` (over-
          conservative — some only touch other regs, but safe)
        - Any label that's the target of a BACKWARD jump (loop top —
          the back-edge means downstream stores to M may invalidate
          the cached value before re-entry)

        Conditional jumps (jcc) DO preserve the tracker on the
        fall-through path; we record the jcc's reg_mem at the time
        of the jump, keyed by target label. At each label, we merge
        the jcc-recorded states with the fall-through state (if any).

        For aliasing, we only trust two stack expressions to be
        disjoint when both are literal ``[ebp + N]`` / ``[ebp - N]``
        offsets (different N). Stores via a register address (e.g.
        ``[ecx]``), or to a label, conservatively invalidate.

        Tracked operands: literal-offset stack reads. Register
        sources, immediates, and labels are not tracked because they
        don't alias-conflict with anything codegen emits (immediates)
        or aren't worth tracking (registers — codegen reloads from
        memory, not register-to-register).
        """
        # Pre-compute labels that are targets of backward jumps.
        # At those labels, we conservatively reset reg_mem because
        # the back-edge may have stores that invalidate cached
        # state before re-entry.
        label_positions: dict[str, int] = {}
        for k, ln in enumerate(lines):
            if ln.kind == "label":
                label_positions[ln.label] = k
        loop_top_labels: set[str] = set()
        for k, ln in enumerate(lines):
            if ln.kind == "instr" and (
                ln.op == "jmp" or ln.op.startswith("j")
            ):
                target = _branch_target(ln)
                if target is None:
                    continue
                target_idx = label_positions.get(target)
                if target_idx is not None and target_idx <= k:
                    loop_top_labels.add(target)

        out: list[Line] = []
        reg_mem: str | None = None  # The literal text of REG's known mem source
        jcc_states: dict[str, set[str | None]] = {}
        prev_unconditional = False
        for line in lines:
            if line.kind != "instr":
                if line.kind == "label":
                    label = line.label
                    saved = jcc_states.pop(label, None)
                    if label in loop_top_labels:
                        # Loop top: a back-edge may store to memory
                        # between iterations, invalidating any cached
                        # reg_mem value. Conservatively reset.
                        reg_mem = None
                    elif prev_unconditional:
                        if saved is not None and len(saved) == 1:
                            (only,) = saved
                            reg_mem = only
                        else:
                            reg_mem = None
                    else:
                        if saved is None:
                            pass
                        else:
                            states = saved | {reg_mem}
                            if len(states) == 1:
                                (only,) = states
                                reg_mem = only
                            else:
                                reg_mem = None
                    prev_unconditional = False
                out.append(line)
                continue
            op = line.op
            ops = line.operands
            if op == "mov":
                parts = _operands_split(ops)
                if parts is not None:
                    dest, src = parts
                    dest_low = dest.strip().lower()
                    src_norm = src.strip()
                    if dest_low == reg32:
                        # mov REG, X
                        if (
                            reg_mem is not None
                            and self._is_ebp_offset_mem(src_norm)
                            and src_norm == reg_mem
                        ):
                            self.stats[stat_key] = (
                                self.stats.get(stat_key, 0) + 1
                            )
                            continue
                        if self._is_ebp_offset_mem(src_norm):
                            reg_mem = src_norm
                        else:
                            reg_mem = None
                        out.append(line)
                        continue
                    if "[" in dest:
                        if reg_mem is not None and not self._mem_disjoint(
                            reg_mem, dest.strip()
                        ):
                            reg_mem = None
                        if dest_low in sub_regs:
                            reg_mem = None
                        out.append(line)
                        continue
                    if dest_low in sub_regs:
                        reg_mem = None
                    out.append(line)
                    continue
            if op == "call":
                reg_mem = None
                prev_unconditional = False
                out.append(line)
                continue
            if op in {"jmp", "ret", "iret", "iretd", "retf",
                      "retn", "leave", "enter"}:
                if op == "jmp":
                    target = line.operands.strip()
                    jcc_states.setdefault(target, set()).add(reg_mem)
                reg_mem = None
                prev_unconditional = op != "enter"
                out.append(line)
                continue
            if op.startswith("j"):
                target = line.operands.strip()
                jcc_states.setdefault(target, set()).add(reg_mem)
                prev_unconditional = False
                out.append(line)
                continue
            if op in {"cmp", "test", "push"}:
                out.append(line)
                continue
            if (
                self._references_reg_family(ops, reg32)
                or op in PeepholeOptimizer._IMPLICIT_REG_USERS
            ):
                reg_mem = None
                out.append(line)
                continue
            if "[" in ops:
                parts = _operands_split(ops)
                if parts is not None:
                    dest, _ = parts
                    if "[" in dest and reg_mem is not None and not (
                        self._mem_disjoint(reg_mem, dest.strip())
                    ):
                        reg_mem = None
                else:
                    if reg_mem is not None:
                        reg_mem = None
            out.append(line)
        return out

    @staticmethod
    def _is_ebp_offset_mem(text: str) -> bool:
        """Match exactly `[ebp + N]` or `[ebp - N]` (or `dword [...]`,
        etc.) where N is a numeric literal."""
        # Strip optional size prefix (`dword`, `word`, `byte`).
        stripped = text.strip()
        for prefix in ("dword ", "word ", "byte ", "qword "):
            if stripped.lower().startswith(prefix):
                stripped = stripped[len(prefix):].lstrip()
                break
        return bool(re.fullmatch(
            r"\[\s*ebp\s*[+-]\s*\d+\s*\]", stripped, re.IGNORECASE
        ))

    @staticmethod
    def _mem_disjoint(a: str, b: str) -> bool:
        """Conservative aliasing check: True if `a` and `b` are
        provably-disjoint memory references. Only handles literal
        ``[ebp + N]`` offsets (different N → disjoint).

        Returns False (= "may alias") for anything we can't reason
        about (register-base derefs, labels, etc.).
        """
        a_off = PeepholeOptimizer._ebp_offset(a)
        b_off = PeepholeOptimizer._ebp_offset(b)
        if a_off is None or b_off is None:
            return False
        return a_off != b_off

    @staticmethod
    def _ebp_offset(text: str) -> int | None:
        """Extract the offset from `[ebp + N]` / `[ebp - N]`. Returns
        None for non-matching forms."""
        stripped = text.strip()
        for prefix in ("dword ", "word ", "byte ", "qword "):
            if stripped.lower().startswith(prefix):
                stripped = stripped[len(prefix):].lstrip()
                break
        m = re.fullmatch(
            r"\[\s*ebp\s*([+-])\s*(\d+)\s*\]", stripped, re.IGNORECASE
        )
        if not m:
            return None
        sign, num = m.group(1), m.group(2)
        return int(num) if sign == "+" else -int(num)

    def _pass_label_offset_fold(self, lines: list[Line]) -> list[Line]:
        """Collapse ``mov reg, LABEL; add reg, IMM`` (or ``sub reg, IMM``)
        into ``mov reg, LABEL ± IMM``. NASM resolves the ``LABEL ± IMM``
        at assemble time, producing a single 5-byte ``mov reg, imm32``
        and eliminating the 3-byte add/sub. Saves 3 bytes per match.

        Common in pointer arithmetic on globals: ``mov eax, _g;
        add eax, 8`` to compute ``&g[2]`` or ``&s.field``.

        Conditions:
        - First line is ``mov R, X`` where X is non-numeric, non-memory,
          non-register (i.e., a label or label-relative expression).
        - Second line is ``add R, IMM`` or ``sub R, IMM`` (numeric
          literal immediate; not a memory or register).
        - Same R.
        - The flags-after-add are dead (no Jcc / setCC / cmov / adc /
          sbb reading them before they're overwritten). The folded
          ``mov`` doesn't set flags, so the original add/sub's flag
          effects must be unobserved.
        """
        out: list[Line] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if (
                i + 1 < len(lines)
                and line.kind == "instr"
                and line.op == "mov"
            ):
                parts = _operands_split(line.operands)
                if parts is not None:
                    dest, src = parts
                    dest_low = dest.strip().lower()
                    src_norm = src.strip()
                    if (
                        self._is_general_register(dest_low)
                        and self._is_label_like(src_norm)
                    ):
                        nxt = lines[i + 1]
                        if (
                            nxt.kind == "instr"
                            and nxt.op in ("add", "sub")
                        ):
                            nxt_parts = _operands_split(nxt.operands)
                            if nxt_parts is not None:
                                ndest, nsrc = nxt_parts
                                imm = self._parse_numeric_immediate(
                                    nsrc.strip()
                                )
                                if (
                                    ndest.strip().lower() == dest_low
                                    and imm is not None
                                    and self._flags_safe_after(lines, i + 2)
                                ):
                                    op_sign = "+" if nxt.op == "add" else "-"
                                    indent = self._extract_indent(line.raw)
                                    new_src = f"{src_norm} {op_sign} {imm}"
                                    new_raw = (
                                        f"{indent}mov     "
                                        f"{dest.strip()}, {new_src}"
                                    )
                                    new_line = Line(
                                        raw=new_raw,
                                        kind="instr",
                                        op="mov",
                                        operands=(
                                            f"{dest.strip()}, {new_src}"
                                        ),
                                    )
                                    out.append(new_line)
                                    self.stats["label_offset_fold"] = (
                                        self.stats.get(
                                            "label_offset_fold", 0
                                        ) + 1
                                    )
                                    i += 2
                                    continue
            out.append(line)
            i += 1
        return out

    @staticmethod
    def _is_general_register(text: str) -> bool:
        return text.strip().lower() in {
            "eax", "ebx", "ecx", "edx", "esi", "edi", "ebp", "esp",
        }

    @staticmethod
    def _is_label_like(text: str) -> bool:
        """True if text looks like a label or label-arithmetic expr —
        not a memory deref, not a pure numeric literal, not a register
        name."""
        s = text.strip()
        if not s:
            return False
        if s.startswith("[") or s.endswith("]"):
            return False
        if PeepholeOptimizer._is_general_register(s):
            return False
        if s.lower() in {
            "al", "bl", "cl", "dl", "ah", "bh", "ch", "dh",
            "ax", "bx", "cx", "dx", "si", "di", "bp", "sp",
        }:
            return False
        # Pure numeric literal (decimal, hex, octal, binary)?
        try:
            int(s, 0)
            return False
        except ValueError:
            pass
        # Looks like a label (or label expression). Could be `_foo`,
        # `_foo + 4`, `.L1_top`, `__heap`, etc.
        return True

    @staticmethod
    def _parse_numeric_immediate(text: str) -> int | None:
        """Parse `42`, `0x1A`, `0o17`, `0b101`. Returns None for
        anything else (registers, memory, labels, expressions)."""
        s = text.strip()
        if not s:
            return None
        try:
            return int(s, 0)
        except ValueError:
            return None

    def _pass_cmp_load_collapse(self, lines: list[Line]) -> list[Line]:
        """Collapse ``mov reg, [mem]; ...chain...; cmp reg, X`` into
        ``...chain...; cmp dword [mem], X`` when ``reg`` is dead
        after the cmp AND the chain doesn't read/write reg or write
        to [mem]. Also handles ``mov reg, [mem]; ...chain...; test
        reg, reg`` → ``...chain...; cmp dword [mem], 0`` (test r,r
        and cmp r,0 set the same flags for standard Jcc).

        x86 supports ``cmp r/m32, imm`` and ``cmp r/m32, r32``
        directly, so the load can be elided. Saves 2-3 bytes per
        match (drops the mov; cmp gains a memory operand of equal
        or 1-byte-larger encoding).

        The fast path matches the immediate-adjacent shape (chain
        length 0). The slow path walks forward through a chain,
        bailing on:
        - any instruction that reads or writes EAX (we'd lose the
          loaded value or the chain would clobber it).
        - any memory store that may alias [mem].
        - any control flow (label, jmp, ret, call) — calls clobber
          EAX per cdecl.

        Restricted to EAX for now: most cmp+load chains funnel through
        EAX, and limiting scope keeps the safety analysis simple.
        """
        out: list[Line] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if (
                line.kind == "instr"
                and line.op == "mov"
            ):
                parts = _operands_split(line.operands)
                if parts is not None:
                    dest, src = parts
                    dest_low = dest.strip().lower()
                    src_norm = src.strip()
                    if (
                        dest_low == "eax"
                        and src_norm.startswith("[")
                        and src_norm.endswith("]")
                    ):
                        # Find the cmp/test at the end of the chain.
                        cmp_idx = self._find_cmp_after_load(
                            lines, i + 1, src_norm
                        )
                        if cmp_idx is not None:
                            new_cmp = self._collapse_cmp_with_load(
                                lines[cmp_idx], src_norm
                            )
                            if (
                                new_cmp is not None
                                and self._reg_dead_after(
                                    lines, cmp_idx + 1, "eax"
                                )
                            ):
                                indent = self._extract_indent(
                                    line.raw
                                )
                                new_raw = f"{indent}{new_cmp}"
                                new_line = Line(
                                    raw=new_raw,
                                    kind="instr",
                                    op="cmp",
                                    operands=new_cmp.split(
                                        None, 1
                                    )[1] if " " in new_cmp else "",
                                )
                                # Emit chain (between mov and cmp),
                                # then the new cmp. Drop mov + old
                                # cmp.
                                for k in range(i + 1, cmp_idx):
                                    out.append(lines[k])
                                out.append(new_line)
                                self.stats["cmp_load_collapse"] = (
                                    self.stats.get(
                                        "cmp_load_collapse", 0
                                    ) + 1
                                )
                                i = cmp_idx + 1
                                continue
            out.append(line)
            i += 1
        return out

    def _find_cmp_after_load(
        self, lines: list[Line], start: int, mem: str
    ) -> int | None:
        """Walk forward from `start` looking for `cmp eax, X` (X non-
        memory) or `test eax, eax`. Bail on any chain instruction
        that:
        - references EAX (read or write — dest or operand)
        - stores to memory that may alias `mem`
        - is a label, control flow, or call

        Returns the cmp/test index on success, None on failure.
        """
        for k in range(start, len(lines)):
            ln = lines[k]
            if ln.kind == "label":
                return None
            if ln.kind != "instr":
                continue
            op = ln.op
            # Found the cmp/test?
            if op in ("cmp", "test"):
                opp = _operands_split(ln.operands or "")
                if opp is None:
                    return None
                d, s = opp
                if d.strip().lower() == "eax":
                    return k
                # Some other cmp/test — bail conservatively.
                return None
            # Control flow / call — bail.
            if op in ("ret", "iret", "iretd", "retf", "retn",
                      "leave", "enter", "call"):
                return None
            if op == "jmp" or op.startswith("j"):
                return None
            # Any reference to EAX (or sub-regs) — bail.
            if (
                _references_register(ln.operands or "", "eax")
                or op in PeepholeOptimizer._IMPLICIT_REG_USERS
            ):
                return None
            # Memory write that may alias `mem` — bail. The first
            # operand is the destination for most ALU/move ops; for
            # 1-operand RMW ops (inc/dec/not/neg/pop/setCC) the only
            # operand is the dest. Skip known read-only ops where
            # first operand is just a source.
            ops_text = ln.operands or ""
            if ops_text and op not in self._READ_ONLY_FIRST_MEM:
                first_op = (
                    ops_text.split(",", 1)[0].strip()
                    if "," in ops_text
                    else ops_text.strip()
                )
                if "[" in first_op:
                    if not self._mem_disjoint(
                        self._strip_size_prefix(first_op),
                        self._strip_size_prefix(mem.strip()),
                    ):
                        return None
        return None

    # Ops whose first memory operand is a SOURCE (read-only). All
    # other ops with a memory first operand write to it.
    _READ_ONLY_FIRST_MEM: frozenset[str] = frozenset({
        "cmp", "test",
        "lea",  # dest is reg; mem only as effective addr (read-only)
        "push",
        # Single-operand mul/div/idiv/imul: mem operand is source,
        # result lands in EDX:EAX (caught by IMPLICIT_REG_USERS).
        "mul", "div", "idiv", "imul",
        # Conditional jumps with mem target — control flow, but
        # they're already bailed on at the control-flow check above.
        # Listed here for completeness.
    })

    @staticmethod
    def _collapse_cmp_with_load(
        nxt: Line, src_mem: str
    ) -> str | None:
        """If `nxt` is `cmp eax, X` (X non-memory) or `test eax, eax`,
        return the replacement instruction text. Otherwise None."""
        if nxt.kind != "instr":
            return None
        if nxt.op == "cmp":
            np = _operands_split(nxt.operands)
            if np is None:
                return None
            cdest, csrc = np
            if cdest.strip().lower() != "eax":
                return None
            if "[" in csrc:
                return None
            return f"cmp     dword {src_mem}, {csrc.strip()}"
        if nxt.op == "test":
            np = _operands_split(nxt.operands)
            if np is None:
                return None
            d, s = np
            if (
                d.strip().lower() == "eax"
                and s.strip().lower() == "eax"
            ):
                return f"cmp     dword {src_mem}, 0"
        return None

    def _pass_rmw_collapse(self, lines: list[Line]) -> list[Line]:
        """Collapse ``mov reg, [mem]; OP reg, SRC; mov [mem], reg``
        into ``OP dword [mem], SRC`` when reg is dead after the store.

        OP ∈ {add, sub, and, or, xor}. SRC is a numeric immediate or
        a non-EAX general-purpose register. x86 supports both
        ``OP r/m32, imm`` and ``OP r/m32, r32`` forms.

        Saves 5 bytes per match (3-byte load + 3-byte op + 3-byte
        store → single 3-4-byte memory-RMW instruction).

        Common in compound assignments to memory locations:
        ``int x; x += 5;`` (immediate form) and ``x += y;`` (register
        form, where y has been hoisted into a register).

        Conditions:
        - Line A: ``mov eax, [mem]`` (plain mov; mem must be a memory
          deref).
        - Line B: ``OP eax, SRC`` where OP is add/sub/and/or/xor and
          SRC is a numeric literal OR a non-EAX 32-bit register.
        - Line C: ``mov [mem], eax`` (same mem as line A).
        - EAX is dead after line C (CFG-aware scan).
        """
        out: list[Line] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if (
                i + 2 < len(lines)
                and line.kind == "instr"
                and line.op == "mov"
            ):
                ap = _operands_split(line.operands)
                if ap is not None:
                    adest, asrc = ap
                    adest_low = adest.strip().lower()
                    asrc_norm = asrc.strip()
                    if (
                        adest_low == "eax"
                        and asrc_norm.startswith("[")
                        and asrc_norm.endswith("]")
                    ):
                        b = lines[i + 1]
                        if (
                            b.kind == "instr"
                            and b.op in {"add", "sub", "and", "or", "xor"}
                        ):
                            bp = _operands_split(b.operands)
                            if bp is not None:
                                bdest, bsrc = bp
                                bsrc_norm = bsrc.strip()
                                src_for_rewrite = self._rmw_source_text(
                                    bsrc_norm
                                )
                                if (
                                    bdest.strip().lower() == "eax"
                                    and src_for_rewrite is not None
                                ):
                                    c = lines[i + 2]
                                    if (
                                        c.kind == "instr"
                                        and c.op == "mov"
                                    ):
                                        cp = _operands_split(c.operands)
                                        if cp is not None:
                                            cdest, csrc = cp
                                            if (
                                                cdest.strip()
                                                == asrc_norm
                                                and csrc.strip().lower()
                                                == "eax"
                                                and self._reg_dead_after(
                                                    lines, i + 3, "eax"
                                                )
                                            ):
                                                indent = (
                                                    self._extract_indent(
                                                        line.raw
                                                    )
                                                )
                                                spacer = " " * (
                                                    8 - len(b.op)
                                                )
                                                new_raw = (
                                                    f"{indent}{b.op}"
                                                    f"{spacer}dword "
                                                    f"{asrc_norm}, "
                                                    f"{src_for_rewrite}"
                                                )
                                                new_line = Line(
                                                    raw=new_raw,
                                                    kind="instr",
                                                    op=b.op,
                                                    operands=(
                                                        f"dword "
                                                        f"{asrc_norm}, "
                                                        f"{src_for_rewrite}"
                                                    ),
                                                )
                                                out.append(new_line)
                                                self.stats[
                                                    "rmw_collapse"
                                                ] = (
                                                    self.stats.get(
                                                        "rmw_collapse", 0
                                                    ) + 1
                                                )
                                                i += 3
                                                continue
            out.append(line)
            i += 1
        return out

    @staticmethod
    def _rmw_source_text(src: str) -> str | None:
        """For an `OP eax, src` where the rewrite goes to `OP [mem], src`,
        decide if `src` is acceptable.

        Acceptable:
        - numeric immediate (NASM accepts `OP r/m32, imm8` and imm32)
        - non-EAX 32-bit GP register

        Not acceptable: memory deref (would be mem-mem), EAX itself
        (the EAX value is stale after we drop the load), labels.
        """
        s = src.strip()
        # Numeric immediate.
        imm = PeepholeOptimizer._parse_numeric_immediate(s)
        if imm is not None:
            return str(imm)
        # Non-EAX register.
        sl = s.lower()
        if sl in {"ebx", "ecx", "edx", "esi", "edi"}:
            return sl
        return None

    def _pass_fst_fstp_collapse(self, lines: list[Line]) -> list[Line]:
        """Collapse ``fst <addr>; fstp st0`` into a single
        ``fstp <addr>``. Saves 2 bytes per match.

        ``fst <addr>`` stores st0 to memory without popping; ``fstp
        st0`` then pops the FPU stack (writing st0 to st0 first,
        which is a no-op). The combined effect is equivalent to a
        single ``fstp <addr>`` (store-and-pop). Same final FPU
        stack state, same memory write, fewer bytes.

        uc386 lowers ``*p = float_expr`` and similar through this
        pattern — see ``_eval_float_to_st0`` callers that store via
        ``fst`` followed by an explicit ``fstp st0`` cleanup.
        """
        out: list[Line] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if (
                i + 1 < len(lines)
                and line.kind == "instr"
                and line.op == "fst"
            ):
                nxt = lines[i + 1]
                if (
                    nxt.kind == "instr"
                    and nxt.op == "fstp"
                    and nxt.operands.strip().lower()
                    in ("st0", "st(0)")
                ):
                    indent = self._extract_indent(line.raw)
                    new_raw = f"{indent}fstp    {line.operands}"
                    new_line = Line(
                        raw=new_raw,
                        kind="instr",
                        op="fstp",
                        operands=line.operands,
                    )
                    out.append(new_line)
                    self.stats["fst_fstp_collapse"] = (
                        self.stats.get("fst_fstp_collapse", 0) + 1
                    )
                    i += 2
                    continue
            out.append(line)
            i += 1
        return out

    # Mapping: pop-form FPU op → memory-form (same direction).
    _FPU_POP_TO_MEM: dict[str, str] = {
        "faddp": "fadd",
        "fmulp": "fmul",
        "fsubp": "fsub",
        "fdivp": "fdiv",
        "fsubrp": "fsubr",
        "fdivrp": "fdivr",
    }

    def _pass_fpu_op_collapse(self, lines: list[Line]) -> list[Line]:
        """Collapse ``fld <addr>; faddp st1, st0`` (and friends) into
        a single ``fadd <addr>``. Same FPU stack/memory state, 2
        bytes saved per match.

        Maps:
          fld <addr>; faddp [st1, st0]  → fadd <addr>
          fld <addr>; fmulp [st1, st0]  → fmul <addr>
          fld <addr>; fsubp [st1, st0]  → fsub <addr>
          fld <addr>; fdivp [st1, st0]  → fdiv <addr>
          fld <addr>; fsubrp [st1, st0] → fsubr <addr>
          fld <addr>; fdivrp [st1, st0] → fdivr <addr>

        The pop-form ``faddp st1, st0`` computes ``st1 = st1 + st0;
        pop`` — the new top is ``old_st1 + addr_value``. The memory
        form ``fadd <addr>`` computes ``st0 = st0 + <addr_value>``.
        With the surrounding context (st0 was loaded from <addr>,
        st1 was the previous top), both produce the same final
        ``old_st0 + <addr_value>`` on the new top.

        The pop-form variants with explicit ``st1, st0`` operands
        are accepted; bare ``faddp`` (which defaults to st1, st0)
        is also accepted for completeness, though our codegen
        currently always emits the explicit form.
        """
        out: list[Line] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if (
                i + 1 < len(lines)
                and line.kind == "instr"
                and line.op == "fld"
            ):
                nxt = lines[i + 1]
                mem_form = self._FPU_POP_TO_MEM.get(nxt.op or "")
                if mem_form is not None and self._is_pop_st1_st0(nxt):
                    indent = self._extract_indent(line.raw)
                    spacer = " " * (8 - len(mem_form))
                    new_raw = (
                        f"{indent}{mem_form}{spacer}{line.operands}"
                    )
                    new_line = Line(
                        raw=new_raw,
                        kind="instr",
                        op=mem_form,
                        operands=line.operands,
                    )
                    out.append(new_line)
                    self.stats["fpu_op_collapse"] = (
                        self.stats.get("fpu_op_collapse", 0) + 1
                    )
                    i += 2
                    continue
            out.append(line)
            i += 1
        return out

    @staticmethod
    def _is_pop_st1_st0(line: Line) -> bool:
        """Match the ``st1, st0`` operand form (or the bare form,
        which defaults to ``st1, st0``)."""
        ops = line.operands.strip().lower()
        if ops == "":
            return True
        no_space = ops.replace(" ", "")
        return no_space in {"st1,st0", "st(1),st(0)"}

    def _pass_add_one_to_inc(self, lines: list[Line]) -> list[Line]:
        """``add reg, 1`` → ``inc reg`` (and ``sub reg, 1`` →
        ``dec reg``). Saves 2 bytes per register match (3-byte add-imm8
        → 1-byte inc-reg).

        Also handles ``add <size> [mem], 1`` → ``inc <size> [mem]``
        for byte/word/dword sizes — saves 1 byte per match in 32-bit
        mode (e.g., ``add dword [ebp + 8], 1`` is 4 bytes,
        ``inc dword [ebp + 8]`` is 3 bytes; same 1-byte delta for byte
        and word forms).

        ``inc``/``dec`` leave CF unchanged while ``add``/``sub`` set
        CF. The rewrite is safe only when CF is dead after — no
        ``jc/jnc/ja/jb/jae/jbe/setc/setnc/setb/setbe/etc.`` reads CF
        before flags are overwritten by another flag-clobberer or the
        function exits.

        Conservative: uses `_flags_safe_after` which bails on any
        flag-reader (not just CF readers). Most uc386 codegen patterns
        do `add reg, 1` followed by `test`/`cmp`/another arithmetic
        op (which clobbers flags), so the conservative check still
        fires often.

        Applies to all 8 32-bit GP registers and to memory operands
        with explicit `byte`/`word`/`dword` size.
        """
        out: list[Line] = []
        for i, line in enumerate(lines):
            if (
                line.kind == "instr"
                and line.op in ("add", "sub")
            ):
                parts = _operands_split(line.operands)
                if parts is not None:
                    dest, src = parts
                    dest_stripped = dest.strip()
                    dest_low = dest_stripped.lower()
                    is_reg = self._is_general_register(dest_low)
                    is_sized_mem = self._is_sized_memory_operand(
                        dest_stripped
                    )
                    if (
                        (is_reg or is_sized_mem)
                        and src.strip() == "1"
                        and self._flags_safe_after(lines, i + 1)
                    ):
                        new_op = "inc" if line.op == "add" else "dec"
                        indent = self._extract_indent(line.raw)
                        spacer = " " * (8 - len(new_op))
                        # Preserve original spacing/casing for memory
                        # operands; canonicalize register name.
                        new_dest = (
                            dest_low if is_reg else dest_stripped
                        )
                        new_raw = f"{indent}{new_op}{spacer}{new_dest}"
                        new_line = Line(
                            raw=new_raw,
                            kind="instr",
                            op=new_op,
                            operands=new_dest,
                        )
                        out.append(new_line)
                        self.stats["add_one_to_inc"] = (
                            self.stats.get("add_one_to_inc", 0) + 1
                        )
                        continue
            out.append(line)
        return out

    @staticmethod
    def _is_sized_memory_operand(text: str) -> bool:
        """Return True if `text` is a NASM sized memory operand like
        ``dword [ebp + 8]`` / ``byte [eax]`` / ``word [_glob]``.

        Matches the size keyword (byte/word/dword/qword) followed by
        whitespace and a bracketed memory expression. Not strict on
        the bracket contents (any `[...]`).
        """
        m = re.match(
            r"^(byte|word|dword|qword)\s+\[.*\]\s*$",
            text,
            re.IGNORECASE,
        )
        return m is not None

    # Instructions whose ZF/SF are set based on the result AND which
    # ALSO clear OF and CF — the same flag state as `test reg, reg`.
    # After any of these, a subsequent `test reg, reg` (with the same
    # reg as the dest) is fully redundant: ALL flag bits match what
    # the prior op set, so any subsequent Jcc reads the same state.
    #
    # Restricted to logical ops only. add/sub/inc/dec/neg/shl/shr/sar
    # set OF and CF based on overflow / shifted-out bits — those
    # diverge from test's "always 0" — so dropping the test changes
    # the flag state seen by jg/jl/ja/jb/jo/etc. A specific failure:
    # `sub eax, ecx; test eax, eax; jg L` with overflow gives SF != OF,
    # so jg is not-taken; dropping the test gives SF maybe == OF
    # depending on overflow direction. Caught by torture's signed-
    # comparison-after-subtraction tests (20000403-1, bf-sign-2).
    _ZF_SF_SETTING_RMW_OPS: frozenset[str] = frozenset({
        "and", "or", "xor",
    })

    def _pass_redundant_test_collapse(self, lines: list[Line]) -> list[Line]:
        """Drop ``test reg, reg`` when the immediately-preceding
        instruction was a flag-setting arithmetic on the same reg.
        The prior op already set ZF/SF based on its result, so a
        ``test reg, reg`` is redundant — same flag state.

        Saves 2 bytes per match. Common in:

            and  eax, MASK   ; sets flags
            test eax, eax    ; redundant
            jz   target

        becomes:

            and  eax, MASK
            jz   target
        """
        out: list[Line] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if (
                line.kind == "instr"
                and line.op == "test"
            ):
                parts = _operands_split(line.operands)
                if parts is not None:
                    a, b = parts
                    al = a.strip().lower()
                    bl = b.strip().lower()
                    # Self-test: `test reg, reg` (same reg both sides).
                    if al == bl and self._is_general_register(al):
                        # Look back for the preceding instruction that
                        # set this reg as its destination.
                        prev_idx = self._find_prev_instr(out)
                        if prev_idx is not None:
                            prev = out[prev_idx]
                            if prev.op in self._ZF_SF_SETTING_RMW_OPS:
                                # All current ops in the safe set are
                                # 2-operand (and/or/xor). The
                                # destination is the first operand.
                                pp = _operands_split(prev.operands)
                                if (
                                    pp is not None
                                    and pp[0].strip().lower() == al
                                ):
                                    self.stats[
                                        "redundant_test_collapse"
                                    ] = (
                                        self.stats.get(
                                            "redundant_test_collapse", 0
                                        ) + 1
                                    )
                                    i += 1
                                    continue
            out.append(line)
            i += 1
        return out

    @staticmethod
    def _find_prev_instr(out: list[Line]) -> int | None:
        """Return the index of the last `instr` line in `out`, or
        None if there isn't one (or there's a label/directive in
        between, which is a basic-block boundary)."""
        for i in range(len(out) - 1, -1, -1):
            ln = out[i]
            if ln.kind == "instr":
                return i
            if ln.kind in ("label", "directive", "data"):
                return None  # block boundary
        return None

    # Match `byte [...]` or `word [...]` (with optional whitespace).
    _NARROWING_LOAD_SRC_RE = re.compile(
        r"^(byte|word)\s+(\[.*\])\s*$",
        re.IGNORECASE,
    )

    # Jcc's that read ONLY ZF — they're flag-state-invariant under
    # the byte-vs-dword cmp difference. Any other Jcc (jl/jg/ja/jb/
    # js/jno/etc.) reads SF, OF, or CF, whose values DO differ between
    # `test eax, eax` (after movzx of byte 0xFF: SF=0, since EAX bit
    # 31 = 0) and `cmp byte [...], 0` (8-bit subtraction of 0 from
    # 0xFF: SF=1). So the rewrite is unsafe for non-ZF Jcc.
    _ZF_ONLY_JCC: frozenset[str] = frozenset({
        "jz", "jnz", "je", "jne",
    })

    def _pass_narrowing_load_test_collapse(
        self, lines: list[Line]
    ) -> list[Line]:
        """Collapse ``movsx/movzx eax, byte/word [SRC]; test eax, eax;
        j[n]z LBL`` into ``cmp byte/word [SRC], 0; j[n]z LBL``.

        The narrowing load fills EAX with a sign- or zero-extended
        byte/word from memory. ``test eax, eax`` then checks the result
        for zero. A sign- or zero-extended byte/word is zero iff the
        source byte/word is itself zero — so the same ZF state results
        from a direct ``cmp <size> [SRC], 0`` against the narrow memory
        operand.

        Restricted to ZF-only Jcc (jz/jnz/je/jne). For signed Jcc
        (jl/jg/jge/jle) or unsigned Jcc (jb/ja/jae/jbe), the byte
        comparison sets SF/OF/CF differently than the dword comparison
        does on a movzx-extended value: e.g., for byte 0xFF, byte cmp
        sets SF=1 (8-bit MSB) while ``test eax, eax`` after movzx sets
        SF=0 (32-bit MSB of 0x000000FF). Caught a regression in
        torture's ``doloop-1`` where ``unsigned char z; --z > 0`` used
        ``movzx; cmp; setg; ...; jnz`` — the inner setg-chain collapses
        to ``cmp byte; jg``, which gives the wrong answer for z=255.
        Restricting to ZF-only Jcc avoids the trap.

        Saves 2 bytes per match.

        Conditions:
        - First line is ``movsx eax, byte/word [SRC]`` or
          ``movzx eax, byte/word [SRC]``.
        - Second line is ``test eax, eax``.
        - Third line is a ZF-only Jcc (jz/jnz/je/jne).
        - EAX is dead after the test (CFG-aware scan via
          `_reg_dead_after`).
        """
        out: list[Line] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if (
                i + 2 < len(lines)
                and line.kind == "instr"
                and line.op in ("movsx", "movzx")
            ):
                parts = _operands_split(line.operands)
                if parts is not None:
                    dest, src = parts
                    if dest.strip().lower() == "eax":
                        m = self._NARROWING_LOAD_SRC_RE.match(
                            src.strip()
                        )
                        if m is not None:
                            size = m.group(1).lower()
                            mem = m.group(2)
                            nxt = lines[i + 1]
                            jcc_line = self._next_instr_after(
                                lines, i + 2
                            )
                            if (
                                nxt.kind == "instr"
                                and nxt.op == "test"
                                and jcc_line is not None
                                and jcc_line.op in self._ZF_ONLY_JCC
                            ):
                                np = _operands_split(nxt.operands)
                                if (
                                    np is not None
                                    and np[0].strip().lower() == "eax"
                                    and np[1].strip().lower() == "eax"
                                    and self._reg_dead_after(
                                        lines, i + 2, "eax"
                                    )
                                ):
                                    indent = self._extract_indent(
                                        line.raw
                                    )
                                    new_raw = (
                                        f"{indent}cmp     "
                                        f"{size} {mem}, 0"
                                    )
                                    new_line = Line(
                                        raw=new_raw,
                                        kind="instr",
                                        op="cmp",
                                        operands=(
                                            f"{size} {mem}, 0"
                                        ),
                                    )
                                    out.append(new_line)
                                    self.stats[
                                        "narrowing_load_test_collapse"
                                    ] = (
                                        self.stats.get(
                                            "narrowing_load_test_collapse",
                                            0,
                                        ) + 1
                                    )
                                    i += 2
                                    continue
            out.append(line)
            i += 1
        return out

    @staticmethod
    def _next_instr_after(
        lines: list[Line], start: int
    ) -> "Line | None":
        """Return the next ``instr`` line at or after `start`, skipping
        blanks/comments. Returns None at labels/directives/data (basic-
        block boundaries) or end-of-list."""
        for j in range(start, len(lines)):
            ln = lines[j]
            if ln.kind == "instr":
                return ln
            if ln.kind in ("label", "directive", "data"):
                return None
        return None

    def _pass_jcc_jmp_inversion(
        self, lines: list[Line]
    ) -> list[Line]:
        """Collapse ``jcc L1; jmp L2; L1:`` into ``j!cc L2; L1:`` —
        invert the condition and drop the unconditional jmp. The
        ``L1:`` label can stay (harmless) or be dropped if no other
        code references it; we leave it in place for simplicity.

        Saves 5 bytes per match (the dropped `jmp imm32` is 5 bytes;
        a short `jmp imm8` is 2 bytes — either way, dropping the jmp
        is a strict win).

        Common in if-else / ternary lowering where one arm collapses
        to a fall-through after `redundant_eax_load`:
            jle .L1_false
            jmp .L_end          ; branch when fallthrough = "true case"
            .L1_false:
            mov eax, [b]        ; "false case"
            .L_end:
        becomes:
            jg .L_end             ; invert condition
            .L1_false:           ; harmless leftover
            mov eax, [b]
            .L_end:
        """
        out: list[Line] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            # Match jcc.
            if (
                line.kind == "instr"
                and line.op.startswith("j")
                and line.op != "jmp"
                and line.op[1:] in self._CC_INVERSE
            ):
                cc = line.op[1:]
                jcc_target = line.operands.strip()
                # Look for the next instr line (skip blanks/comments).
                j_idx = i + 1
                while (
                    j_idx < len(lines)
                    and lines[j_idx].kind in ("blank", "comment")
                ):
                    j_idx += 1
                if (
                    j_idx < len(lines)
                    and lines[j_idx].kind == "instr"
                    and lines[j_idx].op == "jmp"
                ):
                    jmp_line = lines[j_idx]
                    jmp_target = jmp_line.operands.strip()
                    # Look for the label after the jmp.
                    k_idx = j_idx + 1
                    while (
                        k_idx < len(lines)
                        and lines[k_idx].kind in ("blank", "comment")
                    ):
                        k_idx += 1
                    if (
                        k_idx < len(lines)
                        and lines[k_idx].kind == "label"
                        and lines[k_idx].label == jcc_target
                    ):
                        # Match! Rewrite jcc with inverted CC and
                        # the jmp's target.
                        inv_cc = self._CC_INVERSE[cc]
                        indent = self._extract_indent(line.raw)
                        new_op = f"j{inv_cc}"
                        spacer = " " * max(8 - len(new_op), 1)
                        new_raw = (
                            f"{indent}{new_op}{spacer}{jmp_target}"
                        )
                        new_line = Line(
                            raw=new_raw,
                            kind="instr",
                            op=new_op,
                            operands=jmp_target,
                        )
                        out.append(new_line)
                        # Drop the jmp (between i and j_idx, also
                        # drop any blanks/comments between).
                        # Keep the label and continue from there.
                        self.stats["jcc_jmp_inversion"] = (
                            self.stats.get("jcc_jmp_inversion", 0) + 1
                        )
                        i = j_idx + 1
                        continue
            out.append(line)
            i += 1
        return out

    # Map NASM size keyword to the EAX-family sub-register that
    # carries 0 when EAX = 0.
    _ZERO_INIT_SIZE_TO_REG: dict[str, str] = {
        "byte": "al",
        "word": "ax",
        "dword": "eax",
    }

    def _pass_zero_init_collapse(
        self, lines: list[Line]
    ) -> list[Line]:
        """Replace 1+ adjacent ``mov <size> [m], 0`` stores with
        ``xor eax, eax`` followed by per-store ``mov [m], <reg>``.

        Each ``mov dword [m], 0`` is 7 bytes; the rewrite produces
        ``mov [m], eax`` at 3 bytes plus one upfront ``xor eax, eax``
        at 2 bytes. Net savings:
        - 1 store: 7 → 5 (xor + mov), save 2 bytes.
        - N stores: 7N → 2 + 3N, save 4N - 2 bytes.

        Byte stores: 4 → 3 (1 byte saved + 2 bytes xor amortized).
        Word stores: 5 (66-prefixed) → 4, save 1 byte.

        Conditions:
        - 1+ adjacent ``mov <size> [m], 0`` instructions (size in
          {byte, word, dword}; mixed sizes within a chain are OK).
        - EAX is dead after the chain (we'll write 0 to it).
        - Flags are safe after the chain (xor sets ZF/SF/PF/CF/OF;
          subsequent flag-readers would see different state).

        Common in function prologues with multiple zero-initialized
        locals (e.g. ``int a = 0, b = 0, c = 0;``).
        """
        out: list[Line] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if not (
                line.kind == "instr"
                and self._is_zero_imm_store(line)
            ):
                out.append(line)
                i += 1
                continue
            # Found a zero-imm-store. Collect adjacent ones.
            chain_idx = [i]
            j = i + 1
            while j < len(lines):
                nxt = lines[j]
                if nxt.kind in ("blank", "comment"):
                    j += 1
                    continue
                if (
                    nxt.kind == "instr"
                    and self._is_zero_imm_store(nxt)
                ):
                    chain_idx.append(j)
                    j += 1
                    continue
                break
            # Validity: EAX dead after, flags safe after.
            after_idx = j
            if not self._reg_dead_after(lines, after_idx, "eax"):
                # Conservative: skip the whole chain (don't fire).
                for k in chain_idx:
                    out.append(lines[k])
                i = j
                continue
            if not self._flags_safe_after(lines, after_idx):
                for k in chain_idx:
                    out.append(lines[k])
                i = j
                continue
            # Emit xor + per-store rewrites.
            indent = self._extract_indent(line.raw)
            out.append(Line(
                raw=f"{indent}xor     eax, eax",
                kind="instr",
                op="xor",
                operands="eax, eax",
            ))
            for k in chain_idx:
                ld = lines[k]
                parts = _operands_split(ld.operands)
                if parts is None:
                    # Shouldn't happen if _is_zero_imm_store
                    # accepted it; bail conservatively.
                    out.append(ld)
                    continue
                dest, _ = parts
                # Strip the size keyword and pick the matching reg.
                size_kw, mem = self._split_sized_mem(dest.strip())
                reg = self._ZERO_INIT_SIZE_TO_REG.get(
                    size_kw.lower(), "eax"
                )
                out.append(Line(
                    raw=f"{indent}mov     {mem}, {reg}",
                    kind="instr",
                    op="mov",
                    operands=f"{mem}, {reg}",
                ))
            self.stats["zero_init_collapse"] = (
                self.stats.get("zero_init_collapse", 0)
                + len(chain_idx)
            )
            i = j
        return out

    @staticmethod
    def _is_zero_imm_store(line: Line) -> bool:
        """Return True if this is ``mov <byte|word|dword> [m], 0``.
        Only sized memory destinations are recognized (NASM requires
        the size keyword for memory + immediate stores)."""
        if line.kind != "instr" or line.op != "mov":
            return False
        parts = _operands_split(line.operands)
        if parts is None:
            return False
        dest, src = parts
        if src.strip() != "0":
            return False
        m = re.match(
            r"^(byte|word|dword)\s+\[.*\]\s*$",
            dest.strip(),
            re.IGNORECASE,
        )
        return m is not None

    @staticmethod
    def _split_sized_mem(text: str) -> tuple[str, str]:
        """Split ``"<size> [m]"`` into ``("<size>", "[m]")``."""
        m = re.match(
            r"^(byte|word|dword|qword)\s+(\[.*\])\s*$",
            text,
            re.IGNORECASE,
        )
        if m is None:
            return ("", text)
        return (m.group(1), m.group(2))

    # Map sub-register name → enclosing 32-bit register. Writing to
    # any sub-register changes the 32-bit reg's value, so we must
    # invalidate the "is zero" state for the enclosing reg.
    _SUBREG_TO_GP32: dict[str, str] = {
        "ax": "eax", "al": "eax", "ah": "eax",
        "bx": "ebx", "bl": "ebx", "bh": "ebx",
        "cx": "ecx", "cl": "ecx", "ch": "ecx",
        "dx": "edx", "dl": "edx", "dh": "edx",
        "si": "esi",
        "di": "edi",
        "bp": "ebp",
        "sp": "esp",
    }

    def _pass_disp_load_collapse(
        self, lines: list[Line]
    ) -> list[Line]:
        """Collapse ``add REG, DISP; mov DST, [REG]`` into
        ``mov DST, [REG + DISP]`` using x86's disp addressing form.

        Saves bytes:
        - DISP fits in imm8 (-128..127): save 2 bytes (add 3 bytes
          + mov 2 bytes → mov 3 bytes).
        - DISP needs imm32: save 1 byte (add 5/6 bytes + mov 2 bytes
          → mov 6 bytes).

        Common in `p->member` struct member access where the
        codegen emits an explicit `add reg, offset` before the
        dereference, even for small offsets.

        Conditions:
        - Two consecutive instr lines.
        - Line A: ``add REG, DISP`` (DISP is a numeric literal).
        - Line B: ``mov DST, [REG]`` (plain deref of REG, no
          existing offset/SIB).
        - REG either == DST (overwritten by the load) or dead after
          line B.
        """
        out = list(lines)
        i = 0
        while i + 1 < len(out):
            a = out[i]
            b = out[i + 1]
            if not (
                a.kind == "instr" and a.op == "add"
                and b.kind == "instr" and b.op == "mov"
            ):
                i += 1
                continue
            ap = _operands_split(a.operands)
            bp = _operands_split(b.operands)
            if ap is None or bp is None:
                i += 1
                continue
            reg = ap[0].strip().lower()
            try:
                disp = int(ap[1].strip())
            except ValueError:
                i += 1
                continue
            if not self._is_general_register(reg):
                i += 1
                continue
            dst_reg = bp[0].strip().lower()
            src = bp[1].strip()
            if not self._is_general_register(dst_reg):
                i += 1
                continue
            # Source must be `[REG]` (no offset, no SIB).
            src_stripped = src
            for prefix in ("dword ", "word ", "byte ", "qword "):
                if src_stripped.lower().startswith(prefix):
                    src_stripped = src_stripped[
                        len(prefix):
                    ].lstrip()
                    break
            mem_re = re.match(
                r"^\[([a-zA-Z]+)\s*\]$", src_stripped
            )
            if mem_re is None:
                i += 1
                continue
            mem_reg = mem_re.group(1).lower()
            if mem_reg != reg:
                i += 1
                continue
            # Liveness: REG dead after line B (we drop the add).
            # If REG == DST, the mov overwrites REG anyway — safe.
            if reg != dst_reg and not self._reg_dead_after(
                out, i + 2, reg
            ):
                i += 1
                continue
            # Build the rewritten mov.
            indent = self._extract_indent(b.raw)
            sign = "+" if disp >= 0 else "-"
            new_src = f"[{reg} {sign} {abs(disp)}]"
            new_raw = f"{indent}mov     {dst_reg}, {new_src}"
            new_line = Line(
                raw=new_raw,
                kind="instr",
                op="mov",
                operands=f"{dst_reg}, {new_src}",
            )
            new_out = out[:i] + [new_line] + out[i + 2:]
            out = new_out
            self.stats["disp_load_collapse"] = (
                self.stats.get(
                    "disp_load_collapse", 0
                ) + 1
            )
            continue
        return out

    def _pass_index_load_collapse(
        self, lines: list[Line]
    ) -> list[Line]:
        """Collapse ``shl IDX, N; add BASE, IDX; mov DST, [BASE]``
        (or ``mov DST, [BASE + DISP]``) into a SIB-form load.
        ``[BASE]`` → ``[BASE + IDX*SCALE]``;
        ``[BASE + DISP]`` → ``[BASE + IDX*SCALE + DISP]``.

        x86 supports the SIB byte form ``[base + index*scale]``
        (with optional 8/32-bit displacement) directly in mov memory
        operands, eliminating the need for explicit shl + add to
        compute the address. Saves 4 bytes per match (shl + add).

        Conditions:
        - Three consecutive instr lines (no labels/comments between).
        - Line A: ``shl IDX, N`` where N ∈ {1, 2, 3} (scale 2/4/8;
          scale 1 = no-op shl, irrelevant).
        - Line B: ``add BASE, IDX`` (different reg).
        - Line C: ``mov DST, [BASE]`` or ``mov DST, [BASE +/- DISP]``.
        - IDX must be dead after Line C (we drop the shl, so IDX
          retains its pre-shl value; subsequent code expecting the
          shifted value would be wrong).
        - BASE must be either the same reg as DST (overwritten by
          the load) OR dead after Line C.

        Restricted to GP 32-bit registers.
        """
        SCALE = {1: 2, 2: 4, 3: 8}
        # mov, movsx, movzx — all support SIB-form memory operand.
        LOAD_OPS = {"mov", "movsx", "movzx"}
        out = list(lines)
        i = 0
        while i + 2 < len(out):
            a = out[i]
            b = out[i + 1]
            c = out[i + 2]
            if not (
                a.kind == "instr" and a.op == "shl"
                and b.kind == "instr" and b.op == "add"
                and c.kind == "instr" and c.op in LOAD_OPS
            ):
                i += 1
                continue
            # Parse line A.
            ap = _operands_split(a.operands)
            if ap is None:
                i += 1
                continue
            idx_reg = ap[0].strip().lower()
            try:
                n = int(ap[1].strip())
            except ValueError:
                i += 1
                continue
            if n not in SCALE:
                i += 1
                continue
            if not self._is_general_register(idx_reg):
                i += 1
                continue
            # Parse line B.
            bp = _operands_split(b.operands)
            if bp is None:
                i += 1
                continue
            base_reg = bp[0].strip().lower()
            b_src = bp[1].strip().lower()
            if (
                not self._is_general_register(base_reg)
                or b_src != idx_reg
                or base_reg == idx_reg
            ):
                i += 1
                continue
            # Parse line C.
            cp = _operands_split(c.operands)
            if cp is None:
                i += 1
                continue
            dst_reg = cp[0].strip().lower()
            c_src = cp[1].strip()
            if not self._is_general_register(dst_reg):
                i += 1
                continue
            # Source must be `[BASE]` or `[BASE +/- DISP]`.
            c_src_stripped = c_src
            size_prefix = ""
            for prefix in ("dword ", "word ", "byte ", "qword "):
                if c_src_stripped.lower().startswith(prefix):
                    size_prefix = c_src_stripped[:len(prefix)]
                    c_src_stripped = c_src_stripped[
                        len(prefix):
                    ].lstrip()
                    break
            mem_re = re.match(
                r"^\[\s*([a-zA-Z]+)"
                r"(?:\s*([+-])\s*(\d+))?"
                r"\s*\]$",
                c_src_stripped,
            )
            if mem_re is None:
                i += 1
                continue
            mem_reg = mem_re.group(1).lower()
            if mem_reg != base_reg:
                i += 1
                continue
            disp_sign = mem_re.group(2)
            disp_val = mem_re.group(3)
            # Liveness: IDX dead after line C; BASE either ==
            # DST (overwritten) or dead after.
            if not self._reg_dead_after(out, i + 3, idx_reg):
                i += 1
                continue
            if base_reg != dst_reg and not self._reg_dead_after(
                out, i + 3, base_reg
            ):
                i += 1
                continue
            # Rewrite: drop A and B; replace C with SIB-form load.
            # Preserve the original op (mov / movsx / movzx).
            indent = self._extract_indent(c.raw)
            scale = SCALE[n]
            if disp_sign is None:
                new_src = (
                    f"{size_prefix}[{base_reg} + {idx_reg}*{scale}]"
                )
            else:
                new_src = (
                    f"{size_prefix}[{base_reg} + {idx_reg}*{scale} "
                    f"{disp_sign} {disp_val}]"
                )
            opname = c.op
            spacer = " " * max(8 - len(opname), 1)
            new_raw = (
                f"{indent}{opname}{spacer}{dst_reg}, {new_src}"
            )
            new_line = Line(
                raw=new_raw,
                kind="instr",
                op=opname,
                operands=f"{dst_reg}, {new_src}",
            )
            new_out = out[:i] + [new_line] + out[i + 3:]
            out = new_out
            self.stats["index_load_collapse"] = (
                self.stats.get(
                    "index_load_collapse", 0
                ) + 1
            )
            # Restart from the rewrite point.
            continue
        return out

    def _pass_push_disp_collapse(
        self, lines: list[Line]
    ) -> list[Line]:
        """Collapse ``add REG, DISP; push dword [REG]`` into
        ``push dword [REG + DISP]`` using x86's disp addressing form.

        Sister of ``disp_load_collapse`` for the push case. Saves
        2 bytes per match when DISP fits in imm8, 1 byte for imm32.
        Common in `push p->member` arg-setup paths.

        Conditions:
        - Two consecutive instr lines.
        - Line A: ``add REG, DISP`` (DISP is a numeric literal).
        - Line B: ``push dword [REG]`` (plain deref, no offset/SIB).
        - REG dead after Line B (push doesn't write REG).
        """
        out = list(lines)
        i = 0
        while i + 1 < len(out):
            a = out[i]
            b = out[i + 1]
            if not (
                a.kind == "instr" and a.op == "add"
                and b.kind == "instr" and b.op == "push"
            ):
                i += 1
                continue
            ap = _operands_split(a.operands)
            if ap is None:
                i += 1
                continue
            reg = ap[0].strip().lower()
            try:
                disp = int(ap[1].strip())
            except ValueError:
                i += 1
                continue
            if not self._is_general_register(reg):
                i += 1
                continue
            # Push source: `[REG]` only (with optional size prefix).
            b_src = b.operands.strip()
            for prefix in ("dword ", "word ", "byte ", "qword "):
                if b_src.lower().startswith(prefix):
                    b_src = b_src[len(prefix):].lstrip()
                    break
            mem_re = re.match(r"^\[([a-zA-Z]+)\s*\]$", b_src)
            if mem_re is None:
                i += 1
                continue
            if mem_re.group(1).lower() != reg:
                i += 1
                continue
            # REG dead after the push (push reads but doesn't write).
            if not self._reg_dead_after(out, i + 2, reg):
                i += 1
                continue
            indent = self._extract_indent(b.raw)
            sign = "+" if disp >= 0 else "-"
            new_src = f"dword [{reg} {sign} {abs(disp)}]"
            new_raw = f"{indent}push    {new_src}"
            new_line = Line(
                raw=new_raw,
                kind="instr",
                op="push",
                operands=new_src,
            )
            out = out[:i] + [new_line] + out[i + 2:]
            self.stats["push_disp_collapse"] = (
                self.stats.get("push_disp_collapse", 0) + 1
            )
            continue
        return out

    def _pass_push_index_collapse(
        self, lines: list[Line]
    ) -> list[Line]:
        """Collapse ``shl IDX, N; add BASE, IDX; push dword [BASE]``
        into ``push dword [BASE + IDX*SCALE]`` (SCALE = 2^N).

        Sister of ``index_load_collapse`` for the push case. Saves
        4 bytes per match. Common in cdecl arg-push of array elements
        like ``f(arr[i])``.

        Conditions:
        - Three consecutive instr lines.
        - Line A: ``shl IDX, N`` where N ∈ {1, 2, 3}.
        - Line B: ``add BASE, IDX`` (different reg).
        - Line C: ``push dword [BASE]`` (plain deref of BASE).
        - Both IDX and BASE dead after Line C (push doesn't write
          either; we drop the shl and add).
        """
        SCALE = {1: 2, 2: 4, 3: 8}
        out = list(lines)
        i = 0
        while i + 2 < len(out):
            a = out[i]
            b = out[i + 1]
            c = out[i + 2]
            if not (
                a.kind == "instr" and a.op == "shl"
                and b.kind == "instr" and b.op == "add"
                and c.kind == "instr" and c.op == "push"
            ):
                i += 1
                continue
            ap = _operands_split(a.operands)
            bp = _operands_split(b.operands)
            if ap is None or bp is None:
                i += 1
                continue
            idx_reg = ap[0].strip().lower()
            try:
                n = int(ap[1].strip())
            except ValueError:
                i += 1
                continue
            if n not in SCALE:
                i += 1
                continue
            if not self._is_general_register(idx_reg):
                i += 1
                continue
            base_reg = bp[0].strip().lower()
            b_src = bp[1].strip().lower()
            if (
                not self._is_general_register(base_reg)
                or b_src != idx_reg
                or base_reg == idx_reg
            ):
                i += 1
                continue
            # Push source: `[BASE]` only.
            c_src = c.operands.strip()
            for prefix in ("dword ", "word ", "byte ", "qword "):
                if c_src.lower().startswith(prefix):
                    c_src = c_src[len(prefix):].lstrip()
                    break
            mem_re = re.match(r"^\[([a-zA-Z]+)\s*\]$", c_src)
            if mem_re is None:
                i += 1
                continue
            if mem_re.group(1).lower() != base_reg:
                i += 1
                continue
            # IDX and BASE both dead after the push.
            if not self._reg_dead_after(out, i + 3, idx_reg):
                i += 1
                continue
            if not self._reg_dead_after(out, i + 3, base_reg):
                i += 1
                continue
            indent = self._extract_indent(c.raw)
            scale = SCALE[n]
            new_src = f"dword [{base_reg} + {idx_reg}*{scale}]"
            new_raw = f"{indent}push    {new_src}"
            new_line = Line(
                raw=new_raw,
                kind="instr",
                op="push",
                operands=new_src,
            )
            out = out[:i] + [new_line] + out[i + 3:]
            self.stats["push_index_collapse"] = (
                self.stats.get("push_index_collapse", 0) + 1
            )
            continue
        return out

    # Compound-assign binops (used by _pass_compound_assign_collapse).
    # imul is excluded — `imul r/m32` (one-operand) writes EDX:EAX,
    # not the obvious in-place form.
    _COMPOUND_BINOPS: frozenset[str] = frozenset({
        "add", "sub", "and", "or", "xor",
    })

    def _pass_compound_assign_collapse(
        self, lines: list[Line]
    ) -> list[Line]:
        """Collapse the compound-assignment frame into an in-place
        memory-RMW.

        Match (looking from a final ``mov [m], eax``):
            push    dword [m]              ; L_push
            ... chain ...                   ; produces value in EAX
            mov     ecx, eax                ; L_pop - 3
            pop     eax                     ; L_pop - 2
            <OP>    eax, ecx                ; L_pop - 1
            mov     [m], eax                ; L_pop

        Replace with:
            ... chain ...                   ; unchanged
            <OP>    [m], eax                ; in-place RMW

        Drops the entire framing (push, mov ecx eax, pop, OP eax ecx,
        mov [m] eax) — saves 8 bytes per match.

        Conditions:
        - L_push must be ``push dword [m]`` with same `m` as the final
          store.
        - The chain (between L_push and L_pop - 3) must not modify
          [m] (no stores to that address) and must be stack-balanced
          (no other push/pop or sub/add esp).
        - ECX must be dead after L_pop (typically true since ECX is
          a scratch register in our codegen).

        Common in compound assignments to slot-typed lvalues where
        the rhs computation isn't reducible via simpler passes (e.g.,
        `s += p[i]` where p[i] requires multi-instruction array
        indexing).
        """
        out = list(lines)
        i = 0
        while i < len(out):
            line = out[i]
            if (
                line.kind == "instr"
                and line.op == "mov"
                and self._is_eax_to_mem_store(line)
            ):
                # The mov [m], eax is line i. Look back for the
                # frame pattern.
                store_dest = self._mem_dest(line)
                if store_dest is None:
                    i += 1
                    continue
                # Try the 4-line tail first (canonical form for
                # any OP). Then fall back to the 3-line short tail
                # (commutative-OP-only fast path the codegen emits).
                match = self._match_compound_assign_frame(
                    out, i, store_dest
                )
                tail_count = 4 if match is not None else None
                if match is None:
                    match = self._match_compound_assign_frame_short(
                        out, i, store_dest
                    )
                    if match is not None:
                        tail_count = 3
                if match is None:
                    i += 1
                    continue
                push_idx, op_str = match
                # Verify ECX dead after the store.
                if not self._reg_dead_after(out, i + 1, "ecx"):
                    i += 1
                    continue
                # Determine if EAX is dead after the store. If yes,
                # we can drop the framing entirely. If no, we drop
                # the framing AND insert ``mov eax, [m]`` to restore
                # EAX = new [m] value (the canonical tail leaves
                # EAX = lhs OP rhs = new [m]; the reload restores
                # this for downstream consumers).
                eax_dead = self._reg_dead_after(out, i + 1, "eax")
                # Rewrite: drop push, framing tail, mov [m] eax —
                # replace with ``OP [m], eax`` in place of the tail.
                indent = self._extract_indent(line.raw)
                spacer = " " * max(8 - len(op_str), 1)
                new_raw = (
                    f"{indent}{op_str}{spacer}{store_dest}, eax"
                )
                new_line = Line(
                    raw=new_raw,
                    kind="instr",
                    op=op_str,
                    operands=f"{store_dest}, eax",
                )
                # Chain runs from push_idx+1 to (i - tail_count)
                # inclusive — exclusive slice end = i - tail_count + 1.
                # For tail_count=4 (canonical) this is i - 3; for
                # tail_count=3 (commutative short form) it's i - 2.
                replace_lines = [new_line]
                # If EAX is live, append `mov eax, [m]` to reload.
                # Lets downstream code see EAX = new [m] value,
                # matching the canonical tail's post-state.
                if not eax_dead:
                    reload_raw = (
                        f"{indent}mov     eax, {store_dest}"
                    )
                    reload_line = Line(
                        raw=reload_raw,
                        kind="instr",
                        op="mov",
                        operands=f"eax, {store_dest}",
                    )
                    replace_lines.append(reload_line)
                new_out = []
                new_out.extend(out[:push_idx])
                new_out.extend(
                    out[push_idx + 1: i - tail_count + 1]
                )
                new_out.extend(replace_lines)
                new_out.extend(out[i + 1:])
                self.stats["compound_assign_collapse"] = (
                    self.stats.get(
                        "compound_assign_collapse", 0
                    ) + 1
                )
                out = new_out
                # Restart from the rewrite point (chain start).
                i = push_idx
                continue
            i += 1
        return out

    @staticmethod
    def _is_eax_to_mem_store(line: Line) -> bool:
        """Is this ``mov [...], eax``?"""
        if line.kind != "instr" or line.op != "mov":
            return False
        parts = _operands_split(line.operands)
        if parts is None:
            return False
        dest, src = parts
        if "[" not in dest:
            return False
        if src.strip().lower() != "eax":
            return False
        return True

    @staticmethod
    def _mem_dest(line: Line) -> str | None:
        """Extract the memory operand from ``mov [...], eax``."""
        parts = _operands_split(line.operands)
        if parts is None:
            return None
        dest, _ = parts
        return dest.strip()

    def _match_compound_assign_frame(
        self,
        lines: list[Line],
        store_idx: int,
        store_dest: str,
    ) -> tuple[int, str] | None:
        """Look back from `store_idx` for the compound-assign frame.

        Expects (going backward from store_idx):
            store_idx - 1: ``OP eax, ecx`` (OP in _COMPOUND_BINOPS).
            store_idx - 2: ``pop eax``.
            store_idx - 3: ``mov ecx, eax``.
            ...chain (no stack manipulation, no [m] modification)...
            push_idx: ``push dword [m]`` matching store_dest.

        Returns (push_idx, op_str) on match, None otherwise. The
        chain is identified as `push_idx+1 .. store_idx-4`.
        """
        # Walk back through the framing tail.
        ops_at = lambda j: (
            lines[j] if 0 <= j < len(lines) else None
        )
        # store_idx - 1: OP eax, ecx.
        op_line = ops_at(store_idx - 1)
        if op_line is None or op_line.kind != "instr":
            return None
        if op_line.op not in self._COMPOUND_BINOPS:
            return None
        op_parts = _operands_split(op_line.operands)
        if op_parts is None:
            return None
        if (
            op_parts[0].strip().lower() != "eax"
            or op_parts[1].strip().lower() != "ecx"
        ):
            return None
        op_str = op_line.op
        # store_idx - 2: pop eax.
        pop_line = ops_at(store_idx - 2)
        if (
            pop_line is None
            or pop_line.kind != "instr"
            or pop_line.op != "pop"
            or pop_line.operands.strip().lower() != "eax"
        ):
            return None
        # store_idx - 3: mov ecx, eax.
        xfer_line = ops_at(store_idx - 3)
        if (
            xfer_line is None
            or xfer_line.kind != "instr"
            or xfer_line.op != "mov"
        ):
            return None
        xfer_parts = _operands_split(xfer_line.operands)
        if xfer_parts is None:
            return None
        if (
            xfer_parts[0].strip().lower() != "ecx"
            or xfer_parts[1].strip().lower() != "eax"
        ):
            return None
        # Walk back from store_idx - 4 to find the matching push.
        # Track stack depth — balanced inner push/pop pairs are
        # OK (e.g. nested array-indexing pushes).
        push_target = f"dword {store_dest}"
        depth = 0
        for j in range(store_idx - 4, -1, -1):
            ln = lines[j]
            if ln.kind != "instr":
                if ln.kind == "label":
                    return None  # crossed a basic-block boundary
                continue
            if ln.op == "pop":
                # Walking backward past a pop — stack was deeper.
                depth += 1
                continue
            if ln.op == "push":
                if depth > 0:
                    # Matches a later (in source order) pop.
                    depth -= 1
                    continue
                # Outer push at chain start.
                if (
                    ln.operands.strip().lower()
                    == push_target.lower()
                ):
                    return (j, op_str)
                return None
            if (
                ln.op in ("add", "sub")
                and ln.operands.strip().lower().startswith("esp,")
            ):
                return None
            if ln.op == "call":
                return None  # call clobbers ECX/EDX, also stack
            # Memory write to store_dest — bail.
            mod_parts = _operands_split(ln.operands or "")
            if mod_parts is not None:
                m_dest, _ = mod_parts
                if (
                    "[" in m_dest
                    and self._mem_overlaps(
                        m_dest.strip(), store_dest
                    )
                ):
                    return None
        return None

    # Commutative subset of compound-assign ops. The short-tail form
    # `pop ecx; OP eax, ecx; mov [m], eax` is only correct when OP
    # is commutative (so `rhs OP lhs == lhs OP rhs`). sub is
    # excluded.
    _COMPOUND_BINOPS_COMMUTATIVE: frozenset[str] = frozenset({
        "add", "and", "or", "xor",
    })

    def _match_compound_assign_frame_short(
        self,
        lines: list[Line],
        store_idx: int,
        store_dest: str,
    ) -> tuple[int, str] | None:
        """Look back from `store_idx` for the 3-line short tail.
        The codegen emits this directly for commutative compound
        ops:
            store_idx - 1: ``OP eax, ecx`` (commutative OP).
            store_idx - 2: ``pop ecx``.
            ...chain (no stack manipulation, no [m] modification)...
            push_idx: ``push dword [m]`` matching store_dest.

        For commutative OPs, ``rhs OP lhs == lhs OP rhs``, so the
        codegen skips the canonical save-restore (``mov ecx, eax;
        pop eax``) and uses the simpler ``pop ecx`` directly.

        Returns (push_idx, op_str) on match, None otherwise.
        """
        ops_at = lambda j: (
            lines[j] if 0 <= j < len(lines) else None
        )
        # store_idx - 1: OP eax, ecx (commutative).
        op_line = ops_at(store_idx - 1)
        if op_line is None or op_line.kind != "instr":
            return None
        if op_line.op not in self._COMPOUND_BINOPS_COMMUTATIVE:
            return None
        op_parts = _operands_split(op_line.operands)
        if op_parts is None:
            return None
        if (
            op_parts[0].strip().lower() != "eax"
            or op_parts[1].strip().lower() != "ecx"
        ):
            return None
        op_str = op_line.op
        # store_idx - 2: pop ecx.
        pop_line = ops_at(store_idx - 2)
        if (
            pop_line is None
            or pop_line.kind != "instr"
            or pop_line.op != "pop"
            or pop_line.operands.strip().lower() != "ecx"
        ):
            return None
        # Walk back from store_idx - 3 to find the matching push.
        # Track stack depth — balanced inner push/pop pairs OK.
        push_target = f"dword {store_dest}"
        depth = 0
        for j in range(store_idx - 3, -1, -1):
            ln = lines[j]
            if ln.kind != "instr":
                if ln.kind == "label":
                    return None
                continue
            if ln.op == "pop":
                depth += 1
                continue
            if ln.op == "push":
                if depth > 0:
                    depth -= 1
                    continue
                if (
                    ln.operands.strip().lower()
                    == push_target.lower()
                ):
                    return (j, op_str)
                return None
            if (
                ln.op in ("add", "sub")
                and ln.operands.strip().lower().startswith("esp,")
            ):
                return None
            if ln.op == "call":
                return None
            mod_parts = _operands_split(ln.operands or "")
            if mod_parts is not None:
                m_dest, _ = mod_parts
                if (
                    "[" in m_dest
                    and self._mem_overlaps(
                        m_dest.strip(), store_dest
                    )
                ):
                    return None
        return None

    @staticmethod
    def _mem_overlaps(addr1: str, addr2: str) -> bool:
        """Conservative overlap check: True if two memory operands
        could refer to the same bytes. Disjoint ebp-relative memory
        operands return False; everything else returns True (be
        defensive).

        For our pattern (we want to detect modification of `addr2`
        within the chain), a False return means "definitely
        disjoint, safe to ignore"."""
        # Strip optional size keyword.
        def strip(text: str) -> str:
            t = text.strip()
            for prefix in ("dword ", "word ", "byte ", "qword "):
                if t.lower().startswith(prefix):
                    t = t[len(prefix):].lstrip()
            return t.lower()

        a, b = strip(addr1), strip(addr2)
        if a == b:
            return True
        # Both ebp-rel literal? Disjoint if different offsets.
        ebp_re = re.compile(
            r"^\[ebp\s*([+-])\s*(\d+)\]$"
        )
        ma, mb = ebp_re.match(a), ebp_re.match(b)
        if ma and mb:
            sa = (1 if ma.group(1) == "+" else -1) * int(ma.group(2))
            sb = (1 if mb.group(1) == "+" else -1) * int(mb.group(2))
            return sa == sb
        # Conservative: assume overlap.
        return True

    def _pass_redundant_xor_zero(
        self, lines: list[Line]
    ) -> list[Line]:
        """Drop ``xor reg, reg`` when reg is already known to be zero
        from a recent prior ``xor reg, reg`` and nothing modified it
        since.

        Tracks a per-register "is zero" state forward through a
        single basic block. Invalidated by:
        - Any write to the reg (mov, add, sub, xor with another src,
          inc, dec, neg, shl, etc.).
        - A label (control-flow merge — multiple incoming paths
          might have different states).
        - Unconditional jmp / ret / leave / enter (next instr is
          unreachable as fall-through).
        - Calls (preserve EBX/ESI/EDI/EBP per cdecl, but clobber
          EAX/ECX/EDX).

        Conditional jumps (jcc) preserve the state on the fall-
        through path; the taken path's state is forgotten (we don't
        track per-target zero states like redundant_eax_load does).

        Saves 2 bytes per match (xor reg, reg is `33 modrm` for the
        gp 32-bit registers = 2 bytes).
        """
        out: list[Line] = []
        zero_regs: set[str] = set()
        # Caller-saved regs that get clobbered by `call`.
        CALLER_SAVED = {"eax", "ecx", "edx"}
        for line in lines:
            if line.kind != "instr":
                if line.kind == "label":
                    zero_regs = set()
                out.append(line)
                continue
            op = line.op
            ops = line.operands
            # Detect `xor reg, reg`: drop if reg is already 0.
            if op == "xor":
                parts = _operands_split(ops)
                if parts is not None:
                    a, b = parts
                    al = a.strip().lower()
                    bl = b.strip().lower()
                    if al == bl and self._is_general_register(al):
                        if al in zero_regs:
                            self.stats[
                                "redundant_xor_zero"
                            ] = (
                                self.stats.get(
                                    "redundant_xor_zero", 0
                                ) + 1
                            )
                            continue
                        zero_regs.add(al)
                        out.append(line)
                        continue
            # Other instructions: invalidate any reg they write.
            # `mov [...], reg` doesn't write reg (just reads).
            # `mov reg, X` writes reg.
            if op == "mov":
                parts = _operands_split(ops)
                if parts is not None:
                    dest, _ = parts
                    dest_low = dest.strip().lower()
                    # Direct 32-bit reg write.
                    if (
                        self._is_general_register(dest_low)
                        and dest_low in zero_regs
                    ):
                        zero_regs.discard(dest_low)
                    # Sub-register write (al/ah/ax/bl/bh/bx/etc.) —
                    # invalidates the enclosing 32-bit reg.
                    elif dest_low in self._SUBREG_TO_GP32:
                        gp32 = self._SUBREG_TO_GP32[dest_low]
                        zero_regs.discard(gp32)
                # `mov [...], reg` — no reg-dest write, preserve.
                out.append(line)
                continue
            # cmp/test/push: read-only on regs.
            if op in {"cmp", "test", "push"}:
                out.append(line)
                continue
            # Calls: clobber caller-saved regs.
            if op == "call":
                for r in CALLER_SAVED:
                    zero_regs.discard(r)
                out.append(line)
                continue
            # Unconditional jumps / returns: invalidate everything.
            if op in {"jmp", "ret", "iret", "iretd", "retf",
                      "retn", "leave", "enter"}:
                zero_regs = set()
                out.append(line)
                continue
            # Conditional jumps (jcc): preserve state on fallthrough.
            if op.startswith("j"):
                out.append(line)
                continue
            # Instructions that may write a tracked reg. Conservative:
            # if any tracked reg appears in operands as dest (or
            # appears at all for read+write ops), invalidate.
            # For 2-operand ops: dest is first operand.
            parts = _operands_split(ops)
            if parts is not None:
                dest, _ = parts
                dest_stripped = dest.strip().lower()
                if (
                    self._is_general_register(dest_stripped)
                    and dest_stripped in zero_regs
                ):
                    zero_regs.discard(dest_stripped)
                elif dest_stripped in self._SUBREG_TO_GP32:
                    gp32 = self._SUBREG_TO_GP32[dest_stripped]
                    zero_regs.discard(gp32)
            # Single-operand ops (inc/dec/neg/not/idiv/etc.): operand
            # is dest+source.
            else:
                op_stripped = ops.strip().lower()
                if self._is_general_register(op_stripped):
                    zero_regs.discard(op_stripped)
                elif op_stripped in self._SUBREG_TO_GP32:
                    gp32 = self._SUBREG_TO_GP32[op_stripped]
                    zero_regs.discard(gp32)
            # `cdq` writes EDX; `lodsd` writes EAX; etc. — implicit
            # writers. Conservative: clear all caller-saved regs.
            if op in PeepholeOptimizer._IMPLICIT_REG_USERS:
                for r in CALLER_SAVED:
                    zero_regs.discard(r)
            out.append(line)
        return out

    # Commutative two-operand binops where `OP eax, ecx` and
    # `OP ecx, eax` produce identical EAX results. imul (two-operand
    # form) is included; sub/div/idiv etc. are NOT.
    _COMMUTATIVE_BINOPS: frozenset[str] = frozenset({
        "add", "and", "or", "xor", "imul",
    })

    def _pass_transfer_pop_collapse(
        self, lines: list[Line]
    ) -> list[Line]:
        """Collapse the binop tail `mov ecx, eax; pop eax; OP eax, ecx`
        into `pop ecx; OP eax, ecx` when OP is commutative.

        The original sequence saves the chain's EAX value into ECX,
        restores the LHS from stack to EAX, then computes EAX = LHS
        OP RHS. For commutative ops the operand order doesn't matter,
        so we can pop the saved LHS into ECX directly and compute
        EAX = RHS OP LHS in place.

        Saves 2 bytes per match (drops the `mov ecx, eax`).

        Conditions:
        - Three consecutive instr lines.
        - Line A: ``mov ecx, eax``.
        - Line B: ``pop eax``.
        - Line C: ``OP eax, ecx`` with OP ∈ {add, and, or, xor, imul}.
        - ECX is dead before the chain (the chain only uses ECX as
          scratch; we change which value lands there but the next
          read of ECX overwrites it). This is the codegen invariant
          for stack-machine binops.

        Note: ECX's post-collapse value is the LHS instead of the
        RHS. Any reader of ECX between `OP` and the next ECX-writer
        would see a different value. The codegen never generates
        such readers — the binop tail always either ends the
        expression or feeds into a `push eax` / next subexpression
        that re-loads ECX from scratch.
        """
        out = list(lines)
        i = 0
        while i + 2 < len(out):
            a = out[i]
            b = out[i + 1]
            c = out[i + 2]
            if not (
                a.kind == "instr" and a.op == "mov"
                and a.operands.strip().lower() == "ecx, eax"
                and b.kind == "instr" and b.op == "pop"
                and b.operands.strip().lower() == "eax"
                and c.kind == "instr"
                and c.op in PeepholeOptimizer._COMMUTATIVE_BINOPS
            ):
                i += 1
                continue
            cp = _operands_split(c.operands)
            if cp is None:
                i += 1
                continue
            cdst, csrc = cp
            if (
                cdst.strip().lower() != "eax"
                or csrc.strip().lower() != "ecx"
            ):
                i += 1
                continue
            # Rewrite: drop A, replace B with `pop ecx`, keep C.
            indent = self._extract_indent(b.raw)
            new_pop = Line(
                raw=f"{indent}pop     ecx",
                kind="instr",
                op="pop",
                operands="ecx",
            )
            out = out[:i] + [new_pop, c] + out[i + 3:]
            self.stats["transfer_pop_collapse"] = (
                self.stats.get("transfer_pop_collapse", 0) + 1
            )
            continue
        return out

    def _pass_dup_push_pop_self_op(
        self, lines: list[Line]
    ) -> list[Line]:
        """Collapse the canonical "X OP X" pattern where the codegen
        pushes one copy of X and reloads X for the second operand.

        Pattern (4 consecutive instr lines):
            push    X
            mov     reg1, X
            pop     reg2
            OP      reg1, reg2

        Rewrite to:
            mov     reg1, X
            OP      reg1, reg1

        Saves 2 instructions per match. The push/pop pair is dead
        because both copies of X are equal (the mov reloaded it from
        the same memory) — the OP can use reg1 as both operands.

        Common in `arr[i] * arr[i]` style where the codegen evaluates
        the left to memory-push form (because evaluating the right
        will clobber EAX), then reloads from the same memory address.

        Conditions:
        - Line 1 op = "push", any operand X.
        - Line 2 op = "mov", parts = (reg1, X) where reg1 is a 32-bit
          GP register and X matches line 1's push operand text.
        - Line 3 op = "pop", operand = reg2 (32-bit GP register, ≠
          reg1).
        - Line 4 op ∈ {add, and, or, xor, imul} (commutative), parts
          = (reg1, reg2).
        - X must not reference esp (push/pop modify ESP between
          line 1 and line 3; if X depends on esp, the mov in line 2
          could read different bytes than the push wrote).
        - X must not reference reg1 if reg1 is non-eax — actually
          this isn't an issue because we keep line 2 (the mov) which
          loads X to reg1 in the original order. The OP comes after
          the load, so reg1 holds X. We're just changing the OP's
          second operand from reg2 (= X via pop) to reg1 (= X via
          mov). Both equal X.
        - reg2 must be dead after line 4. The original ends with
          reg2 = X; the rewrite leaves reg2 unchanged.
        """
        out = list(lines)
        i = 0
        while i < len(out):
            instrs = self._next_n_instrs(out, i, 4)
            if instrs is None:
                i += 1
                continue
            (a_idx, a), (b_idx, b), (c_idx, c), (d_idx, d) = instrs
            if a.op != "push" or b.op != "mov" or c.op != "pop":
                i += 1
                continue
            if d.op not in PeepholeOptimizer._COMMUTATIVE_BINOPS:
                i += 1
                continue
            push_x = a.operands.strip()
            # Line 2: mov reg1, X
            bp = _operands_split(b.operands)
            if bp is None:
                i += 1
                continue
            b_dst, b_src = bp
            reg1 = b_dst.strip().lower()
            if reg1 not in PeepholeOptimizer._GP32:
                i += 1
                continue
            # Compare push X and mov src X. They must read identical
            # bytes; we compare normalized text. The push may have a
            # `dword` prefix that the mov source omits; strip it.
            push_x_norm = self._strip_size_prefix(push_x)
            mov_src_norm = self._strip_size_prefix(b_src.strip())
            if push_x_norm.lower() != mov_src_norm.lower():
                i += 1
                continue
            # X must not reference esp (push moves esp).
            if _references_register(push_x_norm, "esp"):
                i += 1
                continue
            # Line 3: pop reg2
            reg2 = c.operands.strip().lower()
            if reg2 not in PeepholeOptimizer._GP32:
                i += 1
                continue
            if reg2 == reg1:
                i += 1
                continue
            # Line 4: OP reg1, reg2
            dp = _operands_split(d.operands)
            if dp is None:
                i += 1
                continue
            d_dst, d_src = dp
            if d_dst.strip().lower() != reg1:
                i += 1
                continue
            if d_src.strip().lower() != reg2:
                i += 1
                continue
            # reg2 must be dead after line 4.
            if not self._reg_dead_after(out, d_idx + 1, reg2):
                i += 1
                continue
            # Rewrite: drop A (push) and C (pop); modify D's operand.
            indent = self._extract_indent(d.raw)
            spaces = max(1, 8 - len(d.op))
            new_d = Line(
                raw=f"{indent}{d.op}{' ' * spaces}{reg1}, {reg1}",
                kind="instr",
                op=d.op,
                operands=f"{reg1}, {reg1}",
            )
            out = (
                out[:a_idx]
                + [b]
                + [new_d]
                + out[d_idx + 1:]
            )
            self.stats["dup_push_pop_self_op"] = (
                self.stats.get("dup_push_pop_self_op", 0) + 1
            )
            # Don't advance i; new line at i may fire another pass.
            continue
        return out

    _GP32: frozenset[str] = frozenset({
        "eax", "ebx", "ecx", "edx", "esi", "edi", "ebp", "esp",
    })

    def _pass_push_pop_op_to_memop(
        self, lines: list[Line]
    ) -> list[Line]:
        """Collapse the LHS-saved-across-call binop pattern:

            push    dword [ebp ± N]    ; save LHS
            ... chain (incl. calls) ... ; produces RHS in EAX
            pop     reg2                ; restore LHS into reg2
            OP      reg1, reg2          ; reg1 = chain result (EAX)

        Rewrite to:

            ... chain ...
            OP      reg1, [ebp ± N]    ; OP reg-mem (commutative)

        Saves 2 instructions per match (drops the push and the pop).
        Common in recursive functions where the LHS is saved across
        the recursive call (e.g. ``n * factorial(n-1)``).

        Conditions:
        - Line 1: ``push <size?> [ebp + N]`` or ``push <size?> [ebp - N]``
          where N is a numeric literal. Restricted to ebp-relative
          addressing because ebp is callee-saved (cdecl), so X's
          memory location is preserved across calls.
        - Lines 2..N: chain. Stack-balanced (every inner push has a
          matching inner pop or `add esp, K` cleanup). Must not store
          to X (no `mov [ebp ± N], ...` with the same offset).
          Calls allowed; we trust cdecl + ebp-callee-saved.
        - Line N+1: ``pop reg2`` (32-bit GP register, ≠ reg1).
        - Line N+2: ``OP reg1, reg2`` where OP ∈ commutative binops.
        - reg2 must be dead after line N+2.
        - Chain must not take the address of X via ``lea reg, [ebp ± N]``
          with the same offset (a pointer to X could be passed to a
          call that mutates it).
        - Excludes the narrow 4-line case (chain length 1 with same X)
          which is handled by `_pass_dup_push_pop_self_op` and saves
          1 byte more (reg-reg form vs reg-mem form).
        """
        out = list(lines)
        i = 0
        while i < len(out):
            push_line = out[i]
            if (
                push_line.kind != "instr"
                or push_line.op != "push"
            ):
                i += 1
                continue
            push_x = push_line.operands.strip()
            x_clean = self._strip_size_prefix(push_x)
            x_off = self._ebp_offset(x_clean)
            if x_off is None:
                i += 1
                continue
            # Function-wide aliasing check: bail if any `lea` in the
            # function takes the address of X's slot. If `&X` is
            # captured anywhere, a call in the chain might mutate X
            # via that pointer, and the saved-value-vs-current-value
            # difference would be observable. (The codegen pushes
            # unconditionally — it doesn't know whether the call
            # could mutate; my pass assumes X is unchanged.)
            if self._function_takes_ebp_addr(out, i, x_off):
                i += 1
                continue
            # Walk forward to find the matching pop. Track stack
            # depth — balanced inner push/pop pairs and add/sub esp
            # adjustments are part of the chain.
            depth = 1
            pop_idx = None
            chain_safe = True
            j = i + 1
            while j < len(out):
                ln = out[j]
                if ln.kind == "label":
                    chain_safe = False
                    break
                if ln.kind != "instr":
                    j += 1
                    continue
                op = ln.op
                if op == "push":
                    depth += 1
                    j += 1
                    continue
                if op == "pop":
                    depth -= 1
                    if depth == 0:
                        pop_idx = j
                        break
                    j += 1
                    continue
                if op in ("ret", "iret", "iretd", "retf", "retn",
                          "leave", "jmp", "enter"):
                    chain_safe = False
                    break
                if (
                    op in ("add", "sub")
                    and ln.operands.lower().startswith("esp,")
                ):
                    # Adjust depth: `add esp, 4` pops 1, `sub esp, 4`
                    # pushes 1. We don't know the K dynamically — but
                    # after `call`, ``add esp, K`` is the standard
                    # cleanup. Treat as zero-net inner push/pop:
                    # if it's `add esp, K`, decrement depth by K/4;
                    # if `sub esp, K`, increment by K/4.
                    parts_esp = _operands_split(ln.operands)
                    if parts_esp is None:
                        chain_safe = False
                        break
                    _, esp_amt = parts_esp
                    try:
                        amt = int(esp_amt.strip(), 0)
                    except ValueError:
                        chain_safe = False
                        break
                    if amt % 4 != 0:
                        chain_safe = False
                        break
                    delta = amt // 4
                    if op == "add":
                        depth -= delta
                    else:
                        depth += delta
                    if depth <= 0:
                        # add esp, K can pop the saved X if K is too
                        # large. Bail.
                        chain_safe = False
                        break
                    j += 1
                    continue
                # Memory store check: bail if writes to X's slot.
                op_parts = _operands_split(ln.operands or "")
                if op_parts is not None:
                    m_dest, _ = op_parts
                    if "[" in m_dest:
                        m_off = self._ebp_offset(
                            self._strip_size_prefix(m_dest.strip())
                        )
                        if m_off is not None and m_off == x_off:
                            chain_safe = False
                            break
                # LEA of address-of-X: bail (pointer escape).
                if op == "lea":
                    if op_parts is not None:
                        _, lea_src = op_parts
                        lea_off = self._ebp_offset(
                            lea_src.strip()
                        )
                        if lea_off is not None and lea_off == x_off:
                            chain_safe = False
                            break
                j += 1
            if not chain_safe or pop_idx is None:
                i += 1
                continue
            # Chain must have at least 2 instructions (the narrow
            # 1-instr case is handled by _pass_dup_push_pop_self_op
            # and saves 1 byte more via reg-reg form).
            chain_instrs = sum(
                1 for k in range(i + 1, pop_idx)
                if out[k].kind == "instr"
            )
            if chain_instrs < 2:
                i += 1
                continue
            # Pop reg2 (32-bit GP).
            pop_reg = out[pop_idx].operands.strip().lower()
            if pop_reg not in PeepholeOptimizer._GP32:
                i += 1
                continue
            # Find the OP line right after pop (skip blanks/comments).
            op_idx = None
            for k in range(pop_idx + 1, len(out)):
                if out[k].kind == "label":
                    break
                if out[k].kind == "instr":
                    op_idx = k
                    break
            if op_idx is None:
                i += 1
                continue
            op_line = out[op_idx]
            if op_line.op not in PeepholeOptimizer._COMMUTATIVE_BINOPS:
                i += 1
                continue
            opp = _operands_split(op_line.operands)
            if opp is None:
                i += 1
                continue
            reg1 = opp[0].strip().lower()
            op_src = opp[1].strip().lower()
            if reg1 not in PeepholeOptimizer._GP32 or reg1 == pop_reg:
                i += 1
                continue
            if op_src != pop_reg:
                i += 1
                continue
            # reg2 (pop_reg) dead after op_idx.
            if not self._reg_dead_after(out, op_idx + 1, pop_reg):
                i += 1
                continue
            # Rewrite. New OP line uses memory operand X.
            indent = self._extract_indent(op_line.raw)
            spacer = " " * max(8 - len(op_line.op), 1)
            mem_text = f"dword {x_clean}"
            new_op_raw = (
                f"{indent}{op_line.op}{spacer}{reg1}, {mem_text}"
            )
            new_op_line = Line(
                raw=new_op_raw,
                kind="instr",
                op=op_line.op,
                operands=f"{reg1}, {mem_text}",
            )
            new_out = []
            new_out.extend(out[:i])  # before push
            new_out.extend(out[i + 1:pop_idx])  # chain (drop push)
            new_out.extend(out[pop_idx + 1:op_idx])  # between pop & op
            new_out.append(new_op_line)
            new_out.extend(out[op_idx + 1:])
            out = new_out
            self.stats["push_pop_op_to_memop"] = (
                self.stats.get("push_pop_op_to_memop", 0) + 1
            )
            # Don't advance i; new line may fire another pass.
            continue
        return out

    def _function_takes_ebp_addr(
        self, lines: list[Line], cur_idx: int, off: int
    ) -> bool:
        """Scan the enclosing function for any `lea reg, [ebp ± off]`
        that captures the address of the slot at the given offset.

        Walks outward from cur_idx until hitting a function-defining
        label (a top-level label whose name starts with `_` and isn't
        a `.local`). Conservatively returns True on syntactic match;
        we don't try to prove that the captured address doesn't
        escape — any escape opens the door to mid-chain mutation.
        """
        # Find function start: walk backward to the most recent
        # top-level label (kind=label, label starts with `_`, no `.`).
        start = 0
        for k in range(cur_idx - 1, -1, -1):
            ln = lines[k]
            if ln.kind == "label":
                lab = ln.label
                if lab and lab.startswith("_") and "." not in lab:
                    start = k + 1
                    break
        # Find function end: walk forward to the next top-level
        # label (or end of file).
        end = len(lines)
        for k in range(cur_idx + 1, len(lines)):
            ln = lines[k]
            if ln.kind == "label":
                lab = ln.label
                if lab and lab.startswith("_") and "." not in lab:
                    end = k
                    break
        # Build the X-pattern text we're matching against.
        sign = "+" if off > 0 else "-"
        target = f"[ebp {sign} {abs(off)}]"
        for k in range(start, end):
            ln = lines[k]
            if ln.kind != "instr" or ln.op != "lea":
                continue
            opp = _operands_split(ln.operands)
            if opp is None:
                continue
            _, src = opp
            src_norm = src.strip().lower()
            if src_norm == target:
                return True
        return False

    @staticmethod
    def _strip_size_prefix(text: str) -> str:
        """Strip optional NASM size prefix (`dword`, `word`, `byte`,
        `qword`) from a memory operand text."""
        s = text.strip()
        for prefix in ("dword ", "word ", "byte ", "qword "):
            if s.lower().startswith(prefix):
                return s[len(prefix):].lstrip()
        return s

    def _pass_label_load_collapse(
        self, lines: list[Line]
    ) -> list[Line]:
        """Collapse ``mov REG1, LABEL; mov REG2, [REG1]`` into
        ``mov REG2, [LABEL]`` using x86's disp32 absolute addressing.

        Saves 1 byte per match (5+2=7 bytes → 6 bytes for disp32-
        absolute mov). Common in global-variable access where the
        codegen emits a label load followed by a deref.

        Also handles the scaled-index case:
        ``mov REG1, LABEL; mov REG2, [REG1 + IDX*SCALE]`` →
        ``mov REG2, [LABEL + IDX*SCALE]``. Saves 1 byte (drops the
        REG1 load; the SIB-form mov adds disp32 to the addressing
        mode).

        Conditions:
        - Two consecutive instr lines.
        - Line A: ``mov REG1, LABEL`` where LABEL is a non-numeric,
          non-memory expression (label or label-arithmetic).
        - Line B: ``mov REG2, [REG1]`` or ``mov REG2, [REG1 + IDX*SCALE]``
          (no offset on REG1).
        - REG1 either == REG2 (overwritten by the load) or dead
          after Line B.
        """
        out = list(lines)
        i = 0
        while i + 1 < len(out):
            a = out[i]
            b = out[i + 1]
            if not (
                a.kind == "instr" and a.op == "mov"
                and b.kind == "instr" and b.op == "mov"
            ):
                i += 1
                continue
            ap = _operands_split(a.operands)
            bp = _operands_split(b.operands)
            if ap is None or bp is None:
                i += 1
                continue
            r1 = ap[0].strip().lower()
            label = ap[1].strip()
            r2 = bp[0].strip().lower()
            src = bp[1].strip()
            if (
                not self._is_general_register(r1)
                or not self._is_general_register(r2)
            ):
                i += 1
                continue
            # LABEL must not be a number, register, or memory ref.
            if (
                "[" in label
                or self._is_general_register(label.lower())
            ):
                i += 1
                continue
            try:
                int(label)
                # numeric — that's `mov reg, IMM`. Not a label.
                i += 1
                continue
            except ValueError:
                pass
            # Source: `[REG1]` or `[REG1 + IDX*SCALE]`. Strip size.
            src_stripped = src
            for prefix in ("dword ", "word ", "byte ", "qword "):
                if src_stripped.lower().startswith(prefix):
                    src_stripped = src_stripped[
                        len(prefix):
                    ].lstrip()
                    break
            # Plain `[REG1]`.
            m_plain = re.match(
                r"^\[\s*([a-zA-Z]+)\s*\]$", src_stripped
            )
            # SIB form `[REG1 + IDX*SCALE]`.
            m_sib = re.match(
                r"^\[\s*([a-zA-Z]+)\s*\+\s*"
                r"([a-zA-Z]+)\s*\*\s*([1248])\s*\]$",
                src_stripped,
            )
            if m_plain:
                base = m_plain.group(1).lower()
                sib_tail = ""
            elif m_sib:
                base = m_sib.group(1).lower()
                idx = m_sib.group(2).lower()
                scale = m_sib.group(3)
                # Don't fire when idx == r1 (the SIB also references
                # r1). Reordering: if idx is the label-loaded reg, we
                # can't rewrite — the index would be the label value.
                if idx == r1:
                    i += 1
                    continue
                sib_tail = f" + {idx}*{scale}"
            else:
                i += 1
                continue
            if base != r1:
                i += 1
                continue
            # Liveness: r1 dead after B (or r1 == r2).
            if r1 != r2 and not self._reg_dead_after(
                out, i + 2, r1
            ):
                i += 1
                continue
            # Build rewrite: `mov r2, [LABEL[+ idx*scale]]`.
            indent = self._extract_indent(b.raw)
            new_src = f"[{label}{sib_tail}]"
            new_raw = f"{indent}mov     {r2}, {new_src}"
            new_line = Line(
                raw=new_raw,
                kind="instr",
                op="mov",
                operands=f"{r2}, {new_src}",
            )
            out = out[:i] + [new_line] + out[i + 2:]
            self.stats["label_load_collapse"] = (
                self.stats.get("label_load_collapse", 0) + 1
            )
            continue
        return out

    def _pass_pop_index_push_collapse(
        self, lines: list[Line]
    ) -> list[Line]:
        """Collapse the array-index-then-push pattern that uses a
        prior `pop` for the base.

        Pattern:
            shl IDX, N           ; scale index
            pop BASE             ; restore previously-pushed base
            add IDX, BASE        ; compute address
            push dword [IDX]     ; push arr[i]

        Rewrite:
            pop BASE
            push dword [BASE + IDX*SCALE]

        Saves 4 bytes per match (drops shl + add; the SIB-form push
        adds 1 byte over the plain `push dword [reg]` form).

        Conditions:
        - 4 consecutive instr lines.
        - Line A: ``shl IDX, N`` for N ∈ {1, 2, 3}.
        - Line B: ``pop BASE``.
        - Line C: ``add IDX, BASE``.
        - Line D: ``push dword [IDX]``.
        - IDX dead after Line D (the SIB-form push reads IDX but
          doesn't write; original ended with IDX = base + idx*scale,
          new ends with IDX = unscaled index — different post-state).
        """
        SCALE = {1: 2, 2: 4, 3: 8}
        out = list(lines)
        i = 0
        while i + 3 < len(out):
            a = out[i]
            b = out[i + 1]
            c = out[i + 2]
            d = out[i + 3]
            if not (
                a.kind == "instr" and a.op == "shl"
                and b.kind == "instr" and b.op == "pop"
                and c.kind == "instr" and c.op == "add"
                and d.kind == "instr" and d.op == "push"
            ):
                i += 1
                continue
            ap = _operands_split(a.operands)
            cp = _operands_split(c.operands)
            if ap is None or cp is None:
                i += 1
                continue
            idx_reg = ap[0].strip().lower()
            try:
                n = int(ap[1].strip())
            except ValueError:
                i += 1
                continue
            if n not in SCALE:
                i += 1
                continue
            base_reg = b.operands.strip().lower()
            c_dst = cp[0].strip().lower()
            c_src = cp[1].strip().lower()
            if (
                not self._is_general_register(idx_reg)
                or not self._is_general_register(base_reg)
                or c_dst != idx_reg
                or c_src != base_reg
                or idx_reg == base_reg
            ):
                i += 1
                continue
            # Line D's operand: `dword [IDX]`.
            d_op = d.operands.strip()
            for prefix in ("dword ", "word ", "byte ", "qword "):
                if d_op.lower().startswith(prefix):
                    d_op_stripped = d_op[len(prefix):].lstrip()
                    break
            else:
                d_op_stripped = d_op
            mem_re = re.match(
                r"^\[\s*([a-zA-Z]+)\s*\]$", d_op_stripped
            )
            if mem_re is None or mem_re.group(1).lower() != idx_reg:
                i += 1
                continue
            # IDX dead after line D.
            if not self._reg_dead_after(out, i + 4, idx_reg):
                i += 1
                continue
            # Build rewrite.
            scale = SCALE[n]
            indent_b = self._extract_indent(b.raw)
            indent_d = self._extract_indent(d.raw)
            new_pop = Line(
                raw=b.raw,
                kind="instr", op="pop", operands=base_reg,
            )
            new_push_src = (
                f"dword [{base_reg} + {idx_reg}*{scale}]"
            )
            new_push = Line(
                raw=f"{indent_d}push    {new_push_src}",
                kind="instr",
                op="push",
                operands=new_push_src,
            )
            out = out[:i] + [new_pop, new_push] + out[i + 4:]
            self.stats["pop_index_push_collapse"] = (
                self.stats.get(
                    "pop_index_push_collapse", 0
                ) + 1
            )
            continue
        return out

    def _pass_pop_index_load_collapse(
        self, lines: list[Line]
    ) -> list[Line]:
        """Sister of ``_pass_pop_index_push_collapse`` for the LOAD
        end. Collapse the array-index-then-load pattern that uses a
        prior ``pop`` for the base.

        Pattern:
            shl IDX, N           ; scale index
            pop BASE             ; restore previously-pushed base
            add IDX, BASE        ; compute address
            mov DST, [IDX]       ; load arr[i]

        Rewrite:
            pop BASE
            mov DST, [BASE + IDX*SCALE]

        Saves 5 bytes per match (drops shl + add; the SIB-form mov
        adds 1 byte over the plain ``mov DST, [reg]`` form).

        Conditions:
        - 4 consecutive instr lines.
        - Line A: ``shl IDX, N`` for N ∈ {1, 2, 3}.
        - Line B: ``pop BASE``.
        - Line C: ``add IDX, BASE``.
        - Line D: ``mov DST, [IDX]``.
        - IDX dead after Line D unless IDX == DST (then DST kills
          IDX's register naturally — it's the load target).
        - DST may be IDX, BASE, or different — all valid SIB forms.

        Common in compound-assign chains where the rhs is an array
        load through a previously-saved base — e.g. ``s += arr[i]``
        (after compound_assign_collapse fired) or the index-load
        directly inside any expression.
        """
        SCALE = {1: 2, 2: 4, 3: 8}
        out = list(lines)
        i = 0
        while i + 3 < len(out):
            a = out[i]
            b = out[i + 1]
            c = out[i + 2]
            d = out[i + 3]
            if not (
                a.kind == "instr" and a.op == "shl"
                and b.kind == "instr" and b.op == "pop"
                and c.kind == "instr" and c.op == "add"
                and d.kind == "instr" and d.op == "mov"
            ):
                i += 1
                continue
            ap = _operands_split(a.operands)
            cp = _operands_split(c.operands)
            dp = _operands_split(d.operands)
            if ap is None or cp is None or dp is None:
                i += 1
                continue
            idx_reg = ap[0].strip().lower()
            try:
                n = int(ap[1].strip())
            except ValueError:
                i += 1
                continue
            if n not in SCALE:
                i += 1
                continue
            base_reg = b.operands.strip().lower()
            c_dst = cp[0].strip().lower()
            c_src = cp[1].strip().lower()
            if (
                not self._is_general_register(idx_reg)
                or not self._is_general_register(base_reg)
                or c_dst != idx_reg
                or c_src != base_reg
                or idx_reg == base_reg
            ):
                i += 1
                continue
            # Line D: mov DST, [IDX]
            d_dst = dp[0].strip().lower()
            d_src = dp[1].strip()
            if not self._is_general_register(d_dst):
                i += 1
                continue
            for prefix in ("dword ", "word ", "byte ", "qword "):
                if d_src.lower().startswith(prefix):
                    d_src = d_src[len(prefix):].lstrip()
                    break
            mem_re = re.match(
                r"^\[\s*([a-zA-Z]+)\s*\]$", d_src
            )
            if mem_re is None or mem_re.group(1).lower() != idx_reg:
                i += 1
                continue
            # IDX must be dead after the load — unless it's the
            # load target (in which case the load itself is the
            # final write).
            if d_dst != idx_reg:
                if not self._reg_dead_after(out, i + 4, idx_reg):
                    i += 1
                    continue
            # Build rewrite.
            scale = SCALE[n]
            indent_d = self._extract_indent(d.raw)
            new_pop = Line(
                raw=b.raw,
                kind="instr", op="pop", operands=base_reg,
            )
            new_load_src = (
                f"[{base_reg} + {idx_reg}*{scale}]"
            )
            new_load = Line(
                raw=f"{indent_d}mov     {d_dst}, {new_load_src}",
                kind="instr",
                op="mov",
                operands=f"{d_dst}, {new_load_src}",
            )
            out = out[:i] + [new_pop, new_load] + out[i + 4:]
            self.stats["pop_index_load_collapse"] = (
                self.stats.get(
                    "pop_index_load_collapse", 0
                ) + 1
            )
            continue
        return out

    def _pass_push_pop_to_mov(
        self, lines: list[Line]
    ) -> list[Line]:
        """Replace ``push X; ... stack-balanced chain ... ; pop REG``
        with ``mov REG, X`` (placed where the pop was), dropping the
        push entirely.

        Saves 1 byte per match — push X + pop reg vs. mov reg, X.

        X may be:
        - A numeric immediate or label/label-arithmetic expression
          (the original case).
        - A memory operand (``[ebp ± N]``, ``[label]``, etc.) where
          the chain doesn't write to memory aliasing X.

        Conditions:
        - First instr: ``push X``.
        - Walk forward, tracking stack depth. Match the FIRST
          ``pop REG`` at depth 0.
        - Chain between push and pop must not access [esp + N], call,
          or ret/leave/enter. Any jump fences the rewrite.
        - If X is memory, the chain must not write to memory that
          may alias X.

        Common after pop_index_*_collapse drops the indexing chain
        but leaves the original `push BASE` and `pop BASE_REG` —
        which now have no chain between them or a trivial chain.
        Also fires on struct-copy retptr-save patterns.
        """
        out = list(lines)
        i = 0
        while i < len(out):
            a = out[i]
            if not (a.kind == "instr" and a.op == "push"):
                i += 1
                continue
            push_op = a.operands.strip()
            # Skip register pushes — those are saving register values
            # and the codegen needs the round-trip.
            if self._is_general_register(push_op.lower()):
                i += 1
                continue
            # Optional size keyword strip.
            stripped = push_op
            for prefix in ("dword ", "word ", "byte ", "qword "):
                if stripped.lower().startswith(prefix):
                    stripped = stripped[len(prefix):].lstrip()
                    break
            # Is it a memory operand?
            push_is_mem = stripped.startswith("[")
            # For memory operands, restrict to ebp-relative literal
            # offsets (`[ebp + N]` / `[ebp - N]`). ebp is callee-saved
            # and frame-relative; the chain can't change ebp without
            # invoking control flow we already bail on. Other forms
            # like `[eax + ecx*4]` reference registers the chain may
            # modify, making the post-pop value differ from the
            # pushed value.
            if push_is_mem and self._ebp_offset(stripped) is None:
                i += 1
                continue
            # Walk forward looking for matching pop.
            depth = 1  # we're past the push
            match_idx = None
            failed = False
            j = i + 1
            while j < len(out):
                ln = out[j]
                if ln.kind == "label":
                    # Label is an external entry point — control
                    # flow may reach the pop without our push.
                    failed = True
                    break
                if ln.kind != "instr":
                    j += 1
                    continue
                if ln.op == "push":
                    depth += 1
                    j += 1
                    continue
                if ln.op == "pop":
                    if depth == 1:
                        # This matches our push.
                        target_reg = ln.operands.strip().lower()
                        if not self._is_general_register(target_reg):
                            failed = True
                        else:
                            match_idx = j
                        break
                    depth -= 1
                    j += 1
                    continue
                # Stack-affecting ops fence the search.
                if ln.op in {
                    "call", "ret", "iret", "iretd", "retf",
                    "retn", "leave", "enter",
                }:
                    failed = True
                    break
                if ln.op.startswith("j"):
                    # Any jump fences (control flow may not reach pop).
                    failed = True
                    break
                # Operand-references-esp fence.
                if "esp" in (ln.operands or "").lower():
                    failed = True
                    break
                # If X is memory, bail on chain instructions that
                # may write to memory aliasing X.
                if push_is_mem:
                    ops_text = ln.operands or ""
                    if (
                        ops_text
                        and ln.op
                        not in self._READ_ONLY_FIRST_MEM
                    ):
                        first_op = (
                            ops_text.split(",", 1)[0].strip()
                            if "," in ops_text
                            else ops_text.strip()
                        )
                        if "[" in first_op:
                            if not self._mem_disjoint(
                                self._strip_size_prefix(first_op),
                                self._strip_size_prefix(stripped),
                            ):
                                failed = True
                                break
                j += 1
            if failed or match_idx is None:
                i += 1
                continue
            # Build rewrite: drop push at i, replace pop at
            # match_idx with `mov target_reg, push_op`.
            indent = self._extract_indent(out[match_idx].raw)
            new_mov = Line(
                raw=f"{indent}mov     {target_reg}, {stripped}",
                kind="instr",
                op="mov",
                operands=f"{target_reg}, {stripped}",
            )
            new_out = (
                out[:i] + out[i + 1: match_idx]
                + [new_mov] + out[match_idx + 1:]
            )
            out = new_out
            self.stats["push_pop_to_mov"] = (
                self.stats.get("push_pop_to_mov", 0) + 1
            )
            # Don't advance i (might match again from the same i).
            continue
        return out

    def _pass_sib_const_index_fold(
        self, lines: list[Line]
    ) -> list[Line]:
        """Fold a constant-index SIB-form load. When the index
        register is loaded with a constant immediately before a
        SIB-form load that uses it, fold the constant into the
        displacement.

        Pattern (with optional displacement):
            mov IDX, IMM
            mov DST, [BASE + IDX*SCALE]
            mov DST, [BASE + IDX*SCALE + DISP]
            mov DST, [BASE + IDX*SCALE - DISP]

        Rewrite (drop the index load; fold IMM*SCALE into disp):
            mov DST, [BASE + (IMM*SCALE)]
            mov DST, [BASE + (IMM*SCALE + DISP)]
            mov DST, [BASE + (IMM*SCALE - DISP)]

        When the new displacement is 0, emit `[BASE]`. When BASE is
        the same register as IDX (rare), the rewrite is unsafe
        (BASE's value would change after the rewrite drops the
        const-load) and we bail.

        Saves 5 bytes per match (drops 5-byte `mov reg, imm32`).

        Conditions:
        - Two consecutive instr lines.
        - Line A: ``mov IDX, IMM`` where IDX is a 32-bit GP reg
          and IMM is a numeric literal.
        - Line B: ``mov DST, [BASE + IDX*SCALE [DISP]]`` (with
          DISP optional, ±N form).
        - IDX dead after Line B (unless IDX == DST, in which
          case the load target overwrites IDX naturally).
        """
        SCALE_VALUES = {1, 2, 4, 8}
        # Loads using SIB form: mov, movsx, movzx.
        LOAD_OPS = {"mov", "movsx", "movzx"}
        out = list(lines)
        i = 0
        while i + 1 < len(out):
            a = out[i]
            b = out[i + 1]
            if not (
                a.kind == "instr"
                and a.op in {"mov", "xor"}
                and b.kind == "instr" and b.op in LOAD_OPS
            ):
                i += 1
                continue
            ap = _operands_split(a.operands)
            bp = _operands_split(b.operands)
            if ap is None or bp is None:
                i += 1
                continue
            idx_reg = ap[0].strip().lower()
            if not self._is_general_register(idx_reg):
                i += 1
                continue
            # Determine constant value of idx_reg.
            if a.op == "xor":
                # `xor reg, reg` zeros the register.
                if ap[1].strip().lower() != idx_reg:
                    i += 1
                    continue
                imm_val = 0
            else:
                # `mov reg, IMM` — must be a numeric literal.
                imm_str = ap[1].strip()
                try:
                    if imm_str.lower().startswith("0x"):
                        imm_val = int(imm_str, 16)
                    elif imm_str.lower().startswith("-0x"):
                        imm_val = -int(imm_str[1:], 16)
                    else:
                        imm_val = int(imm_str)
                except ValueError:
                    i += 1
                    continue
            # Parse Line B.
            dst_reg = bp[0].strip().lower()
            b_src = bp[1].strip()
            if not self._is_general_register(dst_reg):
                i += 1
                continue
            # Strip size prefix.
            size_prefix = ""
            for prefix in ("dword ", "word ", "byte ", "qword "):
                if b_src.lower().startswith(prefix):
                    size_prefix = b_src[:len(prefix)]
                    b_src = b_src[len(prefix):].lstrip()
                    break
            # Match SIB form.
            sib_re = re.match(
                r"^\[\s*([a-zA-Z]+)\s*\+\s*([a-zA-Z]+)\s*\*"
                r"\s*(\d+)"
                r"(?:\s*([+-])\s*(\d+))?"
                r"\s*\]$",
                b_src,
            )
            if sib_re is None:
                i += 1
                continue
            base_reg = sib_re.group(1).lower()
            sib_idx = sib_re.group(2).lower()
            try:
                scale = int(sib_re.group(3))
            except ValueError:
                i += 1
                continue
            if scale not in SCALE_VALUES:
                i += 1
                continue
            disp_sign = sib_re.group(4)
            disp_str = sib_re.group(5)
            disp = 0
            if disp_str is not None:
                try:
                    disp = int(disp_str)
                except ValueError:
                    i += 1
                    continue
                if disp_sign == "-":
                    disp = -disp
            # SIB index reg must match our IDX.
            if sib_idx != idx_reg:
                i += 1
                continue
            # BASE must NOT be IDX (otherwise BASE's value changes
            # after we drop the const-load).
            if base_reg == idx_reg:
                i += 1
                continue
            if not self._is_general_register(base_reg):
                i += 1
                continue
            # IDX dead after, unless DST == IDX.
            if dst_reg != idx_reg:
                if not self._reg_dead_after(out, i + 2, idx_reg):
                    i += 1
                    continue
            # Compute new displacement.
            new_disp = imm_val * scale + disp
            indent = self._extract_indent(b.raw)
            if new_disp == 0:
                new_src = f"{size_prefix}[{base_reg}]"
            elif new_disp > 0:
                new_src = (
                    f"{size_prefix}[{base_reg} + {new_disp}]"
                )
            else:
                new_src = (
                    f"{size_prefix}[{base_reg} - {-new_disp}]"
                )
            opname = b.op  # mov / movsx / movzx — preserve.
            spacer = " " * max(8 - len(opname), 1)
            new_raw = (
                f"{indent}{opname}{spacer}{dst_reg}, {new_src}"
            )
            new_line = Line(
                raw=new_raw, kind="instr", op=opname,
                operands=f"{dst_reg}, {new_src}",
            )
            out = out[:i] + [new_line] + out[i + 2:]
            self.stats["sib_const_index_fold"] = (
                self.stats.get("sib_const_index_fold", 0) + 1
            )
            continue
        return out

    def _pass_push_const_index_fold(
        self, lines: list[Line]
    ) -> list[Line]:
        """Sister of `sib_const_index_fold` for the `push` form. Fold
        a constant-index SIB-form push into the displacement.

        Pattern (with optional displacement):
            mov IDX, IMM    (or xor IDX, IDX for IMM=0)
            push <size?> [BASE + IDX*SCALE]
            push <size?> [BASE + IDX*SCALE + DISP]
            push <size?> [BASE + IDX*SCALE - DISP]

        Rewrite (drop the index load; fold IMM*SCALE into disp):
            push <size?> [BASE + (IMM*SCALE)]
            push <size?> [BASE + (IMM*SCALE + DISP)]

        Saves 2 bytes per match for `xor reg, reg` (drops 2-byte op).
        Saves 5 bytes per match for `mov reg, IMM` (drops 5-byte op).

        Conditions:
        - Two consecutive instr lines.
        - Line A: ``mov IDX, IMM`` (numeric literal) or
          ``xor IDX, IDX``.
        - Line B: ``push <size?> [BASE + IDX*SCALE [DISP]]``.
        - IDX dead after Line B.
        - BASE != IDX (otherwise BASE's value would change after
          dropping the const-load).
        """
        SCALE_VALUES = {1, 2, 4, 8}
        out = list(lines)
        i = 0
        while i + 1 < len(out):
            a = out[i]
            b = out[i + 1]
            if not (
                a.kind == "instr"
                and a.op in {"mov", "xor"}
                and b.kind == "instr" and b.op == "push"
            ):
                i += 1
                continue
            ap = _operands_split(a.operands)
            if ap is None:
                i += 1
                continue
            idx_reg = ap[0].strip().lower()
            if not self._is_general_register(idx_reg):
                i += 1
                continue
            # Determine constant value of idx_reg.
            if a.op == "xor":
                if ap[1].strip().lower() != idx_reg:
                    i += 1
                    continue
                imm_val = 0
            else:
                imm_str = ap[1].strip()
                try:
                    if imm_str.lower().startswith("0x"):
                        imm_val = int(imm_str, 16)
                    elif imm_str.lower().startswith("-0x"):
                        imm_val = -int(imm_str[1:], 16)
                    else:
                        imm_val = int(imm_str)
                except ValueError:
                    i += 1
                    continue
            # Parse Line B's operand. Push has no destination — the
            # operand is the source (memory).
            push_src = b.operands.strip()
            # Strip size prefix.
            size_prefix = ""
            for prefix in ("dword ", "word ", "byte ", "qword "):
                if push_src.lower().startswith(prefix):
                    size_prefix = push_src[:len(prefix)]
                    push_src = push_src[len(prefix):].lstrip()
                    break
            # Match SIB form.
            sib_re = re.match(
                r"^\[\s*([a-zA-Z]+)\s*\+\s*([a-zA-Z]+)\s*\*"
                r"\s*(\d+)"
                r"(?:\s*([+-])\s*(\d+))?"
                r"\s*\]$",
                push_src,
            )
            if sib_re is None:
                i += 1
                continue
            base_reg = sib_re.group(1).lower()
            sib_idx = sib_re.group(2).lower()
            try:
                scale = int(sib_re.group(3))
            except ValueError:
                i += 1
                continue
            if scale not in SCALE_VALUES:
                i += 1
                continue
            disp_sign = sib_re.group(4)
            disp_str = sib_re.group(5)
            disp = 0
            if disp_str is not None:
                try:
                    disp = int(disp_str)
                except ValueError:
                    i += 1
                    continue
                if disp_sign == "-":
                    disp = -disp
            if sib_idx != idx_reg:
                i += 1
                continue
            if base_reg == idx_reg:
                i += 1
                continue
            if not self._is_general_register(base_reg):
                i += 1
                continue
            # IDX dead after the push (push doesn't write IDX, so
            # IDX retains its constant value). If IDX is read after,
            # the rewrite is unsafe.
            if not self._reg_dead_after(out, i + 2, idx_reg):
                i += 1
                continue
            # Compute new displacement.
            new_disp = imm_val * scale + disp
            indent = self._extract_indent(b.raw)
            if new_disp == 0:
                new_src = f"{size_prefix}[{base_reg}]"
            elif new_disp > 0:
                new_src = (
                    f"{size_prefix}[{base_reg} + {new_disp}]"
                )
            else:
                new_src = (
                    f"{size_prefix}[{base_reg} - {-new_disp}]"
                )
            new_raw = f"{indent}push    {new_src}"
            new_line = Line(
                raw=new_raw, kind="instr", op="push",
                operands=new_src,
            )
            out = out[:i] + [new_line] + out[i + 2:]
            self.stats["push_const_index_fold"] = (
                self.stats.get("push_const_index_fold", 0) + 1
            )
            continue
        return out

    def _pass_byte_stores_to_dword(
        self, lines: list[Line]
    ) -> list[Line]:
        """Pack 4 consecutive ``mov byte [ebp +- N + i], imm8`` stores
        (i = 0, 1, 2, 3) into a single ``mov dword [ebp +- N], imm32``.

        Each `mov byte [m], imm8` is 4 bytes (`C6 ... imm8`). The
        equivalent `mov dword [m], imm32` is 7 bytes. Saves 9 bytes
        per 4-store group (16 → 7).

        Common in `char arr[N] = "string"` initialization where the
        codegen emits per-byte stores for each character.

        Conditions:
        - 4 consecutive instr lines.
        - All match ``mov byte [<base> ± N + j], imm8`` with the
          same base and consecutive offsets j = 0, 1, 2, 3 (or
          j = N, N+1, N+2, N+3 for any starting point).
        - All imm8 values are integer literals.

        Limited to ``[ebp ± N]`` addressing for safety. Other forms
        (`[reg + N]` register-base, label addresses, etc.) skipped.
        """
        out = list(lines)
        i = 0
        while i + 3 < len(out):
            chunk = out[i:i + 4]
            packed = self._try_pack_byte_stores(chunk)
            if packed is None:
                i += 1
                continue
            # Replace 4 lines with 1.
            out = out[:i] + [packed] + out[i + 4:]
            self.stats["byte_stores_to_dword"] = (
                self.stats.get("byte_stores_to_dword", 0) + 1
            )
            # Don't advance — try to pack the next 4 from i again
            # (in case the next group is also packable).
            continue
        return out

    @staticmethod
    def _try_pack_byte_stores(
        chunk: list[Line],
    ) -> Line | None:
        """If the 4 lines form a packable byte-store group, return
        the replacement dword-store Line. Otherwise None."""
        if len(chunk) != 4:
            return None
        offsets: list[int] = []
        values: list[int] = []
        sign = "+"
        base_disp = 0
        for idx, ln in enumerate(chunk):
            if ln.kind != "instr" or ln.op != "mov":
                return None
            parts = _operands_split(ln.operands)
            if parts is None:
                return None
            dest, src = parts
            dest_norm = dest.strip()
            # Must be `byte [ebp + N]` or `byte [ebp - N]`.
            m = re.fullmatch(
                r"byte\s+\[\s*ebp\s*([+-])\s*(\d+)\s*\]",
                dest_norm,
                re.IGNORECASE,
            )
            if not m:
                return None
            disp = int(m.group(2)) if m.group(1) == "+" else -int(m.group(2))
            offsets.append(disp)
            try:
                values.append(int(src.strip()) & 0xFF)
            except ValueError:
                return None
            if idx == 0:
                sign = m.group(1)
                base_disp = disp
        # Offsets must be consecutive: base, base+1, base+2, base+3.
        if offsets != [base_disp + j for j in range(4)]:
            return None
        # Pack into little-endian dword.
        packed = (
            values[0]
            | (values[1] << 8)
            | (values[2] << 16)
            | (values[3] << 24)
        )
        # Build the rewrite. Use a positive offset+sign that matches
        # the base. The base is at offsets[0], which is base_disp.
        indent = re.match(r"^[ \t]*", chunk[0].raw).group(0)
        if base_disp >= 0:
            addr = f"[ebp + {base_disp}]"
        else:
            addr = f"[ebp - {-base_disp}]"
        new_raw = f"{indent}mov     dword {addr}, {packed}"
        return Line(
            raw=new_raw,
            kind="instr",
            op="mov",
            operands=f"dword {addr}, {packed}",
        )

    def _pass_label_push_collapse(
        self, lines: list[Line]
    ) -> list[Line]:
        """Collapse ``mov REG, LABEL; push dword [REG]`` into
        ``push dword [LABEL]`` using x86's disp32-absolute push.

        Sister of ``label_load_collapse`` for the push case. Saves
        1 byte per match. Common in `printf("%d", glob);` arg-setup.

        Conditions:
        - Two consecutive instr lines.
        - Line A: ``mov REG, LABEL`` (LABEL is non-numeric, non-mem,
          non-register expression).
        - Line B: ``push dword [REG]``.
        - REG dead after Line B.
        """
        out = list(lines)
        i = 0
        while i + 1 < len(out):
            a = out[i]
            b = out[i + 1]
            if not (
                a.kind == "instr" and a.op == "mov"
                and b.kind == "instr" and b.op == "push"
            ):
                i += 1
                continue
            ap = _operands_split(a.operands)
            if ap is None:
                i += 1
                continue
            r1 = ap[0].strip().lower()
            label = ap[1].strip()
            if not self._is_general_register(r1):
                i += 1
                continue
            if (
                "[" in label
                or self._is_general_register(label.lower())
            ):
                i += 1
                continue
            try:
                int(label)
                i += 1
                continue
            except ValueError:
                pass
            # Push source: `dword [REG]`.
            b_op = b.operands.strip()
            for prefix in ("dword ", "word ", "byte ", "qword "):
                if b_op.lower().startswith(prefix):
                    b_op = b_op[len(prefix):].lstrip()
                    break
            mem_re = re.match(r"^\[\s*([a-zA-Z]+)\s*\]$", b_op)
            if mem_re is None or mem_re.group(1).lower() != r1:
                i += 1
                continue
            # REG dead after.
            if not self._reg_dead_after(out, i + 2, r1):
                i += 1
                continue
            indent = self._extract_indent(b.raw)
            new_src = f"dword [{label}]"
            new_raw = f"{indent}push    {new_src}"
            new_line = Line(
                raw=new_raw,
                kind="instr",
                op="push",
                operands=new_src,
            )
            out = out[:i] + [new_line] + out[i + 2:]
            self.stats["label_push_collapse"] = (
                self.stats.get("label_push_collapse", 0) + 1
            )
            continue
        return out

    def _pass_label_store_collapse(
        self, lines: list[Line]
    ) -> list[Line]:
        """Collapse ``mov REG, LABEL; mov <size> [REG], SRC`` into
        ``mov <size> [LABEL], SRC`` using x86's disp32-absolute store.

        Sister of ``label_load_collapse`` for the store case. Saves
        1 byte per match. Common in `glob = constant;` patterns where
        the codegen first loads the label address then stores into it.

        Conditions:
        - Two consecutive instr lines.
        - Line A: ``mov REG, LABEL`` (LABEL is non-numeric, non-mem,
          non-register expression).
        - Line B: ``mov <size> [REG], SRC`` (size in dword/word/byte;
          plain `[REG]` deref, no offset/SIB).
        - REG dead after Line B.
        """
        out = list(lines)
        i = 0
        while i + 1 < len(out):
            a = out[i]
            b = out[i + 1]
            if not (
                a.kind == "instr" and a.op == "mov"
                and b.kind == "instr" and b.op == "mov"
            ):
                i += 1
                continue
            ap = _operands_split(a.operands)
            bp = _operands_split(b.operands)
            if ap is None or bp is None:
                i += 1
                continue
            r1 = ap[0].strip().lower()
            label = ap[1].strip()
            b_dst = bp[0].strip()
            b_src = bp[1].strip()
            if not self._is_general_register(r1):
                i += 1
                continue
            if (
                "[" in label
                or self._is_general_register(label.lower())
            ):
                i += 1
                continue
            try:
                int(label)
                i += 1
                continue
            except ValueError:
                pass
            # Line B's dest must be `<size> [REG]` (with size keyword).
            m = re.match(
                r"^(dword|word|byte|qword)\s+\[\s*([a-zA-Z]+)\s*\]$",
                b_dst,
                re.IGNORECASE,
            )
            if m is None:
                i += 1
                continue
            size_kw = m.group(1).lower()
            mem_reg = m.group(2).lower()
            if mem_reg != r1:
                i += 1
                continue
            # SRC must not reference REG (else dropping the address
            # load breaks). E.g. `mov reg, LABEL; mov [reg], reg`
            # is `*LABEL = LABEL` which doesn't make sense to fold
            # without further analysis.
            if self._references_reg_family(b_src, r1):
                i += 1
                continue
            # REG dead after line B.
            if not self._reg_dead_after(out, i + 2, r1):
                i += 1
                continue
            indent = self._extract_indent(b.raw)
            new_dst = f"{size_kw} [{label}]"
            new_raw = f"{indent}mov     {new_dst}, {b_src}"
            new_line = Line(
                raw=new_raw,
                kind="instr",
                op="mov",
                operands=f"{new_dst}, {b_src}",
            )
            out = out[:i] + [new_line] + out[i + 2:]
            self.stats["label_store_collapse"] = (
                self.stats.get("label_store_collapse", 0) + 1
            )
            continue
        return out

    def _pass_lea_load_collapse(
        self, lines: list[Line]
    ) -> list[Line]:
        """Collapse ``lea REG, [ebp ± N]; mov REG2, [REG + M]`` (or
        ``[REG]``) into ``mov REG2, [ebp ± (N+M)]``.

        Saves 2-3 bytes per match by eliminating the LEA and folding
        the offset arithmetic into the load's addressing. Common in
        local struct member access where the codegen emits an
        explicit `lea` to compute the struct base, then `mov reg,
        [base + offset]` for each member.

        Conditions:
        - Two consecutive instr lines.
        - Line A: ``lea REG, [ebp ± N]`` (stack-relative base).
        - Line B: ``mov REG2, [REG]`` or ``mov REG2, [REG + M]``
          (plain or offset deref of REG).
        - REG dead after Line B (or REG == REG2).
        - Combined offset N+M is representable as a signed 32-bit
          displacement (always true for sane stack frames).
        """
        out = list(lines)
        i = 0
        while i + 1 < len(out):
            a = out[i]
            b = out[i + 1]
            if not (
                a.kind == "instr" and a.op == "lea"
                and b.kind == "instr" and b.op == "mov"
            ):
                i += 1
                continue
            ap = _operands_split(a.operands)
            bp = _operands_split(b.operands)
            if ap is None or bp is None:
                i += 1
                continue
            r1 = ap[0].strip().lower()
            lea_src = ap[1].strip()
            r2 = bp[0].strip().lower()
            b_src = bp[1].strip()
            if (
                not self._is_general_register(r1)
                or not self._is_general_register(r2)
            ):
                i += 1
                continue
            # LEA src must be `[ebp ± N]` (stack-relative).
            m_lea = re.match(
                r"^\[\s*ebp\s*([+-])\s*(\d+)\s*\]$",
                lea_src,
                re.IGNORECASE,
            )
            if m_lea is None:
                i += 1
                continue
            n_disp = int(m_lea.group(2))
            if m_lea.group(1) == "-":
                n_disp = -n_disp
            # Mov src: `[REG]` or `[REG + M]` or `[REG - M]`.
            b_src_stripped = b_src
            for prefix in ("dword ", "word ", "byte ", "qword "):
                if b_src_stripped.lower().startswith(prefix):
                    b_src_stripped = b_src_stripped[
                        len(prefix):
                    ].lstrip()
                    break
            m_plain = re.match(
                r"^\[\s*([a-zA-Z]+)\s*\]$", b_src_stripped
            )
            m_disp = re.match(
                r"^\[\s*([a-zA-Z]+)\s*([+-])\s*(\d+)\s*\]$",
                b_src_stripped,
            )
            if m_plain:
                base = m_plain.group(1).lower()
                m_disp_val = 0
            elif m_disp:
                base = m_disp.group(1).lower()
                m_disp_val = int(m_disp.group(3))
                if m_disp.group(2) == "-":
                    m_disp_val = -m_disp_val
            else:
                i += 1
                continue
            if base != r1:
                i += 1
                continue
            # Liveness: REG dead after, or REG == REG2.
            if r1 != r2 and not self._reg_dead_after(
                out, i + 2, r1
            ):
                i += 1
                continue
            # Build combined offset.
            combined = n_disp + m_disp_val
            sign = "+" if combined >= 0 else "-"
            new_src = f"[ebp {sign} {abs(combined)}]"
            # Preserve the size keyword from the original mov.
            size_kw = ""
            for prefix in ("dword ", "word ", "byte ", "qword "):
                if b_src.lower().startswith(prefix):
                    size_kw = prefix
                    break
            indent = self._extract_indent(b.raw)
            new_raw = f"{indent}mov     {r2}, {size_kw}{new_src}"
            new_line = Line(
                raw=new_raw,
                kind="instr",
                op="mov",
                operands=f"{r2}, {size_kw}{new_src}",
            )
            out = out[:i] + [new_line] + out[i + 2:]
            self.stats["lea_load_collapse"] = (
                self.stats.get("lea_load_collapse", 0) + 1
            )
            continue
        return out

    def _pass_lea_offset_fold(
        self, lines: list[Line]
    ) -> list[Line]:
        """Collapse ``lea REG, [ebp ± N]; add REG, M`` (or ``sub REG,
        M``) into ``lea REG, [ebp ± (N ± M)]``.

        Saves 3 bytes per match (drops the add/sub). LEA doesn't set
        flags, while add/sub does — so the rewrite is only safe when
        the add's flags are dead.

        Conditions:
        - Two consecutive instr lines.
        - Line A: ``lea REG, [ebp ± N]`` (stack-relative).
        - Line B: ``add REG, M`` or ``sub REG, M`` (numeric M).
        - Flags after Line B must be safe (dead before any reader).
        """
        out = list(lines)
        i = 0
        while i + 1 < len(out):
            a = out[i]
            b = out[i + 1]
            if not (
                a.kind == "instr" and a.op == "lea"
                and b.kind == "instr" and b.op in ("add", "sub")
            ):
                i += 1
                continue
            ap = _operands_split(a.operands)
            bp = _operands_split(b.operands)
            if ap is None or bp is None:
                i += 1
                continue
            r1 = ap[0].strip().lower()
            lea_src = ap[1].strip()
            b_dst = bp[0].strip().lower()
            if not self._is_general_register(r1) or b_dst != r1:
                i += 1
                continue
            try:
                imm = int(bp[1].strip())
            except ValueError:
                i += 1
                continue
            # LEA src must be `[ebp ± N]`.
            m = re.match(
                r"^\[\s*ebp\s*([+-])\s*(\d+)\s*\]$",
                lea_src,
                re.IGNORECASE,
            )
            if m is None:
                i += 1
                continue
            n_disp = int(m.group(2))
            if m.group(1) == "-":
                n_disp = -n_disp
            # Apply the add/sub.
            if b.op == "add":
                combined = n_disp + imm
            else:
                combined = n_disp - imm
            # Flags must be dead after line B (LEA doesn't set flags
            # but add/sub does; if flags were used by a downstream
            # Jcc, dropping the add changes behavior).
            if not self._flags_safe_after(out, i + 2):
                i += 1
                continue
            sign = "+" if combined >= 0 else "-"
            new_src = f"[ebp {sign} {abs(combined)}]"
            indent = self._extract_indent(a.raw)
            new_raw = f"{indent}lea     {r1}, {new_src}"
            new_line = Line(
                raw=new_raw,
                kind="instr",
                op="lea",
                operands=f"{r1}, {new_src}",
            )
            out = out[:i] + [new_line] + out[i + 2:]
            self.stats["lea_offset_fold"] = (
                self.stats.get("lea_offset_fold", 0) + 1
            )
            continue
        return out

    def _pass_lea_forward_to_reg(
        self, lines: list[Line]
    ) -> list[Line]:
        """Collapse ``lea REG1, BASE; mov REG2, REG1`` into
        ``lea REG2, BASE`` when REG1 is dead after.

        Saves 2 bytes per match (drops the register copy). Common
        after the codegen emits an address-into-EAX then transfers
        to a different register for further use.

        Conditions:
        - Two consecutive instr lines.
        - Line A: ``lea REG1, BASE``.
        - Line B: ``mov REG2, REG1`` (register copy).
        - REG1 != REG2.
        - REG1 dead after line B.
        """
        out = list(lines)
        i = 0
        while i + 1 < len(out):
            a = out[i]
            b = out[i + 1]
            if not (
                a.kind == "instr" and a.op == "lea"
                and b.kind == "instr" and b.op == "mov"
            ):
                i += 1
                continue
            ap = _operands_split(a.operands)
            bp = _operands_split(b.operands)
            if ap is None or bp is None:
                i += 1
                continue
            r1 = ap[0].strip().lower()
            base = ap[1].strip()
            b_dst = bp[0].strip().lower()
            b_src = bp[1].strip().lower()
            if (
                not self._is_general_register(r1)
                or not self._is_general_register(b_dst)
                or b_src != r1
                or b_dst == r1
            ):
                i += 1
                continue
            if not self._reg_dead_after(out, i + 2, r1):
                i += 1
                continue
            indent = self._extract_indent(a.raw)
            new_raw = f"{indent}lea     {b_dst}, {base}"
            new_line = Line(
                raw=new_raw,
                kind="instr",
                op="lea",
                operands=f"{b_dst}, {base}",
            )
            out = out[:i] + [new_line] + out[i + 2:]
            self.stats["lea_forward_to_reg"] = (
                self.stats.get("lea_forward_to_reg", 0) + 1
            )
            continue
        return out

    def _pass_lea_store_collapse(
        self, lines: list[Line]
    ) -> list[Line]:
        """Collapse ``lea REG, [ebp ± N]; mov <size> [REG + M], SRC``
        (or ``[REG]``) into ``mov <size> [ebp ± (N+M)], SRC``.

        Saves 3 bytes per match. Common in local struct member
        assignment where the codegen emits an explicit `lea` to
        compute the struct base, then `mov [base + offset], value`
        for each member.

        Conditions:
        - Two consecutive instr lines.
        - Line A: ``lea REG, [ebp ± N]`` (stack-relative base).
        - Line B: ``mov <size> [REG]`` or ``mov <size> [REG + M]``
          with SRC.
        - REG dead after line B.
        - SRC must not reference REG.
        """
        out = list(lines)
        i = 0
        while i + 1 < len(out):
            a = out[i]
            b = out[i + 1]
            if not (
                a.kind == "instr" and a.op == "lea"
                and b.kind == "instr" and b.op == "mov"
            ):
                i += 1
                continue
            ap = _operands_split(a.operands)
            bp = _operands_split(b.operands)
            if ap is None or bp is None:
                i += 1
                continue
            r1 = ap[0].strip().lower()
            lea_src = ap[1].strip()
            b_dst = bp[0].strip()
            b_src = bp[1].strip()
            if not self._is_general_register(r1):
                i += 1
                continue
            # LEA src must be `[ebp ± N]`.
            m_lea = re.match(
                r"^\[\s*ebp\s*([+-])\s*(\d+)\s*\]$",
                lea_src,
                re.IGNORECASE,
            )
            if m_lea is None:
                i += 1
                continue
            n_disp = int(m_lea.group(2))
            if m_lea.group(1) == "-":
                n_disp = -n_disp
            # Mov dest must be `<size> [REG]` or `<size> [REG ± M]`.
            m_dst_plain = re.match(
                r"^(dword|word|byte|qword)\s+\[\s*([a-zA-Z]+)\s*\]$",
                b_dst,
                re.IGNORECASE,
            )
            m_dst_disp = re.match(
                r"^(dword|word|byte|qword)\s+\[\s*([a-zA-Z]+)"
                r"\s*([+-])\s*(\d+)\s*\]$",
                b_dst,
                re.IGNORECASE,
            )
            if m_dst_plain:
                size_kw = m_dst_plain.group(1).lower()
                base_reg = m_dst_plain.group(2).lower()
                m_disp_val = 0
            elif m_dst_disp:
                size_kw = m_dst_disp.group(1).lower()
                base_reg = m_dst_disp.group(2).lower()
                m_disp_val = int(m_dst_disp.group(4))
                if m_dst_disp.group(3) == "-":
                    m_disp_val = -m_disp_val
            else:
                i += 1
                continue
            if base_reg != r1:
                i += 1
                continue
            # SRC must not reference REG.
            if self._references_reg_family(b_src, r1):
                i += 1
                continue
            # REG dead after line B.
            if not self._reg_dead_after(out, i + 2, r1):
                i += 1
                continue
            combined = n_disp + m_disp_val
            sign = "+" if combined >= 0 else "-"
            new_dst = f"{size_kw} [ebp {sign} {abs(combined)}]"
            indent = self._extract_indent(b.raw)
            new_raw = f"{indent}mov     {new_dst}, {b_src}"
            new_line = Line(
                raw=new_raw,
                kind="instr",
                op="mov",
                operands=f"{new_dst}, {b_src}",
            )
            out = out[:i] + [new_line] + out[i + 2:]
            self.stats["lea_store_collapse"] = (
                self.stats.get("lea_store_collapse", 0) + 1
            )
            continue
        return out

    def _pass_dead_stack_store(
        self, lines: list[Line]
    ) -> list[Line]:
        """Drop ``mov <size> [ebp ± N], V1`` when a later
        ``mov <size> [ebp ± N], V2`` overwrites it before any read.

        Saves up to 7 bytes per match (drops a dword-imm store).

        Common after struct/array init followed by member assignment:
        ``struct point p = {1, 2, 3}; p.x = 100;`` emits two stores
        to [ebp - 12] back-to-back; the first is dead.

        Conservative scan:
        - Read of [ebp ± N] in any instruction's operands → store is
          alive, bail.
        - Call → bail (callee MIGHT have access to &x via prior
          address-take + global; can't tell without escape analysis).
        - LEA producing an address that might alias [ebp ± N] → bail
          (the lea'd reg could be used for indirect access later).
          Conservative: any LEA that references [ebp ± N] bails.
          Other LEAs (label addresses) don't alias stack slots.
        - Conditional or unconditional jump → bail (control-flow
          boundary).
        - Label → bail (predecessor may differ).
        - Memory write through a register pointer → bail (might
          alias).
        - Reaching another `mov <size> [ebp ± N], V2` (same offset)
          → fire: drop the first store.
        """
        out: list[Line] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.kind != "instr":
                out.append(line)
                i += 1
                continue
            n_disp = self._stack_store_offset(line)
            if n_disp is None:
                out.append(line)
                i += 1
                continue
            # Found a stack-imm-or-reg store. Scan forward.
            if self._dead_stack_store_check(lines, i, n_disp):
                self.stats["dead_stack_store"] = (
                    self.stats.get("dead_stack_store", 0) + 1
                )
                i += 1  # Drop the line.
                continue
            out.append(line)
            i += 1
        return out

    @staticmethod
    def _stack_store_offset(line: Line) -> int | None:
        """If `line` is `mov <size> [ebp ± N], SRC`, return N.
        Otherwise None."""
        if line.kind != "instr" or line.op != "mov":
            return None
        parts = _operands_split(line.operands)
        if parts is None:
            return None
        dest = parts[0].strip()
        m = re.match(
            r"^(dword|word|byte|qword)\s+\[\s*ebp\s*([+-])\s*(\d+)\s*\]$",
            dest,
            re.IGNORECASE,
        )
        if m is None:
            return None
        n = int(m.group(3))
        if m.group(2) == "-":
            n = -n
        return n

    def _dead_stack_store_check(
        self, lines: list[Line], start_idx: int, target_offset: int,
    ) -> bool:
        """Return True if the store at lines[start_idx] is dead —
        an overwrite to the same offset is reached before any read."""
        target_re = re.compile(
            rf"\[\s*ebp\s*[+-]\s*\d+\s*\]",
            re.IGNORECASE,
        )
        # Build the exact text to look for in operands when scanning
        # for reads. Both `[ebp + N]` and `[ebp - N]` (with original
        # sign) are valid — but normalized through the offset.
        for j in range(start_idx + 1, min(len(lines), start_idx + 30)):
            ln = lines[j]
            if ln.kind == "label":
                # Control-flow boundary.
                return False
            if ln.kind != "instr":
                continue
            op = ln.op
            ops = ln.operands
            # Calls bail.
            if op == "call":
                return False
            # Jumps bail (control flow).
            if op in {"jmp", "ret", "iret", "iretd", "retf",
                      "retn", "leave"} or op.startswith("j"):
                return False
            # Find any [ebp ± M] reference in this line.
            for m in target_re.finditer(ops):
                ref = m.group(0)
                rm = re.match(
                    r"^\[\s*ebp\s*([+-])\s*(\d+)\s*\]$",
                    ref,
                    re.IGNORECASE,
                )
                if not rm:
                    continue
                ref_off = int(rm.group(2))
                if rm.group(1) == "-":
                    ref_off = -ref_off
                if ref_off != target_offset:
                    continue
                # Same offset — does the line WRITE or READ it?
                if op == "mov":
                    parts = _operands_split(ops)
                    if parts is None:
                        return False
                    dest = parts[0].strip()
                    src = parts[1].strip()
                    # Dest matches our target offset → overwrite.
                    if (
                        ref in dest
                        and self._stack_store_offset(ln)
                        == target_offset
                    ):
                        return True
                    # Src refs target offset → read.
                    if ref in src:
                        return False
                    # Otherwise (other ops at this offset), conservative.
                    return False
                # Any other op (add/sub/etc.) reading or writing the
                # offset is conservative — bail.
                return False
            # Memory write through register (e.g. mov [eax], X) might
            # alias a stack slot if EAX was lea'd from one earlier.
            # Conservative: bail on any indirect mem write.
            if op == "mov":
                parts = _operands_split(ops)
                if parts is not None:
                    dest = parts[0].strip()
                    if "[" in dest and "ebp" not in dest.lower():
                        # Indirect memory write.
                        return False
            # LEA referencing target offset → bail.
            if op == "lea" and target_re.search(ops):
                for m in target_re.finditer(ops):
                    ref = m.group(0)
                    rm = re.match(
                        r"^\[\s*ebp\s*([+-])\s*(\d+)\s*\]$",
                        ref,
                        re.IGNORECASE,
                    )
                    if rm:
                        ref_off = int(rm.group(2))
                        if rm.group(1) == "-":
                            ref_off = -ref_off
                        if ref_off == target_offset:
                            return False
            # Otherwise, continue scanning.
        return False

    def _pass_value_forward_to_reg(
        self, lines: list[Line]
    ) -> list[Line]:
        """Collapse ``mov REG1, SRC; mov REG2, REG1`` into
        ``mov REG2, SRC`` when REG1 is dead after.

        Saves 2 bytes per match (drops the `mov reg2, reg1` transfer;
        the new `mov reg2, SRC` is the same length as the original
        `mov reg1, SRC`).

        Common after label_offset_fold leaves
        ``mov eax, _label + N; mov ebx, eax`` — the first instruction's
        value is being forwarded through EAX to EBX, but EAX is dead
        after.

        Conditions:
        - Two consecutive instr lines.
        - Line A: ``mov REG1, SRC`` where SRC is not a memory operand
          (immediates, labels, label-arithmetic, register sources).
        - Line B: ``mov REG2, REG1`` (register copy).
        - REG1 != REG2 (else line B is a self-mov, handled by
          self_mov_elimination).
        - REG1 dead after line B.
        """
        out = list(lines)
        i = 0
        while i + 1 < len(out):
            a = out[i]
            b = out[i + 1]
            if not (
                a.kind == "instr" and a.op == "mov"
                and b.kind == "instr" and b.op == "mov"
            ):
                i += 1
                continue
            ap = _operands_split(a.operands)
            bp = _operands_split(b.operands)
            if ap is None or bp is None:
                i += 1
                continue
            r1 = ap[0].strip().lower()
            src = ap[1].strip()
            b_dst = bp[0].strip().lower()
            b_src = bp[1].strip().lower()
            if (
                not self._is_general_register(r1)
                or not self._is_general_register(b_dst)
            ):
                i += 1
                continue
            # Line B must be a register copy from r1.
            if b_src != r1 or b_dst == r1:
                i += 1
                continue
            # SRC must NOT be a memory operand. Memory sources work
            # in principle but x86 doesn't have a `mov [m], [m]` form;
            # `mov ebx, [m]` is the same length as `mov eax, [m]`, so
            # no savings issue, BUT the pass would collide with
            # store_load_collapse and others. Keep simple.
            if "[" in src:
                i += 1
                continue
            # SRC must not be the target register either (would be
            # forwarding `mov r2, r2` which is a self-mov).
            if src.lower() == b_dst:
                i += 1
                continue
            # REG1 dead after line B.
            if not self._reg_dead_after(out, i + 2, r1):
                i += 1
                continue
            # Rewrite: replace lines A and B with `mov b_dst, src`.
            indent = self._extract_indent(a.raw)
            new_raw = f"{indent}mov     {b_dst}, {src}"
            new_line = Line(
                raw=new_raw,
                kind="instr",
                op="mov",
                operands=f"{b_dst}, {src}",
            )
            out = out[:i] + [new_line] + out[i + 2:]
            self.stats["value_forward_to_reg"] = (
                self.stats.get("value_forward_to_reg", 0) + 1
            )
            continue
        return out

    def _pass_self_mov_elimination(
        self, lines: list[Line]
    ) -> list[Line]:
        """Drop `mov REG, REG` where dst and src are the same register
        — these are no-ops on x86 (no flag changes, no value change).

        Saves 2 bytes per match. Common after right_operand_retarget
        leaves a stale `mov ecx, ecx` (or other self-mov) when the
        retargeted chain ends with the same register the codegen would
        have transferred to.

        Conservative: only matches plain register-to-register movs
        where the operand strings are identical (after lowercasing).
        Doesn't drop `mov al, al` / `mov ah, ah` etc. (sub-register
        movs are the same shape but might be uncommon — keep simple).
        """
        out: list[Line] = []
        for line in lines:
            if (
                line.kind == "instr"
                and line.op == "mov"
            ):
                parts = _operands_split(line.operands)
                if parts is not None:
                    dest, src = parts
                    dest_low = dest.strip().lower()
                    src_low = src.strip().lower()
                    if (
                        dest_low == src_low
                        and self._is_general_register(dest_low)
                    ):
                        self.stats["self_mov_elimination"] = (
                            self.stats.get(
                                "self_mov_elimination", 0
                            ) + 1
                        )
                        continue
            out.append(line)
        return out

    def _pass_jmp_to_next_label(self, lines: list[Line]) -> list[Line]:
        """P-jmp-to-next: `jmp X` immediately (modulo blanks/comments)
        followed by `X:` becomes a no-op — drop the `jmp`."""
        out: list[Line] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            target = _jmp_target(line)
            if target is None:
                out.append(line)
                i += 1
                continue
            # Look ahead for the next non-blank/non-comment line.
            j = i + 1
            while j < len(lines) and lines[j].kind in ("blank", "comment"):
                j += 1
            if j < len(lines) and lines[j].kind == "label" and lines[j].label == target:
                # Drop the jmp; keep blanks/comments (they may belong to
                # the label that follows).
                self.stats["jmp_to_next_label"] = (
                    self.stats.get("jmp_to_next_label", 0) + 1
                )
                i += 1
                continue
            out.append(line)
            i += 1
        return out


def optimize(asm_text: str) -> str:
    """Convenience wrapper: run peephole and return the optimized asm."""
    return PeepholeOptimizer().optimize(asm_text)
