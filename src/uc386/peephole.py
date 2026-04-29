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

Patterns to add (see PEEPHOLE_PLAN.md for details): redundant
mov-to-reg, tail calls, jump threading, multi-instruction right-
operand retargeting.
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
    operand = line.operands.strip()
    # Indirect jumps: `jmp eax`, `jmp [eax]`, `jmp [_table + eax*4]`.
    # We don't optimize those.
    if not operand or operand[0] in "[":
        return None
    # Register operand: `jmp eax`. Single bare identifier that's a 32-bit
    # general-purpose register.
    if operand.lower() in {"eax", "ebx", "ecx", "edx", "esi", "edi",
                           "ebp", "esp"}:
        return None
    # Treat the rest as a label name. NASM allows `jmp _foo` and
    # `jmp .local`. No spaces in labels.
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
        """Replace `mov eax, IMM_OR_LABEL; push eax` with `push X`
        when the next instruction overwrites EAX.

        The codegen emits `mov eax, X; push eax` for every cdecl arg
        push. NASM's `push imm8` / `push imm32` instructions skip the
        EAX round-trip entirely. The next instruction must overwrite
        EAX for the rewrite to be safe (otherwise EAX's value-after-
        push is observable).

        Conditions:
        - First instr: `mov eax, src` where src has no `[` (constant).
        - Second instr: `push eax`.
        - Third instr (witness): writes EAX (mov eax, * / xor eax, eax
          / call * / etc.).
        """
        def overwrites_eax(line: Line) -> bool:
            if line.kind != "instr":
                return False
            # `call X` clobbers EAX (caller-saved in cdecl).
            if line.op == "call":
                return True
            # `xor eax, eax` (and other xor reg, reg) clobbers.
            if line.op == "xor":
                parts = _operands_split(line.operands)
                if parts and parts[0].lower() == "eax":
                    return True
                return False
            # `mov eax, anything` overwrites.
            if line.op == "mov":
                parts = _operands_split(line.operands)
                if parts and parts[0].lower() == "eax":
                    return True
                return False
            # `lea eax, ...` overwrites.
            if line.op == "lea":
                parts = _operands_split(line.operands)
                if parts and parts[0].lower() == "eax":
                    return True
                return False
            # `pop eax` overwrites.
            if line.op == "pop":
                if line.operands.strip().lower() == "eax":
                    return True
                return False
            return False

        out: list[Line] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if (line.kind == "instr" and line.op == "mov"
                    and _is_imm_or_label_to_eax(line)):
                instrs = self._next_n_instrs(lines, i + 1, 2)
                if instrs is not None:
                    [(b_idx, b_line), (c_idx, c_line)] = instrs
                    if (b_line.op == "push"
                            and b_line.operands.strip().lower() == "eax"
                            and overwrites_eax(c_line)):
                        # Build `push <src>`. Preserve indentation.
                        parts = _operands_split(line.operands)
                        assert parts is not None
                        _, src = parts
                        leading_ws = re.match(r"^\s*", line.raw).group(0)
                        new_raw = f"{leading_ws}push    {src}"
                        new_line = Line(raw=new_raw, kind="instr",
                                        op="push", operands=src)
                        out.append(new_line)
                        self.stats["push_immediate"] = (
                            self.stats.get("push_immediate", 0) + 1
                        )
                        i = b_idx + 1  # Skip past the push; keep witness
                        continue
            out.append(line)
            i += 1
        return out

    def _pass_imm_store_collapse(self, lines: list[Line]) -> list[Line]:
        """Collapse `mov eax, IMM; mov [addr], eax; mov eax, X` to
        `mov dword [addr], IMM; mov eax, X`.

        The third instruction is the witness that EAX was dead after
        the store — its rewrite of EAX makes the load-via-EAX
        unnecessary. NASM accepts `mov dword [addr], IMM` directly.

        Conditions:
        - Line A: `mov eax, src` where src is a constant expression
          (no `[` — i.e., not a memory load, not a register-to-EAX
          where the register might be live).
        - Line B: `mov [addr], eax` — a 4-byte store from EAX.
        - Line C: `mov eax, anything` — confirms EAX is dead.

        We don't fold without the witness — without line C we can't
        be sure EAX wasn't live for a comparison/return/etc.
        """
        out: list[Line] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if (line.kind == "instr" and line.op == "mov"
                    and self._is_imm_to_eax(line)):
                instrs = self._next_n_instrs(lines, i + 1, 2)
                if instrs is not None and self._matches_imm_store(instrs, line):
                    [(b_idx, b_line), (c_idx, _c)] = instrs
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
                    out.append(new_line)
                    self.stats["imm_store_collapse"] = (
                        self.stats.get("imm_store_collapse", 0) + 1
                    )
                    # Skip past line A (i) and line B (b_idx). Keep
                    # line C in place (the witness mov-eax) since the
                    # caller might want it.
                    i = b_idx + 1
                    continue
            out.append(line)
            i += 1
        return out

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
        """Verify line B (`mov [addr], eax`) and line C (`mov eax, X`)."""
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
        # Line C must be `mov eax, anything` — the EAX overwrite witness.
        if c_line.op != "mov":
            return False
        parts_c = _operands_split(c_line.operands)
        if parts_c is None:
            return False
        c_dest, _c_src = parts_c
        if c_dest.lower() != "eax":
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
                instrs = self._next_n_instrs(lines, i + 1, 2)
                if instrs is not None:
                    [(b_idx, b_line), (c_idx, c_line)] = instrs
                    if (b_line.op in self._IMM_BINOP_OPS
                            and self._is_eax_ecx_binop(b_line)
                            and self._ecx_dead_after(c_line)):
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
