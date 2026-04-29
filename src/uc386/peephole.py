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
