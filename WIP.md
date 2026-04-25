# WIP — resume notes for the new machine

Phase 4 slices 0–31 are done. 257 tests passing.
Phase 4 is feature-complete for typical C; the remaining gaps
are floats (Phase 5 territory) and callee-side va_list / va_arg.

## Bootstrap on the new machine

1. **Clone the sibling repo** so `../uc_core` exists. Without it imports
   fail.
   ```
   git clone git@github.com:avwohl/uc_core.git ../uc_core
   ```

2. **Recreate the venv.** System Python on macOS is 3.9 and *won't*
   work (uc_core uses `dataclass(kw_only=True)`, 3.10+).
   ```
   python3.12 -m venv .venv
   .venv/bin/pip install pytest -e ../uc_core -e .
   ```

3. **Confirm green.**
   ```
   .venv/bin/pytest tests/
   ```
   Should report 74 passed.

4. **NASM (optional).** Tests verify text structure only, so NASM is
   not required to run the suite. If you want to actually assemble
   the output: `brew install nasm`.

## Sync Claude memory (optional but recommended)

The autonomous-operation rules live outside the repo at
`~/.claude/projects/-Users-wohl-src-uc386/memory/`. They are *not*
checked in. Without them the next session asks for permission before
each commit/push.

Files to sync:
- `MEMORY.md` — index
- `feedback_autonomous.md` — "go on without bothering the human"
- `feedback_session_end.md` — update CLAUDE.md, commit, push at every boundary
- `project_python_env.md` — venv / Python 3.12 notes

Either `rsync` that directory, or ask Claude on the new machine to
recreate the rules from this WIP file.

## Where the codegen stands

See `CLAUDE.md` for the full session log and the codegen contract.
Tests double as the spec.

Implemented (Phase 4):
- `int` and pointer locals + cdecl parameters.
- All integer arithmetic, comparisons, bitwise, shifts.
- Assignment + compound assignment + `++`/`--` (Identifier lvalues;
  `*p = rhs` also works).
- `&&` / `||` short-circuit and `?:`.
- Control flow: `if`/`else`, `while`, `do`/`while`, `for`, `break`,
  `continue`.
- Direct function calls; bodyless declarations emit `extern _name`.
- String literals → `.data` section, interned per translation unit.

Implemented in slice 31 (just landed):
- **Bit-fields.** Adjacent same-type bit-fields pack into a 32-bit
  storage unit; reads use shr+and (+sign-extend), writes RMW the
  storage. Cross-unit splitting starts a new unit cleanly.

Deliberately not yet implemented — what's left of Phase 4:
- **Floating point.** `float` / `double` slot codegen via x87 or SSE.
  Big topic — likely a phase of its own.
- **Variadic function definitions** (callee-side va_list / va_arg /
  va_start). Variadic *call sites* already work.
- **Static / register storage classes** beyond what cdecl already
  gives.
- **Typedef.** No special parser support yet — uc_core may already
  resolve typedefs into their underlying types, in which case there's
  nothing to do here.

Suggested first move next session: read `CLAUDE.md`. The
remaining gaps are mostly niche or large topics in their own right.
Multidim arrays is the most "missing common feature" of the bunch.
