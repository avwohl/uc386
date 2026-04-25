# WIP — resume notes for the new machine

Phase 4 slices 0–14 are done. 161 tests passing.
Slice 15+ is the next logical work — see "Where the codegen stands" below.

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

Implemented in slice 14 (just landed):
- **Casts.** `(T)expr` — pointer casts and within-int-family casts are
  no-ops; narrowing to char/short emits `movsx`/`movzx` of `al`/`ax`.
  `_type_of(Cast)` returns the target so subsequent pointer arithmetic
  scales by the right pointee size.

Deliberately not yet implemented — next slices in roughly this order:
- **Compound assignment to non-Identifier lvalues.** `arr[i] += 5` and
  `*p += 5` currently raise. Needs lowering that computes the address
  once into a temp slot rather than re-evaluating the lvalue twice.
- **Function pointers / indirect calls.** Currently `_call` rejects
  any non-Identifier callee.
- **CharLiteral.** `char c = 'a'` would currently raise — `'a'` is a
  CharLiteral (value=97) but `_eval_expr_to_eax` doesn't handle it
  yet. Trivially adding it would unlock more idiomatic char tests.
- **Designated/nested initializers.** `int arr[3] = {[1] = 5}` and
  `int m[2][3] = {{...}, {...}}` both raise. Multidim arrays would
  also need ArrayType-of-ArrayType slot support.
- **Floating point.** `float` / `double` slot codegen via x87 or SSE.
  Big topic — likely a phase of its own.

Suggested first move next session: read `CLAUDE.md`. **Function
pointers / indirect calls** is the smallest remaining slice — one
branch in `_call` to evaluate a non-Identifier callee into a register
and emit `call eax`. Then **CharLiteral** support (one expression
handler, makes `'a'` work) and **compound-assign to lvalues** would
round out Phase 4.
