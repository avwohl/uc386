# WIP — resume notes for the new machine

Phase 4 slices 0–20 are done. 211 tests passing.
Slice 21+ is the next logical work — see "Where the codegen stands" below.

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

Implemented in slice 20 (just landed):
- **Struct by-value params.** Caller reserves `sizeof(struct)` rounded
  to 4 via `sub esp, N` and copies into the slot; callee accesses via
  the accumulated param offset. Struct return still raises pending its
  own slice (the ABI for the hidden first arg + caller-side temp slot
  is the open design question).

Deliberately not yet implemented — next slices in roughly this order:
- **Unions.** Same layout as structs but all members at offset 0 and
  total size = max(member sizes). Member access reuses
  `_member_address` once the registry distinguishes union from struct.
- **Struct return by value.** Hidden first arg pointing at a
  caller-allocated buffer. The trickiest piece is the buffer's
  lifetime — likely a per-call temp area added to the caller's frame.
- **Designated/nested initializers.** `int arr[3] = {[1] = 5}` and
  `int m[2][3] = {{...}, {...}}` both raise. Multidim arrays would
  also need ArrayType-of-ArrayType slot support.
- **Floating point.** `float` / `double` slot codegen via x87 or SSE.
  Big topic — likely a phase of its own.

Suggested first move next session: read `CLAUDE.md`. **Struct copy +
struct init** is the natural follow-on to slice 17, and **switch/case**
is the largest remaining language feature. Both are bite-sized slices
with the infrastructure already in place.
