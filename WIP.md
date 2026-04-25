# WIP — resume notes for the new machine

Snapshot at HEAD `75dbdfd`. No uncommitted changes. 74 tests passing.
Phase 4 slices 0–7 are done; slice 8+ is the next logical work.

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

Deliberately not yet implemented — these are the next slices:
- **Pointer arithmetic with size scaling.** `p + 1` should advance by
  `sizeof(*p)`, but expression-node type info isn't plumbed through
  codegen yet. This is the next blocker; everything below depends on
  the same plumbing.
- **Arrays.** `int arr[N]` (frame allocation `N*sizeof(int)`),
  `arr[i]` indexing, decay to pointer at use sites.
- **`char` / `short` codegen.** Size-aware load (`movsx` / `movzx`
  for sub-word) and store (`mov byte`/`mov word`). Today every slot
  is 4 bytes regardless of declared type.
- **Globals.** Same lowering model as locals but in `.data`/`.bss`
  with named labels instead of `[ebp + disp]`.
- **Casts.** `(int)x`, `(char *)p`, etc.
- **Function pointers / indirect calls.** Currently `_call` rejects
  any non-Identifier callee.

Suggested first move on the new machine: read `CLAUDE.md`, glance at
`tests/test_smoke.py` to see what the contract feels like, then ask
Claude to plumb expression-node type info through codegen as the
prerequisite for pointer arithmetic + arrays + sub-word types.
