# WIP — resume notes for the new machine

Phase 4 slices 0–9 are done. 99 tests passing.
Slice 10+ is the next logical work — see "Where the codegen stands" below.

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

Implemented in slice 9 (just landed):
- **Arrays (uninitialized).** `int arr[N]` allocates `N*sizeof(elem)`
  on the frame; array names decay to addresses; `arr[i]` reads/writes
  via base+i*sizeof(elem); `&arr[i]` produces the element address
  without dereferencing.

Deliberately not yet implemented — next slices in roughly this order:
- **Array initialization.** `int arr[N] = {a, b, c};` and string-init
  `char s[] = "hi";` need `InitializerList` lowering — store each
  value to its slot, optionally zero the tail.
- **`char` / `short` codegen.** Size-aware load (`movsx` / `movzx`
  for sub-word) and store (`mov byte`/`mov word`). Today every slot
  is 4 bytes regardless of declared type. `char *p` arithmetic works,
  but `*p` still emits a 4-byte load — that's a latent bug this slice
  fixes. Also makes `char arr[N]` strings genuinely useful.
- **Globals.** Same lowering model as locals but in `.data`/`.bss`
  with named labels instead of `[ebp + disp]`.
- **Casts.** `(int)x`, `(char *)p`, etc.
- **Compound assignment to non-Identifier lvalues.** `arr[i] += 5` and
  `*p += 5` currently raise. Needs lowering that computes the address
  once into a temp slot rather than re-evaluating the lvalue twice.
- **Function pointers / indirect calls.** Currently `_call` rejects
  any non-Identifier callee.
- **`sizeof` operator.** `_size_of` already exists internally; just
  need to wire `SizeofExpr` and `SizeofType` into expression eval.

Suggested first move next session: read `CLAUDE.md`. Sub-word codegen
is probably the highest-leverage next slice — it closes the
`*char_ptr` correctness gap and unlocks meaningful char/short locals
and `char arr[]` strings.
