# WIP — resume notes for the new machine

Phase 4 slices 0–8 are done. 87 tests passing on the new machine.
Slice 9+ is the next logical work — see "Where the codegen stands" below.

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

Implemented in slice 8 (just landed):
- **Pointer arithmetic with size scaling.** `p + n`, `n + p`, `p - n`,
  `p - q`, `++p`, `--p`, `p += n` etc. all use `sizeof(*p)`. Backed
  by `_type_of` / `_size_of` and the per-slot type map on `_FuncCtx`.

Deliberately not yet implemented — next slices in roughly this order:
- **Arrays.** `int arr[N]` (frame allocation `N*sizeof(int)`),
  `arr[i]` indexing, decay to pointer at use sites. Indexing should
  fall out cheaply now that pointer arithmetic works (`arr[i]` is
  `*(arr+i)`).
- **`char` / `short` codegen.** Size-aware load (`movsx` / `movzx`
  for sub-word) and store (`mov byte`/`mov word`). Today every slot
  is 4 bytes regardless of declared type. `char *p` arithmetic works,
  but `*p` still emits a 4-byte load — that's a latent bug this slice
  fixes.
- **Globals.** Same lowering model as locals but in `.data`/`.bss`
  with named labels instead of `[ebp + disp]`.
- **Casts.** `(int)x`, `(char *)p`, etc.
- **Function pointers / indirect calls.** Currently `_call` rejects
  any non-Identifier callee.

Suggested first move next session: read `CLAUDE.md`, then either start
on arrays (relatively cheap with pointer arithmetic landed) or sub-word
codegen (closes the `*char_ptr` correctness gap).
