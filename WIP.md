# WIP — resume notes for the new machine

Phase 4 slices 0–32 done. Phase 5 (floats) slices 1–9 done.
306 tests passing.

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

Implemented in Phase 4 slice 32 (just landed):
- **Function-static locals.** `static int x = 0;` inside a function
  now lives in `.data`/`.bss` under a mangled label (`_func__x`)
  and persists across calls. Read/write transparently through
  `ctx.local_static_labels`.

Deliberately not yet implemented:
- **Variadic function definitions** (callee-side va_list / va_arg /
  va_start). Variadic *call sites* already work; va_arg etc. would
  need stdarg.h preprocessor support or builtin recognition.
- **`register` storage class.** It's just a hint; we ignore it. No
  correctness gap.

## Phase 5 design questions (floats — settled, kept here for reference)

Floats need decisions before coding:

1. **x87 FPU vs SSE.** The 386/MS-DOS target leans x87 (8-deep stack
   register file). SSE only landed in P3-era CPUs and is irrelevant
   for the DOS runtime. Going with x87 means `fld` / `fstp` /
   `faddp` etc., and a stack discipline that's quite different from
   the eax-everywhere model.

2. **Single-pass vs split flow.** Right now `_eval_expr_to_eax`
   universally produces a 32-bit value in EAX. Floats break that —
   their value lives on `st(0)` (or in memory). The cleanest fix is
   a parallel `_eval_expr_to_st0` for float-typed expressions, with
   `_type_of` driving the dispatch. Mixing int and float in one
   expression then has to convert at the boundary (`fild [int_slot]`
   / `fistp [int_slot]`).

3. **Float ABI.** cdecl returns floats on the FPU stack; callers
   `fstp` to consume. Float args are passed on the cdecl arg stack
   as 4-byte (float) or 8-byte (double) values. This works fine
   alongside the existing int args.

4. **Conversions.** Cast int→float (`fild`), float→int (`fistp`,
   with the FPU control word's rounding mode considered). Promotion
   rules in arithmetic (int + double → double).

A minimal float slice could cover just storage and FloatLiteral
init (no arithmetic / no conversions) if we want to land floats
incrementally. But the bigger win is plumbing the `st(0)` path
end-to-end first.
