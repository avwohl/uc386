# WIP — resume notes for the new machine

Phase 4 slices 0–32 done. Phase 5 (floats) slices 1–9 done.
va_list slice done. uc386's own suite: 310 tests passing.

## Resume checkpoint (2026-04-27)

Both repos clean and pushed to origin/main.

Latest session delivered +44 torture wins (1601 → 1645 combined).
See CLAUDE.md session log for full slice-by-slice details.

Key recent slices:
- vector_size: arithmetic, ABI, scalar broadcast, float vectors,
  compound assign, cast-from-scalar, cast-to-int (simd-1..6 + many pr*)
- LL bitfield: read (don't redundantly cdq) + write
  (`_assign_ll(Member)` route through bitfield store)
- LL switch dispatch (single + range with signedness)
- struct-member `__attribute__((aligned(N)))` honored in layout
- libc additions: memchr, qsort, copysign[f|l],
  __builtin_frame_address, __builtin_clrsbll
- Float globals: emit inf/nan as IEEE-754 hex pattern
- case `A ... B:` range syntax (parser + codegen)

Remaining 84 fails (21 compile + 57 run + 6 timeout) cluster around:
- VLA + alloca runtime support (~12 tests)
- Nested-fn label / trampoline (~8 tests)
- File I/O + vprintf (~8 tests)
- Inline asm with operand binding (~5 tests)
- _Decimal64 / __int128 / _Complex char (~5 tests)
- Bitfield-precision arithmetic (GCC-specific, ~4 tests)
- Long tail of one-offs (alias attribute, label_values + && diff
  in nested fns, etc.)

Next slice ideas in order of yield:
1. VLA + alloca → unlocks a cluster of similar tests at once
2. Nested-fn cross-frame labels via small trampoline shim
3. vprintf in libc (would unlock several printf-chk tests)
4. Bitfield-precision arithmetic (mostly gcc-specific test
   semantics, but a couple unlock cleanly)

## ANSI C testsuite runners

Four runner scripts at the repo root mirror uc80's setup:

- `run_ctests.py`  — c-testsuite (https://github.com/c-testsuite/c-testsuite)
- `run_fujitsu.py` — Fujitsu CompilerTestSuite
                     (https://github.com/AcademySoftwareFoundation/CompilerTestSuite)
- `run_sdcc.py`    — SDCC regression (skeleton; template-expansion not ported)
- `run_gcc_torture.py` — GCC torture suite via llvm-test-suite

Each expects upstream sources cloned at sibling paths under
`../external/`:

    git clone https://github.com/c-testsuite/c-testsuite.git \
        ../external/c-testsuite

The runners default to `--compile-only` because the i386 assemble →
link → run → diff pipeline is not yet wired. Headers in
`lib/include/` (copied from uc80) are passed via `-I`.

Latest tally — `python run_ctests.py --full` and
`python run_gcc_torture.py --full`:

    c-testsuite     215 / 220   (97.7%)  — running for real
    gcc-c-torture  1430 / 1514  (94.5%)  — running for real
    Combined       1645 / 1734  (94.9%)

The full run-mode pipeline is wired:
uc386 → .asm → bundle libc.asm → nasm -f bin → unicorn-engine →
diff stdout against the test's `.expected` (or check exit code).
INT 21h is intercepted by `src/uc386/dos_emu.py` so DOS-style
syscalls reach a Python-side handler. Long-long divide / modulo
goes through INT 0x80 to avoid clobbering EAX with a dispatch byte.

Remaining c-testsuite gaps: 00216 (init-list torture), 00187
(real file I/O), 00200 (shift-type subtleties), 00204 (struct
va_arg in detail), 00219 (const-qualifier edge of `_Generic`).

Remaining torture gaps cluster around features that each warrant
their own slice:
  - __complex__ types (~150 tests)
  - inline asm (`asm("...")`)
  - nested function definitions (gcc extension)
  - VaArgExpr address-of and struct va_arg edges
  - aligned-attribute / packed-struct
  - VLA in struct/param-list with side effects
  - GCC label-values (`&&label`, computed goto)
  - file I/O

To finish the pipeline:
1. NASM assembly (`-f bin` works; for `-f obj`/`-f elf` we'd need a linker).
2. A DOS extender stub or DOS/4GW-compatible loader so `bits 32`
   binaries actually run under DOS.
3. A libc — uc80's `lib/libc.lib` is Z80; we'd need an i386 port.
4. A working DOS emulator hookup (dosbox / dosbox-x / dosemu — the
   latter two are present locally at `/Users/wohl/src/dosemu/`).

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

Implemented in the va_list slice (just landed):
- **Variadic function definitions.** User supplies
  `typedef char *va_list;`; uc_core's parser handles
  `va_arg(ap, T)` as a builtin form, and uc386 lowers va_start
  (lea past the last named param) / va_arg (read + advance) /
  va_end (no-op).

Deliberately not yet implemented:
- **`register` storage class.** It's just a hint; we ignore it. No
  correctness gap.
- **Implicit `va_list` typedef.** Currently the user must declare
  `typedef char *va_list;` themselves. A small uc_core change could
  predefine it, but it's not blocking real code.

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
