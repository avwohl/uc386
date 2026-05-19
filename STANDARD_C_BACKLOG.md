# uc386 — remaining standard-C gcc-c-torture backlog

Tracked tail after the K&R/computed-goto/codegen-bug fix campaign.
Run a test: `.venv/bin/python run_gcc_torture.py --full --kr -v <name>`

## Fixed & pushed this campaign (for context)
- K&R/implicit-int pre-pass (uc_core `fb6a1a1`), computed-goto (`f0353e7`)
- `_Alignof` (uc386 `8372662`), bitfield-width crash (`eea0e11`),
  empty-struct defs (`2ef28c2`), anon-struct shape-hash miscompile
  (`4c59610`), printf/fprintf return value (`1881813`)
- octal char escape (uc_core `c4aeaba`), unsigned constant-fold UAC
  (`7e77976`), wide char constant value (`4da509b`)

gcc-c-torture: 1002 → ~1374/1514 (66% → ~91%); c-testsuite 215/220.

## Remaining genuine standard-C singletons (~30, no clusters)

Each is an independent investigation, ~1 test each.

| Test | Symptom | Diagnosis hint |
|---|---|---|
| 20010325-1.c | exit 1 | **wide string literals**: `L"a" "b"` concat + indexing + `sizeof`. Wide-string codegen (storage as int[] / element size), separate from the now-fixed wide *char* path. |
| va-arg-6.c, va-arg-8.c | exit 1 | varargs ABI corner — inspect `va_arg` slot stepping for the types these use. |
| conversion.c | exit 1 | float→int / rounding mode ("SPU float rounds toward zero" — FP conversion semantics). |
| 970217-1.c | exit 1 | `sub(int i, int array[i++])` — side effect in array-param declarator must be evaluated (i→11). |
| 20010904-1/2.c | exit 1 | (triage) |
| 20020227-1.c, 20020423-1.c (PR c/5430), 20020508-3.c, 20020904-1.c (PR c/7102), 20040411-1.c, 20041218-2.c, 960830-1.c, 991216-2.c | exit 1 | individual codegen-corner miscompiles — bisect each with exit-code repro. |
| pr23467.c, pr40386.c, pr43220.c, pr49039.c, pr49279.c, pr28982b.c | exit 1 / unicorn mem fault | per-PR codegen bugs; pr28982b/pr43220 = bad addressing (UC_ERR mem). |
| 20040811-1.c, 20060412-1.c, vla-dealloc-1.c | unicorn invalid mem/insn | bad codegen output; vla-dealloc-1 = VLA deallocation (C99). |
| bitfld-4.c | exit 1 | a *second* bitfield bug (not the shape-hash one) — specific width/op. |
| eeprof-1.c | exit 1 | (triage) |

## Out of scope / excluded (do not count as standard-C bugs)
- Nested functions / `__label__` (GCC ext, needs closure conversion +
  static-chain ABI): 20010209-1, 20010605-1, nest-stdar-1, pr22061-4, …
- `printf-2.c`, `fprintf-2.c`, `user-printf.c` — DOS file I/O
  (`fopen`/`tmpnam`), not codegen.
- `noinit-attribute.c` — `__attribute__((noinit))` GCC extension.
- `_Complex`/imaginary `i`-suffix literals (GNU), full complex codegen.
- `typeof` inside `va_arg` macro expansion — needs ctx-aware typeof
  resolution at the va_arg site (C23-standard but deep).

## Method that worked
Minimal exit-code-bisected repro → inspect emitted asm → fix at root
(uc_core frontend/_const/ast_optimizer or uc386 codegen/libc) →
regression-guard (uc_core `pytest`, uc386 `pytest`) → `--full --kr`
re-measure → commit per fix.
