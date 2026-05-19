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

## Status (post-grind)

The quick one-line bugs have largely been fixed (octal escape,
unsigned const-fold, wide char, bitfield crash, shape-hash,
printf-return, empty-struct, `_Alignof`). Continued grinding showed
the **remaining items skew to multi-site features, not singletons** —
e.g. wide string/char literal typing and 64-bit `va_arg` each span
several code paths (typing + index + sizeof + storage / dispatch +
load + consumers). They're best tackled as scoped feature tasks, not
in-session one-liners. Precise per-item findings recorded below so
each is resumable without re-triage.

## Remaining genuine standard-C items (~30)

| Test | Symptom | Diagnosis hint |
|---|---|---|
| ~~20010325-1.c~~ | **FIXED** `38d96ef` | wide string/char literal typing: `_wide_char_elem_type` (wchar_t=unsigned short/2) wired into `_type_of` for CharLiteral/StringLiteral/concat-list; sizeof+index follow. |
| <s>(orig note)</s> | | **wide string/char literals — multi-site, treat as a feature not a singleton.** Findings: storage IS correct (`L"ab"` → `dw 97,98,0`, wide intern works). Broken: (a) `_type_of(StringLiteral wide)` returns `char*` not `wchar_t*` because it checks `getattr(expr,"is_wide")` which is never set — must use `string_is_wide(expr.value.text)`; (b) it doesn't handle the adjacent-concat *list* `[L"a","b"]`; (c) `sizeof(L'x')` returns 4 (int) not 2 (`__SIZEOF_WCHAR_T__`=2) — `_type_of(CharLiteral wide)` must be `wchar_t`; (d) index lowering uses 4-byte stride/`mov eax,[eax]` instead of 2-byte `movzx word`. Fix needs coordinated CharLiteral+StringLiteral+list+Index+sizeof wide typing. uc386 wchar_t = `unsigned short` (2 bytes). |
| ~~va-arg-6.c, va-arg-8.c, 991216-2.c~~ | **FIXED** `9f0a307` | 64-bit `va_arg(ap,long long)`: added a VaArgExpr long-long case to `_eval_expr_to_edx_eax` (read 8 + advance 8, load `[ecx]`/`[ecx+4]`→EAX/EDX). 3 tests recovered. |
| conversion.c | exit 1 | float→int / rounding mode ("SPU float rounds toward zero" — FP conversion semantics). |
| 970217-1.c | exit 1 | `sub(int i, int array[i++])` — side effect in array-param declarator must be evaluated (i→11). |
| 20010904-1/2.c | exit 1 | (triage) |
| ~~20020423-1.c~~ (PR c/5430) | **FIXED** (uc_core) | `_nested_const_fold._new` minted folded constants via `make_int_lit(val)` w/o the unsigned flag: `(x+4)-8U` → bare decimal `4294967292`, which doesn't fit int/32-bit-long and wrongly promoted to **long long** (routed to the broken LL `int 0x80` div path). Now `make_int_lit(val, unsigned=is_unsigned)`, mirroring the main const-fold path (`7e77976`). |
| ~~20020508-3.c~~ | **FIXED** (uc386) | `_ll_shift_const` signed `>>` `s≥32` w/ `big_half_in_eax=True`: EAX held the source high half but EDX was caller-unspecified; `sar edx,31` sign-replicated **garbage** (unsigned `xor edx,edx` was unaffected). Seed `mov edx, eax` before the `sar` in the signed fast path. 64-bit rotate `(ll>>60)|(ll<<4)` now correct. |
| ~~20020904-1.c~~ (PR c/7102) | **FIXED** (uc386) | `(u8)255` cast (u8 = `typedef unsigned char`) emitted `movsx eax,al` → sign-extended to −1. `_cast`'s target-type was lowered before the typedef resolver was installed, so it stayed `BasicType(name='u8', is_signed=None)` and the narrowing chose movsx. Resolve the typedef name through `_resolve_typedef_name` (as `_size_of` does) so signedness is the underlying type's → `movzx`. |
| 20020227-1.c | (out of scope) | needs `__complex__ float` member codegen — see _Complex exclusion below, not a corner miscompile. |
| ~~991216-2.c~~ | **FIXED** `9f0a307` | stale duplicate of the 64-bit `va_arg` row above; verified PASS. |
| 20040411-1.c, 20041218-2.c | exit 1 | VLA / variably-modified types (`typedef int c[i+2]`; `struct {char b[n];}`) — C99 VLA feature, multi-site (sizeof of a runtime-sized type), not a one-liner. |
| 960830-1.c | (out of scope) | GCC extended inline `__asm__` with `=a`/`=d`/`%0`/`rm` operand constraints — GNU extension, not standard C. |
| pr23467.c, pr40386.c, pr43220.c, pr49039.c, pr49279.c, pr28982b.c | exit 1 / unicorn mem fault | per-PR codegen bugs; pr28982b/pr43220 = bad addressing (UC_ERR mem). |
| 20040811-1.c, 20060412-1.c, vla-dealloc-1.c | unicorn invalid mem/insn | bad codegen output; vla-dealloc-1 = VLA deallocation (C99). |
| ~~bitfld-4.c~~ | **FIXED** (uc_core) | not a bitfield bug — `_optimize_unary` folded `-123U` to `make_int_lit(4294967173)` w/o the unsigned flag → bare decimal promoted to **long long**, so `x.a (int:12) != -123U` ran as a 64-bit compare and the sign-extended bitfield mismatched. Now `make_int_lit(result, unsigned=int_flags(operand)[2])` for `-`/`+`/`~` (`!` stays int per C99 6.5.3.3). Same fold-unsignedness class as `7e77976`/`96b0e36`. |
| eeprof-1.c | (out of scope) | `-finstrument-functions` / `__cyg_profile_func_enter`/`exit` instrumentation hooks — GCC instrumentation feature, not standard C codegen. |

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
