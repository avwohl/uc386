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

gcc-c-torture: 1002 → 1397/1514 (66% → 92%); c-testsuite 215/220.

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

## Remaining items (~8 tests, all features or GNU extensions)

No standard-C codegen singletons remain — see the two sections below.

### Fixed this campaign

| Test(s) | Commit | Root cause |
|---|---|---|
| ~~20010325-1.c~~ | `38d96ef` | wide string/char literal typing: `_wide_char_elem_type` wired into `_type_of` for CharLiteral/StringLiteral/concat-list. |
| ~~va-arg-6/8.c, 991216-2.c~~ | `9f0a307` | 64-bit `va_arg(ap,long long)` VaArgExpr case in `_eval_expr_to_edx_eax`. |
| ~~20020423-1.c~~ (PR c/5430) | `96b0e36` (uc_core) | `_nested_const_fold._new` minted folded constants w/o the unsigned flag → bare decimal promoted to **long long** → broken LL `int 0x80` div. Now `make_int_lit(val, unsigned=is_unsigned)`. |
| ~~20020508-3.c~~, ~~pr40386.c~~ | `9b0f255` (uc386) | `_ll_shift_const` signed `>>` `s≥32` w/ `big_half_in_eax`: EDX caller-unspecified, `sar edx,31` sign-replicated garbage. Seed `mov edx,eax` first. |
| ~~20020904-1.c~~ (PR c/7102) | `34d45ab` (uc386) | `(u8)255` (u8=typedef unsigned char) emitted `movsx`; `_cast` target stayed `BasicType(name='u8',is_signed=None)`. Resolve typedef via `_resolve_typedef_name` before movzx/movsx. |
| ~~bitfld-4.c~~ | `2ee96e4` (uc_core) | `_optimize_unary` folded `-123U`→bare `4294967173`→long long → 64-bit compare mismatch. `make_int_lit(result, unsigned=int_flags(operand)[2])` for `-`/`+`/`~`. |
| ~~20060412-1.c~~ | `1851278` (uc_core) | DSE: auto-AST splits `Member`(`.`)/`ArrowMember`(`->`); `_expr_references_var`/`_expr_has_pointer_or_call` only matched `ast.Member`(+stale `.is_arrow`), so `p=&t; p=&((T*)p)->m[0];` dropped `p=&t`. Added `_MEMBER`; `ArrowMember`=deref. |
| ~~pr49039.c~~, ~~conversion.c~~ | (cascaded) | recovered by the fold-unsignedness / LL-shift fixes; verified PASS. |

### Remaining genuine standard-C codegen singletons

**None.** The campaign exhausted the standard-C codegen-corner
*miscompiles*, then took on the **C99 VLA / variably-modified-types
feature** end to end — deallocation, `sizeof` of a VLA typedef,
`sizeof` of a struct with a variably-modified member (6.7.7p2
evaluate-once), and the VMT-parameter side effect are all now
fixed (rows below). Every remaining executable failure is a
larger scoped *feature* (`_Complex`) or a GNU extension. Confirmed
by per-test triage; not claimed as passing.

### Feature / out-of-scope (not codegen-corner singletons)

| Test(s) | Class | Note |
|---|---|---|
| ~~pr43220, vla-dealloc-1, 20040811-1~~ | **FIXED** (uc386) | C99 VLA *deallocation*. Basic VLA alloc/index/sizeof already worked; the `__vla_baseline` save+goto-restore mechanism existed but was **dead code** — the `has_any_vla` pre-pass tested `isinstance(sub,(VarDecl,_SynthLocalVar))` while running *before* `_collect_locals` synthesises those, so it never saw the parsed `ArrayDeclarator` and `vla_baseline_disp` stayed `None`. A goto-back never freed VLAs → unbounded stack growth → `UC_ERR_WRITE_UNMAPPED`. Now detect a VLA from the parsed `ArrayDeclarator` (size present, non-`IntLiteral`, not const-foldable). |
| ~~970217-1~~ | **FIXED** (uc_core) | C99 6.7.6.3p7/6.9.1p10 VMT-parameter side effect: `sub(int i, int array[i++])` must evaluate `i++` on entry (i→11). The size expr survives as `pt.size_expr` on the array `ResolvedType`; only `_decay_for_param` dropped it when collapsing to a pointer. `_fd_params` now collects the non-`IntLiteral` dimension exprs down the array chain into `_ParamView.size_side_effects` (purely additive — that field was vestigial; uc386 codegen read-sites 3645/3846/4004 already consume it; uc80/uplm80 ignore it). `_sub` now emits `inc [esp+4]` then returns 11. |
| ~~20040411-1~~ | **FIXED** (uc386) | C99 VLA-typedef `sizeof`. `sizeof(<typedef-name>)` passed an unresolved `TypeName(TypedefNameSpec('c'))` to `_emit_runtime_size_of` → `_type_has_vla` False → static `mov eax,4`. Now resolve the typedef (TypeName→legacy; typedef-name→`_resolve_typedef_name`→`resolved_to_legacy`) at the top of `_emit_runtime_size_of`, so a VLA typedef routes through the runtime path (cf. `_cast` fix `34d45ab`). Gated by `_is_genuine_vla_array` (`aa9dc2c`) so only a genuine — non-const-foldable — VLA array typedef is adopted; constant/struct typedefs keep the byte-identical pre-fix static path (the unguarded `f9d8fe4` mis-routed those into the runtime path → compile regression, caught by the sweep). *Known limitation:* re-evaluates the size expr at the `sizeof` site, not C99 6.7.7p2 evaluate-once-at-typedef-decl — correct whenever the size operands are unmodified between typedef and `sizeof` (the normal case + what the test exercises). |
| ~~20041218-2~~ | **FIXED** (uc386) | C99 VMT-in-struct `sizeof` evaluate-once (6.7.7p2): `struct s{char b[n];}; n++; sizeof(struct s)` must be 123 not 124. The `_capture_struct_vla_member_sizes` machinery only fired for `ast.StructDecl`; an in-function `struct s{…};` is `ast.Declaration`+`StructDef`. Wired capture into the `_collect_locals` Declaration/StructDef branch (run on a `SimpleNamespace` shim of `st`, *before* `_resolve_struct_name` so the registered member ArrayType carries `Identifier(slot)`; stash on the Declaration node) and the eval+store into `_lower_declaration` (at the decl point, before later mutation). Also generalised `_is_genuine_vla_array` to recurse `StructType` members so a struct-VMT typedef is adopted for the runtime-sizeof path while const-foldable struct typedefs stay static (keeps the `aa9dc2c` regression fix). The `_collect_locals` wiring is gated on a genuine-VLA member check (`_const_eval`, not the weak `_try_simple_int_fold`) so a constant member (`int a[ENUM]`, `char b[sizeof T]`) is *not* slot-captured — caught by the sweep as a +4 compile regression first, then gated. |
| pr23467, 20010904-1, 20010904-2 | GNU ext | `__attribute__((aligned(N)))` type-alignment in struct layout/sizeof/stride — multi-site, non-standard. **Parsed but silently ignored**: `struct __attribute__((aligned(16))) A { int x; }` compiles and reports `sizeof == 4`, not 16. Wrong answer rather than a diagnostic — the risk if period headers rely on it. |
| 20020227-1 | C99 `_Complex` | `__complex__ float` member codegen — full complex feature. |
| 960830-1, pr49279 | GNU ext | extended inline `__asm__` with operand constraints. |
| eeprof-1 | GNU ext | `-finstrument-functions` / `__cyg_profile_func_*`. |
| pr28982b | edge | 0x80100-byte by-value struct / stack frame — large-frame, not codegen. |
| ~~(not a torture test)~~ | **FIXED** | `--int` / `--long` / `--long-long` / `--ptr` changed `sizeof` but not storage: the flags feed the frontend (ASTOptimizer const-folds `sizeof`) while codegen sizes from its own `_BASIC_SIZES` and never saw them, so `--long 64` reported `sizeof(long)==8` with `long` locals 4 bytes apart. Making it *work* would mean size-keying every 64-bit path — they key off the name (`_is_long_long`), not the width — so the driver now **refuses** the combination instead of miscompiling, comparing the requested `TypeConfig` against codegen's real table so the guard lifts by itself if codegen ever gains a width. `tests/test_width_flags.py`. |
| ~~(not a torture test)~~ | **FIXED** | stdio position/state group was no-op stubs: `feof` never reported EOF (`while (!feof(f))` hung), `fseek` didn't seek, `ftell` returned 0. Now real — seek via INT 21h AH=0x42 through `_lseek`, per-stream EOF/error in the handle-indexed `_stdio_flags` table, and `fgetc`/`fread` check CF so an I/O error reaches `ferror` instead of being read as a byte count. Also fixed `_lseek` clobbering callee-saved EBX, and dos_emu not clearing CF on successful AH=0x3F/0x40. `tests/test_stdio_position.py`. All four read paths (`fgetc`/`fread`/`fgets`/`getchar`) set the flags, handles are normalized through `__stdio_flag_ptr` so `getchar`'s raw fd 0 and `feof(stdin)`'s `0xF0` sentinel agree, and every INT 21h read/write caller checks CF — ignoring it turned a DOS error *code* into a byte count (`read()` reported 6 bytes read; `fgets()` returned garbage instead of NULL). dos_emu now sets CF on error and reuses the lowest free handle (it used to reach fd 303 after 300 open/close cycles, past the flag table). `errno` is now populated from the DOS codes that were previously discarded at each call site, `strerror` maps it to real messages, and `perror` prints them (`tests/test_errno_strerror.py`). Console output is now line-buffered (`__stdio_putc_con`): printing 2,000 bytes went from **2,000 DOS calls to 2**, flushed on newline / buffer-full / `fflush` / `fclose` / exit, with `setvbuf` honoring all three modes (`tests/test_stdio_buffering.py`). The exit flush is emitted by codegen, since return-from-main never enters the libc, and only for programs that print — `true.bin` is still 18 bytes. Genuinely still missing: `popen`/`pclose`. |

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
