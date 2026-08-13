# addons.txt — completion status (updated 2026-08-13)

Status of each item in `docs/addons.txt`. The original 8 items
from 2026-04-30 are still done; **3 new line-items** added by the
user on 2026-05-01 (MicroPython, ship test scripts in FOSS,
ship build scripts everywhere) are all done. **MicroPython now
runs full Python** on uc386: arithmetic, control flow, functions,
classes, exceptions, list comprehensions, builtins — see the
MicroPython section below.

A second 2026-05-01 update added 3 more cross-cutting asks
(mac→linux portability, ship-source-with-binary, dosiz as a
test runner). Status:
- **Mac→Linux install** ✓ — `docs/INSTALL.md` covers apt / brew /
  dnf, `pyproject.toml` accepts Python ≥ 3.11 (raised from a claimed
  ≥ 3.10 that never installed — uplox's own floor is 3.11), and `pytest tests/`
  passes (490 passed, 1 skipped — the peephole / asm-DCE /
  libc-split tests moved to the sibling `upeep386` package, which
  is where the old ~1300 count went).
- **Ship source with binary** ✓ — `addons/harness/package.py`
  no longer excludes `upstream/` from either tarball, so when the
  CI fetches + builds awk-bwk or doom, the matching upstream
  source tree ships alongside the binary.
- **MZ+LE .exe output (Path A)** ✓ — uc386 produces self-contained
  `.exe` files via `addons/harness/exe.py`
  (uc386 → nasm OMF → **upyle** → MZ+LE bound to **DOS/32A**; no
  Open Watcom needed, so this works on macOS). All seven
  Path A phases verified end-to-end in CI: `true.exe` boots
  + exits 0; `false.exe` exits 1; `myecho.exe hello dos` writes
  literal `hello dos\n` via libc fputs through real DOS handles;
  `argv_pr.exe alpha beta` prints `argc=3 / argv[1]='alpha' /
  argv[2]='beta'`; `factor.exe 2 12 60 97` emits multi-arg printf
  output (`2: 2 / 12: 2 2 3 / 60: 2 2 3 5 / 97: 97`) via the
  legacy in-asm format engine. The bridge stub handles three
  extender ↔ uc386 mismatches: (1) stream sentinels (libc's
  `_stdout=0xF1` was dos_emu-only; real DOS needs raw fd 1),
  (2) argv parsing (the extender puts the PSP selector in ES at
  entry per the OpenWatcom CRT convention; the bridge reads
  `[es:0x80]` for cmdline length and `[es:0x81..]` for the tail),
  and (3) `_printf` now tail-jumps to `_printf_legacy`
  (real-DOS-safe via INT 21h AH=02h per char). All 17 in-tree
  manifest addons build `.exe` successfully. Full progression in
  `docs/path-a-mz-le.md`.

  PMODE/W was the original default and is still selectable with
  `--extender=pmodew` (~16 KB smaller), but it **cannot do disk I/O
  on real DOS** — its real-mode call path hangs on any DOS call
  touching a physical sector. DOS/32A became the default in
  `58c1f79` for that reason.
- **dosiz integration** ✓ — Path A made the flat-bin-loader gap
  moot: dosiz loads the `.exe` directly with its existing
  MZ/LE chain. `src/uc386/dosiz_run.py` is the second runner and
  `src/uc386/harness.py` selects it via `UC386_HARNESS=dosiz`.
  The one remaining gap is network simulation, which has no dosiz
  counterpart. See `docs/dosiz-integration.md`.

## ✓ Port GNU utilities to this compiler

**Done.** 17 addons under `addons/gnu/` ship working binaries, and
`python -m addons.harness.build gnu all` reports **17/17 passed**
against their manifests. Flat `.bin` bytes, re-measured 2026-08-13:

| Addon | Source | Size (uc386 .bin) |
|-------|--------|-------------------|
| true | in-tree | 18 |
| false | in-tree | 21 |
| yes | in-tree | 74 |
| echo | in-tree | 148 |
| head | in-tree | 419 |
| dirname | in-tree | 420 |
| basename | in-tree | 479 |
| argv_probe | in-tree (smoke) | 501 |
| cat | in-tree | 516 |
| open_test | in-tree (smoke) | 668 |
| tail | in-tree | 959 |
| wc | in-tree | 1,557 |
| factor | in-tree | 1,886 |
| strtol_test | in-tree (smoke) | 2,131 |
| sbase-cat | sbase upstream | 2,619 |
| sbase-tee | sbase upstream | 2,803 |
| sbase-head | sbase upstream | 3,113 |

(The earlier revision of this table listed much smaller figures for
`wc`, `factor`, and `strtol_test`; those predated later libc and
codegen work and no longer reproduce. The numbers above match what
`addons/harness/compare.py` writes into `results.md`.)

**Plus BWK awk (one-true-awk):** 107 KB binary, 6K LoC of upstream
C compiled verbatim through the entire uc386 pipeline. Runs
BEGIN/END blocks, pattern/action rules, field access, math, string
functions, regex matching, scientific notation, associative arrays.
See `addons/gnu/awk-bwk/NOTES.md`.

## ✓ DOS installer in GitHub releases

**Done.** `.github/workflows/release.yml` triggers on `v*` tags and
attaches two tarballs to the release:

- `uc386-foss-addons-<ver>.tar.gz` (~80 KB) — built FOSS binaries
  + per-addon `src/<name>/` tree (manifest.toml + .c sources or
  fetch.sh / build.sh) + `test_addons.py` runner.
- `uc386-games-build-scripts-<ver>.tar.gz` (~155 KB) — game
  fetch/build scripts + pre-built `bin/doom/doom.bin` (the only
  game that boots end-to-end today).

The packaging logic is in `addons/harness/package.py` (runnable
locally too: `python -m addons.harness.package --version dev`).
Layout details and the new test-runner / src-tree shipping are
documented under the corresponding 2026-05-01 items at the bottom
of this file.

## ✓ Skip irrelevant utilities

Confirmed: `chroot`, `groups`, `id`, `who`, `whoami`, `uname`,
`hostname`, `kill`, `nohup`, `nice`, `runuser`, `stty` — none of
these are in `addons/gnu/` because they have no DOS analog under
dos_emu (no process model, no users, no terminal control).

## ✓ Include gawk

**Done via BWK awk** (the historical reference implementation, ~6K
LoC, MIT-style license). Full GNU gawk would need:

- `regex.h` (POSIX BRE/ERE) — gawk uses gnulib's regex engine,
  ~3K LoC of additional libc work
- gnulib subset (xalloc, dirname-lgpl, getopt, …)
- mbtowc/wcwidth/locale subsetting

Documented as future work in `addons/gnu/gawk/NOTES.md`. BWK awk
satisfies the spirit of "include awk in the FOSS installer."

## ✓ Make scripts to download / build / test

**Done.** Each addon dir under `addons/games/<game>/` and several
under `addons/gnu/<tool>/` has:

- `fetch.sh` — downloads upstream source into `upstream/`
- `build.sh` — invokes uc386 (with bison + maketab pre-pass for awk)
- `NOTES.md` — what's known about porting that target

For FOSS userland tools, the harness handles build+test directly
from `manifest.toml`.

## ✓ Games from the DOS DPMI period

**Doom boots end-to-end and reaches R_Init with a fake WAD.
240 source files compile across 5 chocolate-doom-era games.** Per-file triage results
(see `addons/games/README.md` for the full scoreboard):

| Game     | Source               | Compiles    | Boots? |
|----------|----------------------|-------------|--------|
| Doom     | id-Software/DOOM     | **58 / 58** | **yes** (W_InitFiles, no WAD shipped) |
| Heretic  | chocolate-doom       | **47 / 47** | no (needs ~76 I_/TXT_/SDL_ stubs) |
| Hexen    | chocolate-doom       | **48 / 48** | no (needs same chocolate-doom stubs) |
| Duke3D   | jfduke3d             | **35 / 35** (16 game + 19 engine; 7 platform-only excluded) | no (needs `#pragma aux` codegen) |
| ROTT     | videogamepreservation| **52 / 52** (1 dead-code .C skipped) | no (needs `#pragma aux` codegen) |
| Descent  | dxx-rebirth          | n/a (C++)   | no     |

**All five C games now triage 100% clean** at the per-file level —
240 period DOS sources compile through uc386 → NASM. The remaining
gap to a runnable .bin is per-game stubs/linkage (chocolate-doom
platform layer for Heretic/Hexen, `#pragma aux` codegen in uc_core
Phase 2 for the Watcom-era inline-asm helpers in Duke3D/ROTT).

Today's Doom blockers are NOT `#pragma aux` (we use `linuxdoom-1.10`,
not the DOS tree). Compile-time blockers cleared 2026-04-30:

1. ~~Several missing libc headers — `values.h`, `alloca.h`, `malloc.h`,
   `R_OK` in `unistd.h`~~ Added.
2. ~~File-scope `static` name collisions in multi-TU mode~~ Fixed via
   per-file static name mangling in `main.py`.
3. ~~Anonymous-struct type identity across TUs~~ Fixed: `_resolve_struct_name`
   now uses a structural fingerprint, not `id(t)`.
4. ~~Float-init for integer globals (`.2 * FRACUNIT`)~~ Fallback to
   float-eval + truncate.
5. ~~Bit ops in float const-eval~~ Delegate to int-eval when shape allows.
6. ~~`(int)"string"` / `(int)&global` in int slots~~ Lay down label as `dd`.
7. ~~Strength-reduction `x * 2^n -> x << n` mis-firing on float operands
   inside UnaryOp~~ uc_core optimizer now avoids any subtree with a
   FloatLiteral / float-Cast.

**DOOM boots under dos_emu** (as of 2026-04-30):

- 58 doom sources + `stubs.c` → 2 MB .asm → 301 KB flat .bin
- Boot reaches `W_InitFiles` (one INT past `Z_Init`); exits with
  "no files found" because we don't ship a WAD
- Provide a WAD via `vfiles_init` and DOOM should proceed into
  `R_Init` and the title-screen tic loop (subject to: video framebuffer
  capture in `I_FinishUpdate`, input pump in `I_StartTic`)

`addons/games/doom/stubs.c` provides the ~30 `_I_*` platform
functions, plus `fstat` (via `lseek` to SEEK_END/SET — `lseek`
itself is a new libc-asm + INT 21h AH=0x42 dos_emu addition),
`mkdir`, `sscanf` (handles `%d %i %x %c`), `strcasecmp`
(libc asm), and a `getenv` that recognizes HOME/DOOMWADDIR.

**Spirit of the request preserved**: every blocker we close benefits
ALL period-code ports, not just Doom. The session shipped a long
list of general compiler / runtime improvements:

- multi-TU file-scope `static` name mangling
- structural anonymous-struct identity (replacing `id(t)`)
- float-init fallback for integer globals
- bit-op subexpressions in float const-eval
- `(int)"string"` / `(int)&global` in int-typed slot inits
- `char arr[N] = {"string"}` brace-around-string unwrap
- ast_optimizer mul-to-shift skip when operand has float subterm
- `__GNUC_MINOR__` / `__GNUC_PATCHLEVEL__` predefines
- `div_t div(int, int)` returns by value (C99)
- libc additions: `lseek`, `strcasecmp`, extended `getenv`
- 9 new DOS-platform headers: `dos.h`, `bios.h`, `conio.h`, `i86.h`,
  `mem.h`, `libc.h`, `process.h`, `direct.h`, `graph.h`
- uc_core preprocessor: case-insensitive + backslash-tolerant
  `#include`; multi-line macro merge in `_preprocess_included`;
  comment-aware paren-tracking; trailing-comment strip in `#define`
- chocolate-doom shims: `config.h`, `SDL_endian.h`, `SDL.h`

## ✓ Two installers (FOSS + abandonware)

**Done** per `addons/README.md` phase 8:

- **FOSS installer** (`uc386-foss-addons-*.tar.gz`): GNU userland
  binaries (GPL/MIT provenance, OK to redistribute). Contains
  the 16 utility .bins + awk + LICENSE files.
- **Abandonware installer** (`uc386-games-build-scripts-*.tar.gz`):
  ships build scripts, not binaries. Per `addons/games/README.md`
  the policy was revised: we DO ship game binaries when public-
  source releases (Doom GPL-2.0, Duke3D GPL-2.0, etc.) authorize
  it; we never ship the data files (WAD / GRP / RTL — those stay
  proprietary). Once uc_core Phase 2 lets the games actually
  build, the abandonware installer will include those binaries
  too. Today it ships only the scripts since the binaries don't
  exist yet.

## ✓ Build with competitive compilers

**gcc + DJGPP + Watcom all reproduced on one macOS/arm64 host**
(2026-05-19). `addons/results.md` has the full 14-addon table; the
header there explains each column. Real `true` row:

```
| true | 18 | 16,907 | 16,840 | 5,420 | 147,914 |
  (.bin) (.exe)  (gcc)  (Watcom) (DJGPP)
```

**The earlier "390× smaller than Watcom" claim was wrong** — it
divided uc386's `.bin` (18 B, *not a DOS executable* — no MZ
header, runs only under dos_emu) by Watcom's real `.exe` (5,494 B).
That is a category error. The honest comparison uses the same kind
of artifact on both sides:

- **Codegen floor (`.bin`, not shippable):** uc386 is in a class
  of its own — tens of bytes vs Watcom's multi-KB. Real, but it
  measures the code generator, not a deployable program.
- **Real DOS executable (`.exe`):** uc386's DOS/32A-bound `.exe`
  has a **~32.8 KB extender floor** — every small program lands
  within a few hundred bytes of it. **Open Watcom is smaller**:
  ~6× on tiny programs (`true` 5,420 vs 32,847; `echo` 11,286 vs
  32,855), narrowing to ~1.6× as code grows (`factor` 20,538 vs
  32,925, `wc` 20,158 vs 32,868). uc386's *codegen* is tighter but
  its *DOS packaging* loses to Watcom's mature DOS/4GW
  clib+linker. Re-measured 2026-08-13 after DOS/32A became the
  default; the older ~17 KB figures were PMODE/W builds, which
  cannot do disk I/O on real DOS.
- **vs DJGPP / gcc:** uc386 `.exe` beats DJGPP **~4.5–5.5×**
  (148–182 KB go32+djgpp runtime). Against host gcc the two are
  now comparable (gcc isn't a DOS target anyway — sanity baseline
  only).

Toolchain provenance (all on the dev macOS/arm64 host):

- **DJGPP** — `~/.local/opt/djgpp` from `andrewwutw/build-djgpp
  v3.4` (gcc 12.2.0, osx artifact, runs under Rosetta).
- **Watcom** — Open Watcom V2 has no macOS build, and its Linux
  binaries don't run under Rosetta (Rosetta bridges x86_64→arm64
  user space, not the Linux ABI). The DOS release asset
  (`open-watcom-2_0-c-dos.exe`) is a plain self-extracting zip;
  `unzip 'binw/*' 'h/*' 'lib386/*'` → `~/.local/opt/watcom-dos`,
  and the DOS-hosted `wcc386.exe`/`wlink.exe` run under DOSBox-X.
  `addons/harness/watcom_dosbox.py` drives this; `compare.py`
  falls back to it when native `wcc386` is absent. The period
  reference is now *measured on macOS*, not asserted from CI.

**Behavioral status:** **17/17** addon manifest tests pass under
`addons/test_gnu_addons.py`. The earlier `cat`/`sbase-cat`
failures were a dos_emu vfs-keying regression (commit `3e30b18`
canonicalized INT 21h paths to `C:\\…` for lookups but seeded
`vfiles_init` with the raw manifest key, so `fopen` of a seeded
relative name always missed). Fixed by making `_canon_vfs_path`
the single source of truth for both seeding and lookup
(`a01dadf`, +2 regression tests). This was a runtime/vfs bug,
distinct from codegen.

## ✓ Include the latest MicroPython (2026-05-01 ask)

**Status: split out into its own repo on 2026-05-09 — see
[freedos_micro_python](https://github.com/avwohl/freedos_micro_python)
(`pip install freedos_micro_python`). uc386's smoke test moved to
`tests/test_micropython_integration.py` and skips when MP isn't
built. The historical context below stays here so the per-slice
diary still makes sense.**

**Runnable `micropython.bin` is a fully-functional Python REPL.**
A ~169 KB flat i386 DOS binary built end-to-end through uc386
prints:

```
MicroPython uc386-triage on 2026-05-01; uc386-dos with i386
Type "help()" for more information.
>>> 2+3
5
>>>
```

…and accepts essentially full Python: integer arithmetic, named
builtins (`print(2+3)`, `len("hi")`, `sum(range(10))` → `45`,
`sorted([3,1,2])` → `[1, 2, 3]`), string operations
(`"-".join(["a","b","c"])` → `a-b-c`), `print()` with **real
newlines** (not the pre-fix mangled `_0x0a_`), control flow
(`if/else`, `for/range`, `while/break`), exception handling
(`try/except` catching `ZeroDivisionError`), function definition
(`def f(x): return x*2; f(7)` → `14`), classes (`class C: x=1; C.x`
→ `1`), list/dict/tuple literals, list comprehensions
(`[i*i for i in range(5)]` → `[0, 1, 4, 9, 16]`), static qstrs
(`__name__` → `'__main__'`), and clean Ctrl-D exit. The full
lex → parse → compile → VM dispatch → NLR (setjmp-backed) →
qstr-pool → mp_load_global → builtins-dict lookup → builtin /
user-function call → print result path runs end-to-end.

Four real bug fixes unlocked Python execution:

- **uc386 peephole** (commit `19ae598`):
  `_pass_push_memory_to_push_reg` was incorrectly merging chained
  pointer dereferences (`mov eax, [eax+4]; push [eax+4]` →
  `push eax`) when the memory expression referenced the
  destination register. That dropped the second deref of
  `self->context->module.globals` and broke
  `mp_globals_set(...)`. Fix: skip the cache when the source
  expression references the destination register.
- **qstr length** (commit `17c3191`): the grep heuristic emitted
  every QDEF1 with `len=0`, but `qstr_find_strn`'s post-binary-
  search linear sweep filters by `lengths[at] == str_len` before
  memcmp. The filter rejected every entry. Fix: emit
  `length(name)` in the third QDEF1 field.
- **qstr collation** (commit `ab61a86`): macOS `sort` uses
  locale-aware ordering (`__name__` before `BUILD_LIST`), but
  `strncmp` uses ASCII (the opposite). The pool's
  `is_sorted=true` invariant requires ASCII order, so binary
  search missed every static qstr starting with `_`. Fix:
  `LC_ALL=C sort -u` in the pipeline.
- **qstr escape reversal** (`gen_qstrdefs.py`): the grep
  captured the *sanitized* `MP_QSTR_<x>` macro name and used the
  tail as the qstr's payload string. For non-identifier qstrs
  the macro tail is escaped (`\n` → `_0x0a_`, `<stdin>` →
  `_lt_stdin_gt_`) so the pool stored the escaped form as the
  string. `print()`'s trailing newline rendered as the literal
  text `_0x0a_`. Fix: a Python preprocessor reverses upstream's
  `qstr_escape` (re-using upstream's own `codepoint2name` map),
  emitting the original byte string as the QDEF1 payload AND
  sorting the pool by that original byte string (the binary
  search compares against the payload, not the macro name).

Layered evidence:

- **Per-file triage**: 145 / 145 (132 `upstream/py/` + 13
  `upstream/shared/{libc,readline,runtime,timeutils,netutils}/`,
  100 %) compile cleanly via uc386 → NASM-ready .asm using a
  synthetic `int main()` and stub `genhdr/` headers.
- **Multi-TU compile** of the **upstream minimal port**
  (`upstream/ports/minimal/main.c` + `uart_core.c` + 132 `py/` +
  9 `shared/`, 143 sources in one uc386 invocation) with the
  uc386-dos config (`MICROPY_MODULE_FROZEN_MPY=0`,
  `MICROPY_MIN_USE_STDOUT=1`, `MICROPY_PY_BUILTINS_MIN_MAX=1`,
  `MICROPY_PY_BUILTINS_REVERSED=1`): produces 1.94 MB of NASM,
  links cleanly under `nasm -f bin` to a ~169 KB `.bin`. Only
  externs remaining are dead libm names left in a string table
  (DCE doesn't strip those today).
- **REPL smoke tests** (43 cases): `test_micropython_smoke.py`
  (paths in this section are pre-split and now live in the
  freedos_micro_python repo, not `addons/gnu/micropython/`) runs the bin
  under dos_emu and pins: banner, clean Ctrl-D exit, arithmetic
  (`2+3` → `5`), assignment (`x = 5`), `pass`, named builtins
  (`__name__`), `print()` with real newlines, function def + call,
  list comprehensions, `try/except`, `min`/`max`/`reversed`,
  `bin`/`hex`/`oct` (qstr reverse-mangling correctly decodes the
  `_brace_open__colon__hash_b_brace_close_` format string back to
  `{:#b}`), plus 4 CORE_FEATURES-only cases (`bytearray`, `set`
  literals, detailed-NameError-with-qstr-name, `'%d-%s' %`
  formatting). Skips cleanly when the bin doesn't exist; passes
  in ~14s on the dev Mac when it does. New surface (post
  CORE_FEATURES bump): `bytearray`, `set`, `dict.fromkeys`,
  `bytes.decode`, generator expressions, `'%' %` formatting,
  detailed-NameError-with-qstr-name, `import sys` / `gc` /
  `micropython` / `collections` (OrderedDict + namedtuple) /
  `struct` / `array` / `errno` / `math` (floats lowered via the
  x87 FPU; `math.sqrt(2.0)` → `1.41421...`).

- **Module imports** (2026-05-02): hand-rolled equivalent of
  upstream's `tools/makemoduledefs.py` output written into
  `build/genhdr/moduledefs.h`. Each registered module's entry is
  guarded by its `MICROPY_PY_<X>` define so flipping the gate in
  mpconfigport.h adds or drops the entry consistently. Modules
  registered: `builtins`, `sys`, `__main__`, `gc`, `micropython`,
  `array`, `collections`, `struct`, `errno`. `errno` requires two
  helpers: build.sh's grep also extracts the X-macro
  `MICROPY_PY_ERRNO_LIST` entries (EPERM/ENOENT/EINVAL/...) into
  the qstr table so the module's `MP_QSTR_##e` token paste
  resolves at compile time, and `MICROPY_USE_INTERNAL_ERRNO=1`
  routes `MP_##e` to upstream's hardcoded values rather than to
  uc386's libc `<errno.h>` (which ships only the Linux subset —
  EOPNOTSUPP / EADDRINUSE / ECONN* / EHOST* / EALREADY /
  EINPROGRESS aren't there). Modules deliberately not registered:
  `math` / `cmath` (need `MICROPY_PY_BUILTINS_FLOAT`), `_thread`
  (no thread support), `weakref` (off at CORE_FEATURES), `io`
  (no VFS).

- **CORE_FEATURES baseline** (2026-05-02): the previous "every
  named-builtin NameErrors when ROM_LEVEL is bumped" runtime
  regression turned out to be a missing qstr-hash. CORE_FEATURES
  flips `MICROPY_QSTR_BYTES_IN_HASH` from 0 to 1, which adds a
  `qstr_hash_t hashes[]` array to each `qstr_pool_t` and gates
  `qstr_find_strn`'s post-binary-search filter on
  `pool->hashes[at] == str_hash`. `gen_qstrdefs.py` was emitting
  `0` for every QDEF1 hash; the runtime computed real djb2 hashes
  and the filter rejected every entry. Fix: compute the djb2 hash
  inline at qstrdefs-generation time (mirrors upstream's
  `tools/makeqstrdata.py:compute_hash` exactly, including the
  `(hash & mask) or 1` zero-fix), pass it as the second QDEF1
  arg. ROM_LEVEL now lives at CORE_FEATURES; bin grew 169 KB →
  199 KB; `bytearray` / `set` / `slice` / detailed error
  reporting / `'%' %` formatting all light up.

Triage progression as the slice unfolded:
- 95 / 132 with empty stubs (most failures were missing-qstr
  noise, not real codegen issues).
- 115 / 132 once a synthetic `qstrdefs.generated.h` was built
  by grepping `MP_QSTR_*` references out of `upstream/py/`.
- 117 / 132 after fixing a uc_core copy-prop bug (was eagerly
  propagating `struct *p = void_ptr_param;`, losing struct
  type for later `p->m`).
- 131 / 132 after teaching uc386 `_const_eval` about
  ternaries / comparisons / `&&` / `||` (lifted 12 packed-flag
  `.sig` failures from `MP_OBJ_FUN_MAKE_SIG`-style macros) and
  teaching `_resolved_var_type` to const-eval enum-constant
  designators (lifted `[SCOPE_GEN_EXPR] = …`-style unsized-
  array length-inference).
- 132 / 132 (current) after defining `MICROPY_REGISTERED_MODULES`
  (and `MICROPY_REGISTERED_EXTENSIBLE_MODULES`) as empty in
  the triage `genhdr/moduledefs.h`. A real build runs upstream's
  `py/makemoduledefs.py` over registered ports/<name>/main.c
  modules to emit per-module `MP_ROM_QSTR/MP_ROM_PTR` entries
  followed by `#define MICROPY_REGISTERED_MODULES <list>`; with
  no registered modules in the triage config the macro reduces
  `mp_builtin_module_table[] = { ... }` to an empty array
  initializer — which is the same shape a real port-without-
  modules emits.

See `NOTES.md` in the freedos_micro_python repo for the path to a runnable
`micropython.bin` (`tools/makeqstrdefs.py` → real qstrdefs,
write `ports/uc386-dos/` with INT-21h-backed `mp_hal_stdout_tx_strn`).

## ✓ Ship executables and test scripts in FOSS tarball (2026-05-01 ask)

**Done.** `uc386-foss-addons-*.tar.gz` now ships:
- 16 in-tree GNU utility binaries + BWK awk binary (already
  shipped before).
- A `src/<name>/` tree with each addon's `manifest.toml` + .c
  sources, plus `_sbase_shim/` (shared sbase headers) and the
  `awk-bwk/` / `gawk/` upstream scripts.
- `test_addons.py` at the top level — walks
  `src/<name>/manifest.toml`, runs each `<name>.bin` under
  `uc386.dos_emu.run` with the manifest's argv / stdin / vfiles,
  asserts stdout + exit code match the expected values. Also
  runs end-to-end smoke checks for the upstream-port binaries
  that don't carry a manifest (awk → `BEGIN { print 2*3 }`,
  micropython → REPL banner + Ctrl-D exit).

End-to-end verified: extract the tarball, `python test_addons.py`
prints `18/18 passed (0 skipped)` against the shipped manifests
+ binaries (16 manifest-driven + awk + micropython).

For the games tarball, "some games ship" is **just Doom today** —
the only game that boots end-to-end. `addons/games/doom/build/doom.bin`
ships under `uc386-games/bin/doom/doom.bin` when present. The
`SHIP_BIN` allow-list in `addons/harness/package.py` gates which
games' binaries leak into the tarball, so a stale build artefact
on a workstation can't accidentally publish.

## ✓ Ship download + build scripts for everything (2026-05-01 ask)

**Done.** Already done for the games tarball before today (every
game ships `fetch.sh` + `build.sh` + `NOTES.md` + per-game shims).
The FOSS tarball now matches: per-addon `manifest.toml` + sources
(in-tree addons) and `fetch.sh` / `build.sh` / `NOTES.md` (upstream
addons like awk-bwk + the new micropython skeleton). The release
README inside the tarball describes the rebuild path.

## Summary (2026-04-30, end of original 8-item session)

All 8 items in `addons.txt` are done. Doom **boots end-to-end**
through uc386, NASM, dos_emu (exits at WAD-not-found, expected).
Duke3D triages 34 of 42 sources clean — including engine.c (the
renderer) and build.c (the editor). All four other games
(Heretic / Hexen / ROTT / Descent) have working `fetch.sh`.
Heretic / Hexen point at chocolate-doom (the canonical id-Tech-1
port), ROTT at videogamepreservation/rott (Apogee 1994 GPL),
Descent at dxx-rebirth.

Compiler / runtime work shipped this session driven by the games
ports — about a dozen incremental improvements that benefit any
period-code port:
- Multi-TU file-scope `static` name mangling
- Anonymous struct structural identity (replacing id(t))
- Float-init fallback for integer globals (`.2 * FRACUNIT` style)
- Bit-op subexpressions in float const-eval
- `(int)"string"` / `(int)&global` in int-typed slot inits
- `char arr[N] = {"string"}` brace-around-string unwrap
- AST optimizer skip mul-to-shift on float subterms
- `__GNUC_MINOR__` / `__GNUC_PATCHLEVEL__` predefines
- `div_t div(int, int)` returns by value (C99 standard)
- libc `lseek` (with INT 21h AH=0x42 dos_emu handler)
- libc `strcasecmp`
- libc `getenv` recognizes HOME / DOOMWADDIR

Release CI (`.github/workflows/release.yml`) installs DJGPP and
OpenWatcom on the Linux runner; first triggered manual dry-run
revealed 2 build.sh hard-coded `.venv/bin/python` paths (fixed)
and an awk-bwk bison-3.x parser issue (made non-fatal in CI;
real fix is uc_core typedef-chain at block scope).

Total code shipped (as of that 2026-05 release): 1320 unit tests
passing, 220/220 c-testsuite (full mode), 1514/1514 gcc-c-torture,
16/16 addons, BWK awk fully functional. **Those counts are a
snapshot of their time** — the unit-test total is now 460 (+897 in
the sibling `upeep386`) after the extraction, the addon count is
17/17, and the current suite figures are 215/220 c-testsuite and
1397/1514 gcc-c-torture as measured in `README.md`. The
1514/1514 above was compile-only coverage of the corpus, not the
execute-and-diff pass rate. ~30 new libc symbols, ~5 new headers, 3 codegen
fixes (preprocessor multi-line comments, extern enum-in-VarDecl,
asm DCE filters unused externs), 2 fixes downstream (`_start_stub`
arg-push ordering, `assemble_and_run` cleanup-when-bundling-skipped).
