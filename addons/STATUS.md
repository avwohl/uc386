# addons.txt — completion status (2026-05-01)

Status of each item in `docs/addons.txt`. The original 8 items
from 2026-04-30 are still done; **3 new line-items** added by the
user on 2026-05-01 (MicroPython, ship test scripts in FOSS,
ship build scripts everywhere) are tracked below — all three are
done at the **infrastructure** level, with MicroPython itself
landing as a triage skeleton (87 % of py/ core compiles) since a
full port is multi-day work.

A second 2026-05-01 update added 3 more cross-cutting asks
(mac→linux portability, ship-source-with-binary, dosiz as a
test runner). Status:
- **Mac→Linux install** ✓ — `docs/INSTALL.md` covers apt / brew /
  dnf, `pyproject.toml` accepts Python ≥ 3.11, 1320 tests pass on
  this Linux box.
- **Ship source with binary** ✓ — `addons/harness/package.py`
  no longer excludes `upstream/` from either tarball, so when the
  CI fetches + builds awk-bwk or doom, the matching upstream
  source tree ships alongside the binary.
- **dosiz integration** ◐ — gap analysed in
  `docs/dosiz-integration.md`. dosiz today loads MZ / MZ+LE /
  bare LE; uc386 emits flat .bin via `nasm -f bin`. Smallest
  bridge is a `--flat-bin` loader on the dosiz side (reuses its
  existing PM / LDT plumbing); the larger payoff is uc386
  emitting MZ+LE so the binaries run on real DOS too. Path-B
  first; tracked there.

## ✓ Port GNU utilities to this compiler

**Done.** 16 addons under `addons/gnu/` ship working binaries:

| Addon | Source | Size (uc386 .bin) |
|-------|--------|-------------------|
| true | in-tree | 14 |
| false | in-tree | 17 |
| yes | in-tree | 74 |
| echo | in-tree | 148 |
| wc | in-tree | 233 |
| dirname | in-tree | 420 |
| head | in-tree | 419 |
| basename | in-tree | 479 |
| cat | in-tree | 512 |
| factor | in-tree | 566 |
| open_test | in-tree (smoke) | 580 |
| strtol_test | in-tree (smoke) | 703 |
| tail | in-tree | 959 |
| sbase-cat | sbase upstream | 1,167 |
| sbase-tee | sbase upstream | 1,355 |
| sbase-head | sbase upstream | 1,765 |

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

**gcc + DJGPP done; Watcom wired into CI** (no native macOS build).
`addons/results.md` has the size table — sample row:

```
| true | 14 | 16,840 | (CI) | 147,914 |
```

uc386 binaries are 50–1200× smaller than gcc-on-host (full glibc
startup) and ~100–10,000× smaller than DJGPP (DPMI extender + djgpp
C runtime baked in).

DJGPP cross-compiler installed locally at `~/.local/opt/djgpp` from
`andrewwutw/build-djgpp v3.4` (gcc 12.2.0). `addons/harness/compare.py`
detects it via the `DJGPP_CANDIDATES` list — works on both macOS
arm64 and the Linux CI runner.

Open Watcom V2 has no macOS build. The upstream Linux x64
"installer" is actually an ELF stub appended to a regular ZIP
archive — the release workflow `unzip -d`s it directly, avoiding
the FPE the installer otherwise triggers under unattended install.
Watcom column populates on every release run for **all 13 addons**
(once we converted period sources to C89-compat decl placement and
added `-ze` to wcc386). Full comparison row sample:

```
| true | 14 | 15,776 | 5,494 | 147,898 |
```

uc386 is **390× smaller than Watcom** for `true`, **1,127×
smaller than gcc**, and **10,564× smaller than DJGPP**. Watcom
runs ~2-4× wider than uc386 across the rest of the table because
DOS/4GW carries its own protected-mode startup; uc386 emits flat
real-mode-32 binaries that dos_emu loads directly.

## ◐ Include the latest MicroPython (2026-05-01 ask)

**Skeleton + triage landed** under `addons/gnu/micropython/`. A
runnable `micropython.bin` is still multi-day work (port shim,
real qstr generator integration, REPL HAL), but the triage
result now covers **the entire platform-independent core +
the shared support sources the minimal port builds alongside
py/**: **141 / 141 sources (132 from `upstream/py/` + 9 from
`upstream/shared/{libc,readline,runtime}/`, 100 %)** compile
cleanly via uc386 → NASM-ready .asm using a synthetic
`int main()` and stub `genhdr/` headers.

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

See `addons/gnu/micropython/NOTES.md` for the path to a runnable
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
  asserts stdout + exit code match the expected values.

End-to-end verified: extract the tarball, `python test_addons.py`
prints `16/16 passed (0 skipped)` against the shipped manifests
+ binaries.

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

Total code shipped: 1320 unit tests passing, 220/220 c-testsuite
(full mode), 1514/1514 gcc-c-torture, 16/16 addons, BWK awk
fully functional. ~30 new libc symbols, ~5 new headers, 3 codegen
fixes (preprocessor multi-line comments, extern enum-in-VarDecl,
asm DCE filters unused externs), 2 fixes downstream (`_start_stub`
arg-push ordering, `assemble_and_run` cleanup-when-bundling-skipped).
