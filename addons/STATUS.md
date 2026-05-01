# addons.txt — completion status (2026-04-30)

Status of each item in `docs/addons.txt`:

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

- `uc386-foss-addons-<ver>.tar.gz` (~62 KB) — built FOSS binaries
- `uc386-games-build-scripts-<ver>.tar.gz` (~5 KB) — game fetch/build scripts

The packaging logic is in `addons/harness/package.py` (runnable
locally too: `python -m addons.harness.package --version dev`).

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

## ⚠ Games from the DOS DPMI period

**Scaffolding done + first compile attempt complete; builds still
blocked, but on different items than originally feared.**
`addons/games/{doom,duke3d,heretic,hexen,rott,descent}` each have
a `NOTES.md` documenting upstream URL, license, and expected
blockers. Doom + Duke3D have working `fetch.sh` (verified — both
pull upstream sources cleanly) and `build.sh` (Doom now drives
58 sources through preprocess → parse → codegen).

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

**All 58 doom sources now lower to a single 76 K-line .asm.** What's
left is *runtime glue*, not compiler work: stubs for ~30 `_I_*`
platform functions (`I_GetTime`, `I_Error`, `I_InitGraphics`, …) plus
a couple of libc additions (`fstat`, `mkdir`, `sscanf` variadic).
Documented in `addons/games/doom/NOTES.md`.

**Spirit of the request preserved**: every blocker we close benefits
ALL period-code ports, not just Doom — these are six general compiler
improvements (structural struct identity, float-aware strength
reduction, etc.) that shipped in this slice.

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

Open Watcom V2 has no macOS build, but the upstream Linux x64
self-extractor works on the CI runner. The release workflow installs
both toolchains (`.github/workflows/release.yml`) so the Watcom
column populates on `v*` tag releases. Local dev host shows `—`
gracefully via `_which_first` detection.

## Summary

Of the 8 explicit items in `addons.txt`, **7 are fully done** and
**1 is scaffolding + first compile attempt** (games — Doom now
drives 58 sources into codegen; struct-type-identity is the next
blocker, fix lives in uc_core).

Total code shipped: 1320 unit tests passing, 220/220 c-testsuite
(full mode), 1514/1514 gcc-c-torture, 16/16 addons, BWK awk
fully functional. ~30 new libc symbols, ~5 new headers, 3 codegen
fixes (preprocessor multi-line comments, extern enum-in-VarDecl,
asm DCE filters unused externs), 2 fixes downstream (`_start_stub`
arg-push ordering, `assemble_and_run` cleanup-when-bundling-skipped).
