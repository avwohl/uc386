# BWK awk (one-true-awk) port — status: **working** (2026-04-30)

**Upstream**: <https://github.com/onetrueawk/awk>
**License**: free / public-source — see `upstream/LICENSE`.
**Size**: ~6K LoC source; uc386 build → ~107 KB binary.

## Build

```sh
./fetch.sh    # downloads upstream master into upstream/
./build.sh    # bison + maketab on host, then uc386 multi-file compile
              # → build/awk.bin
```

`build.sh` requires bison and a host C compiler (only for the
parser-table generators that ship inside upstream). All actual
program logic compiles via uc386.

## Run

```python
from uc386.dos_emu import run
from pathlib import Path

res = run(Path("addons/gnu/awk-bwk/build/awk.bin"),
          argv=["awk", '{ print NR, $0 }'],
          stdin_bytes=b"one\ntwo\nthree\n",
          timeout_seconds=10,
          instruction_limit=200_000_000)
print(res.stdout)
# → "1 one\n2 two\n3 three\n"
```

## What works

Verified via `addons/gnu/awk-bwk/build.sh && python harness`:

- BEGIN / END blocks
- Pattern matching (regex /pattern/)
- Field access (`$0`, `$1`, `NF`, `NR`)
- Arithmetic (int, float, modulo)
- String functions (length, toupper, tolower, split, sprintf, sub, gsub)
- Math functions (sqrt, exp, log, sin, cos)
- Control flow (for, while, if/else)
- Associative arrays
- `--version` flag

Stdin reading **works** (after the FILE* sentinel fix that lifted
stdin off NULL — see `docs/changes.md`).

## What doesn't work yet

- `popen` / `pclose` are stubs that always fail; awk's `getline` from
  a piped command always errors. (No pipe API on DOS without a shell
  layer — see `i386_dos_libc.asm:4219`.)
- Hex floats in input data (strtod doesn't parse `0x1p2`).

Fixed since this was written:

- ~~`system(cmd)` returns -1~~ — `system` is now a real
  implementation: resolves `COMSPEC` (falling back to a default
  path), builds a ` /C <cmd>` tail, and EXECs via INT 21h AH=0x4B,
  reading the child's code back with AH=0x4D.
- ~~exponent in `strtod` lexed but not applied~~ — verified applied:
  `strtod("1e3")` is 1000.0, `strtod("2.5e2")` is 250.0, and
  `strtod("1e-2")` is 0.01.

## Reproducing the build from scratch

```sh
cd addons/gnu/awk-bwk
./fetch.sh
./build.sh
ls -la build/awk.bin   # should be ~107 KB
```

To run the awk-feature battery: see the test harness invocation in
`docs/changes.md` (the BWK awk slice notes the exact commands).
