# Path A: uc386 emits MZ+LE `.exe` for FreeDOS

**Goal**: every uc386-built binary runs unmodified on FreeDOS, DOSBox,
dosiz, and real DOS — alongside the existing flat `.bin` that runs
under `uc386.dos_emu`.

**Status (2026-08-13): done.** All phases landed. Every in-tree GNU
addon builds to a self-contained `.exe`; `true.exe` / `false.exe` boot
under DOSBox in CI with correct errorlevels, `argv_probe.exe` parses a
real command line, and file I/O goes through genuine DOS handles.

## Pipeline

    .c → uc386 → .asm → nasm -f obj → .obj → upyle → MZ+LE .exe

- **uc386** emits NASM-syntax assembly, same as for `.bin`.
- **NASM `-f obj`** turns `.asm` into 32-bit OMF. uc386's `section
  .text` lines are rewritten to `section _TEXT use32 class=CODE`
  first — otherwise the OMF declares 32-bit code as 16-bit and the
  LE loader flips the D-bit clear.
- **[upyle](https://github.com/avwohl/pyle)** is our pure-Python
  OMF→MZ+LE linker. It consumes the codegen `.obj` plus the bridge
  `.obj` and binds the extender stub as the MZ portion with the LE
  payload appended. **No Open Watcom required** — this is why the
  `.exe` path works on macOS.

`wlink` (Open Watcom) is still supported and is required *only* for
the `causeway` and `dos4g` extenders. It is not on the default path.

Build one:

    python -m addons.harness.exe addons/gnu/true/main.c -o true.exe

Needs `nasm` on `PATH` and `pip install upyle`.

## DOS extender choice

`--extender` selects the stub. Floors below are the measured size of a
minimal `.exe` (stub + LE headers + a trivial program), not the bare
stub:

| Extender | License       | `.exe` floor | Linker  | Disk I/O on real DOS |
|----------|---------------|-------------:|---------|----------------------|
| DOS/32A  | free          |    ~32.8 KB  | upyle   | **yes** (default)    |
| PMODE/W  | free (attrib) |    ~16.8 KB  | upyle   | **no** — see below   |
| CauseWay | free/public   |          —   | wlink   | untested here        |
| DOS/4GW  | proprietary   |          —   | wlink   | needs `dos4gw.exe`   |

**DOS/32A is the default** (since `58c1f79`), and `upyle` ships a
pre-bound DOS/32A stub so no `--stub-binary` flag is needed.

The default costs ~16 KB against PMODE/W, and that is deliberate:
**PMODE/W's real-mode call path hangs on any DOS call that touches a
physical sector**, so a PMODE/W build cannot do disk I/O on real DOS.
It silently produced binaries that looked fine under DOSBox and then
locked up on hardware. PMODE/W remains available via
`--extender=pmodew` for programs that only write to stdout, where it
is the smaller correct answer.

## Runtime details

- **PM stack: 256 KB** (`_PM_STACK_BYTES` in `exe.py`), passed
  explicitly to `upyle.write_le(stack_size=...)` rather than relying
  on upyle's default. uc386 does not pin upyle, and an older upyle
  defaults to 64 KB — a runtime whose stack guard is sized for 256 KB
  would then overrun it. `exe.py` warns if upyle is too old to accept
  the argument.
- **Bridge stub** (`_pmodew_start` in `exe.py`, linked into every
  build — the name is historical, it runs under both extenders):
  - **stdio fds**: libc's `_stdout = 0xF1` is a `dos_emu` sentinel;
    real DOS AH=0x40 wants raw 0/1/2. The bridge redefines them with
    real handles and the sentinel definitions are stripped from the
    codegen `.obj` at build time.
  - **argv**: the extender puts the **PSP selector in ES** at entry.
    The bridge reads `[es:0x80]` for the command-line length and
    `[es:0x81..]` for the tail, copies to a flat-DS buffer, and
    tokenizes on space/tab/CR. `argv[0]` is recovered from the
    environment block when available, else a `"program"` placeholder.
- **Startup markers** are off by default (`abec4d4`) because they
  print on the program's own stdout. Enable with `--bridge-markers`
  or `UC386_BRIDGE_MARKERS=1` when debugging entry.

## Engineering notes (historical)

Kept because they cost real time to discover.

**The argv fix took six failed attempts.** The rule is: **USE the ES
the extender hands you; do not load a different selector into ES.**
Earlier experiments (`mov es, 0x21`, `mov fs, fresh-DPMI-selector`)
overwrote it and broke the extender's INT 21h reflection. The bridge
now reads `[es:0x80]` at the very top, before anything can clobber ES.
Confirmed against OpenWatcom's `bld/clib/startup/a/cstrt386.asm`,
which uses `mov esi,es` to find the PSP.

Dead ends, for the record:

- DPMI INT 31h AX=0x06 (Get Segment Base) returned linear=0.
- A flat-DS read at `PSP*16` returned zeros — DS doesn't map
  real-mode memory.
- INT 21h AH=0x62 returned a *different* PSP segment (the extender's
  internal protected-mode PSP), which has no command line at 0x80.

DOSBox quirks hit along the way:

- `core=auto` (dynrec) chokes on PMODE/W's PM setup with
  "DYNREC:Can't run code in this page". `core=normal` works.
- DOSBox 0.74-3 writes mounted host files with 8.3 uppercase names:
  `result.txt` → `RESULT.TXT`.
- Its shell doesn't expand `%errorlevel%`; verify exit codes with
  `if errorlevel N`.
- Naming a test `echo.exe` is a trap — the shell's `echo` builtin
  intercepts `echo.exe hello` as `echo` + `exe hello`. Use `myecho`.

A codegen bug surfaced by the argv probe — `if (v == 0) {...} while
(v > 0)` emitting `jle` with no preceding `cmp`, relying on stale
flags — **has since been fixed**; the loop now emits `test eax, eax`
before the branch.

## Why Path A over Path B

`docs/dosiz-integration.md` describes two ways to bridge uc386's
flat-bin output to a real DOS environment. Path A (uc386 wraps in
MZ+LE) is the better long-term move because the resulting binary runs
everywhere DOS programs run — not just dosiz. FreeDOS is the primary
target, so Path A it is.
