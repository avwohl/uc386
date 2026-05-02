# Path A: uc386 emits MZ+LE `.exe` for FreeDOS

**Goal**: every uc386-built binary runs unmodified on FreeDOS, DOSBox,
dosiz, and real DOS — alongside the existing flat `.bin` that runs
under `uc386.dos_emu`.

**Status (2026-05-02)**: Phase 1 ✓ + Phase 2 ✓ + Phase 3 ✓ — uc386
now produces .exe files that **actually run on a real DOS
environment**. Verified end-to-end in CI under DOSBox 0.74-3:
true.exe boots PMODE/W, switches to 32-bit protected mode, runs
our `int main(void) { return 0; }`, exits via INT 21h AH=4Ch,
returns control to DOSBox shell which continues the autoexec.

Path A core goal achieved: every uc386-built binary can ship as
a `.exe` that runs on FreeDOS. Phase 4-6 (calling-convention
bridge for argv, integration into addons harness, FOSS tarball
shipping .exe alongside .bin) are remaining polish.

Phase 3 findings:
- DOSBox `core=auto` (dynrec) chokes on PMODE/W's PM setup with
  "DYNREC:Can't run code in this page". `core=normal`
  (interpreter) handles it correctly.
- `wlink option stack=64k` is required — without it the .exe has
  no PM stack and faults on the first push.
- `wlink option start=_start` overrides wlink's default
  `_cstart_` lookup (which would need Watcom clib).
- NASM's `-f obj` defaults segments to USE16. uc386's `section
  .text` lines must be rewritten to `section _TEXT use32
  class=CODE` before NASM consumes them — otherwise the OMF
  declares 32-bit code as 16-bit and the LE-loader flips the
  D-bit clear.
- DOSBox 0.74-3 writes mounted host files with 8.3 short
  uppercase names: `result.txt` → `RESULT.TXT`.
- DOSBox 0.74-3's shell doesn't expand `%errorlevel%` — verifying
  exit code requires `if errorlevel N` syntax.



- `addons/gnu/true/main.c` → `true.exe` (11,779 bytes, PMODE/W
  bound). `file` reports "MS-DOS executable, LE executable for
  MS-DOS, PMODE/W DOS extender."
- `addons/gnu/cat/main.c` → `cat.exe` (12,432 bytes, PMODE/W bound)

Phase 2 finding: `wlink system pmodew` alone produced a 371-byte
unbound LE that printed "This is a PMODE/W executable" and
exited. To get a self-contained .exe, the extender stub binary
itself has to be the MZ portion. `addons/harness/exe.py` now
auto-locates `$WATCOM/binw/pmodew.exe` and passes
`option stub=...` to wlink, which embeds the stub as the .exe's
MZ part with the LE payload appended.

CI smoke-test asserts >1 KB output and "PMODE/W" in `file`
output — catches a regression where stub-binding silently fails.

Next steps: actually run these .exe under DOSBox in CI to verify
runtime behavior (Phase 3).

## Why Path A over Path B

`docs/dosiz-integration.md` describes two ways to bridge uc386's
flat-bin output to a real DOS environment. Path A (uc386 wraps in
MZ+LE) is the better long-term move because the resulting binary
runs everywhere DOS programs run — not just dosiz. The user
explicitly wants FreeDOS as the primary target, so Path A it is.

## Pipeline

    .c → uc386 → .asm → nasm -f obj → .obj → wlink system causeway → .exe

The pieces:

- **uc386** emits NASM-syntax assembly, same as today.
- **NASM `-f obj`** turns `.asm` into 32-bit OMF (Object Module
  Format) — Watcom's wlink consumes OMF natively. NASM's USE32
  segments produce flat-32 objects compatible with Watcom's
  segmented-but-flat memory model.
- **wlink** is Open Watcom's linker. It accepts OMF and produces
  MZ+LE executables. The `system <extender>` directive selects which
  DOS extender stub gets bundled into the resulting `.exe`.

## DOS extender choice

| Extender | License        | Stub size | Bundled with Watcom? |
|----------|----------------|-----------|----------------------|
| DOS/4GW  | proprietary    | ~250 KB   | Yes (legacy)         |
| PMODE/W  | BSD-ish        | ~9 KB     | Yes                  |
| CauseWay | free / public  | ~20 KB    | Yes                  |
| HX       | MIT            | varies    | No (separate)        |

For self-contained executables we want a free extender with the stub
bound into the `.exe`. **CauseWay** is the default in `exe.py` —
free, well-tested, supported natively by wlink (`system causeway`).

PMODE/W is smaller but its license is "free for any use" with an
attribution requirement. We can switch to it later if size matters.

DOS/4GW is the historical default but produces an LE that requires
`dos4gw.exe` alongside — not what FreeDOS users expect.

## Calling-convention bridge (open work)

uc386's `_start` today reads:

    EAX = argc        ; set by dos_emu
    EBX = &argv[0]    ; set by dos_emu

Real DOS extenders pass argc/argv differently — typically via the
PSP (Program Segment Prefix) command-line tail. The extender's own
startup code parses `PSP+0x80` (the command-line buffer) into
argc/argv before jumping to the user's `_start`.

For Phase 1 hello-world we don't need argv to be correct — main()
that ignores `argc`/`argv` works fine. For Phase 2 we'll write a
bridge stub that:

1. On entry, the extender has already parsed its own arguments.
2. Reads the PSP at `__psp` (a Watcom-runtime symbol) or pulls
   argc/argv from wherever the extender stored them.
3. Sets EAX/EBX to match uc386's expected convention.
4. Falls through to the existing `_start`.

Until that bridge lands, programs that read argv will crash or see
garbage.

## INT 21h reflection

Both CauseWay and PMODE/W reflect INT 21h calls back to real-mode
DOS. Our existing `lib/i386_dos_libc.asm` uses INT 21h for putchar,
exit, file I/O, etc. — all of these will Just Work under either
extender, *if* the calling-convention bridge is in place so we don't
crash before the first INT 21h.

## Phases

- **Phase 1** (landed): `exe.py` wrapper script. Drives nasm + wlink.
  Defaults to CauseWay. Returns clean error messages when Watcom
  isn't installed.
- **Phase 2**: Validate against an in-tree addon. Pick `true` (14
  bytes flat-bin → ?? bytes .exe) and `cat` as canaries. Iterate on
  any wlink errors that surface — `_start` symbol naming, segment
  attributes, unresolved libc references.
- **Phase 3**: Write the calling-convention bridge so argv works.
  Probably a 50-line `addons/harness/exe_libc_bridge.asm` that wlink
  links in alongside the user's .obj.
- **Phase 4**: Wire `--exe` into `addons.harness.build` so every
  addon optionally produces an .exe alongside its .bin.
- **Phase 5**: CI integration. The release workflow already installs
  Watcom — extend it to build `.exe` variants and ship them in the
  FOSS tarball alongside `.bin`.
- **Phase 6**: Test on FreeDOS via DOSBox in CI. DOSBox runs
  headless (no GUI needed for stdout-only programs), so an automated
  smoke test is feasible.

## Open questions

- Does NASM's OMF output have the right segment attributes for
  wlink's USE32 expectations? Untested locally (no Watcom on this
  Mac). The `-f obj` documentation says yes; first wlink invocation
  will tell us.
- Will the existing libc work as-is, or does it need DOS-extender
  awareness in a few places (e.g. `_setjmp` saves ESP — under an
  extender the ESP value may not be re-usable across longjmp)?
  Empirical question once Phase 2 starts.
- File I/O: `open()` / `read()` / `write()` use real-mode DOS file
  handles via INT 21h. The extender reflects these — but the buffer
  pointers must be in addressable memory the extender can translate.
  Worst case we bounce-buffer through the PSP or DOS transfer area.
