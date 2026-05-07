# DOSBox-X test rig for the MicroPython binary

Boots DOSBox-X with NE2000 + SLIRP networking and runs the
`MP.EXE`-bound MicroPython under it. Validates the binary's full
real-DOS path:

- IVT scan via DPMI INT 31h fn 0x0200 finds the Crynwr driver
- `pktdrv_init` allocates a real-mode callback via DPMI fn 0x0303
- AH=02 access_type registers the trampoline
- AH=06 fetches the NIC's MAC
- DHCP from SLIRP gives `10.0.2.15` (same network layout the
  dos_emu sim defaults to, so `eth_status()` should be identical
  in both environments)

## What the rig contains

| File             | Role                                                  |
|------------------|-------------------------------------------------------|
| `dosbox-x.conf`  | DOSBox-X config: NE2000 at I/O 0x300 IRQ 9 + SLIRP    |
| `autoexec.bat`   | Loads `NE2000.COM` at INT 0x60, runs baseline + `MP.EXE` |
| `fetch.sh`       | Downloads `NE2000.COM` from archive.org (idempotent)  |
| `run.sh`         | Launches DOSBox-X with the config                     |
| `NE2000.COM`     | Crynwr 11.4.3 packet driver (fetched on first run)    |
| `ECHOTEST.EXE`   | Printf-only baseline (built from `addons/gnu/echo`)   |
| `MP.EXE`         | Watcom-bound MicroPython (must be built on Linux)     |
| `RIG.LOG`        | DOSBox-X session output (created on each run)         |

## Quick check (no MP.EXE needed)

```sh
brew install dosbox-x          # macOS
./fetch.sh                     # downloads NE2000.COM
./run.sh                       # boots DOSBox-X, autoexec runs

cat RIG.LOG
# Expected:
#   --- DOSBox-X test rig ---
#   Packet driver for NE2000, version 11.4.3
#   Packet driver software interrupt is 0x60 (96)
#   Interrupt number 0x9 (9)
#   I/O port 0x300 (768)
#   My Ethernet address is XX:XX:XX:XX:XX:XX
#   MP.EXE not found in C:
#   --- Test rig finished ---
```

That confirms DOSBox-X starts cleanly, NE2000.COM loads, and the
packet driver is installed at INT 0x60 — exactly where
`pktdrv_uc386dos.c::pktdrv_detect()` scans for the "PKT DRVR"
signature.

## Full run (with MP.EXE)

`MP.EXE` is the Watcom-linked MicroPython binary produced by
`addons/harness/exe.py`. Open Watcom doesn't have a macOS build,
so the `.exe` step has to happen on Linux:

```sh
# On a Linux dev box (or a CI runner):
cd /path/to/uc386
python -m addons.harness.exe \
    addons/gnu/micropython/build/micropython.bin \
    -o addons/gnu/micropython/dosbox-x-rig/MP.EXE
```

Then copy the `.exe` back to the Mac and re-run `./run.sh`. The
log will continue past the "MP.EXE not found" branch into the
MicroPython REPL banner + DHCP exchange.

## How DOSBox-X's SLIRP matches dos_emu_netsim

Both pin the same network layout (it's the QEMU/SLIRP convention):

| Field         | Address          |
|---------------|------------------|
| Network       | 10.0.2.0/24      |
| Host gateway  | 10.0.2.2         |
| DNS server    | 10.0.2.3         |
| Guest (DHCP)  | 10.0.2.15        |

So a binary that DHCPs to 10.0.2.15 against the dos_emu sim should
DHCP to the same address against DOSBox-X — useful for "same code,
two environments" sanity.

## Known limitation: MP.EXE runtime under PMODE/W

The `.github/workflows/mp-rig.yml` CI run currently gates on:

- `build_port.sh` produces `micropython.bin`
- `addons/harness/exe.py` binds `MP.EXE` cleanly
- DOSBox-X boots the rig and `NE2000.COM` reports its install
  banner with INT 0x60

What's *not* yet validated end-to-end: MP.EXE's actual runtime
output. With cycles=max + 180s timeout under DOSBox-X 2024.03,
MP.EXE produces no observable bytes on stdout — neither the REPL
banner nor any post-`eth_init` print. The autoexec markers
(`before-mp`, `after-mp`) confirm execution stops between them,
but whether MP.EXE crashes silently in startup or hangs on stdin
parsing isn't pinned down yet.

### Bisect — what we know (as of 2026-05-07)

A series of progressive triage iterations under the rig narrowed
the failure to a specific layer:

1. **Bridge stub runs cleanly.** Markers `[bridge-entered]` →
   `[bridge-argv-done]` → `[bridge-pre-jump]` → `[bridge-post-fpu]`
   → `[bridge-pre-bss-zero]` → `[bridge-post-bss-zero]` →
   `[bridge-diag-stub]` → `[bridge-post-diag]` →
   `[bridge-pre-call-main]` all print under MP.EXE.
2. **Codegen `_start`'s rep stosb on a 280 KB BSS range silently
   aborts** the program under PMODE/W — `[bridge-pre-bss-zero]`
   prints but `[bridge-post-bss-zero]` doesn't when it's enabled.
   Workaround: skip the loop entirely (PMODE/W's loader already
   zero-fills BSS at load time per the LE bss_size header).
3. **In-bridge diag stub `_diag_main()` works.** `[bridge-diag-stub]`
   and `[bridge-post-diag]` both print. So "call C-style function
   with enter-prolog → libc INT-21h marker → ret" plumbing works.
4. **`call _main` reaches main() but main()'s `write(1, "[mp-main-entered]", 18)`
   produces NUL bytes instead of the marker string**, then
   execution hits "Illegal/Unhandled opcode FFFF" → INT 6 →
   handler loops forever.
5. **Root cause: LE FIXUP records aren't being applied to absolute
   address references in MP.EXE under PMODE/W.** The push imm32
   for the marker string in MP.EXE's `_main` contains the
   data-section *offset* (0x0000c7a3), not the absolute VA
   (0x7c7a3 = data section base 0x70000 + 0xc7a3). At runtime,
   `_write` reads from address 0xc7a3 (zero-filled / unmapped),
   producing NUL bytes. Subsequent writes to other globals via
   `mov [imm32], eax` corrupt low memory, and eventually the
   CPU jumps to garbage code.
6. **ECHOTEST.EXE works through the same exe.py + wlink + PMODE/W
   path**, so the relocation problem is specific to large
   binaries or to some asm pattern uc386 emits for MicroPython
   that NASM/wlink mishandles.

### Where the trail ends

The bug lives somewhere in: NASM's OMF FIXUP record generation
for the multi-TU MicroPython asm, OR wlink's LE FIXUP packaging,
OR PMODE/W's LE loader fixup application for objects > 256 KB.
Diagnostic next steps:

- Compare ECHOTEST.EXE's FIXUP records to MP.EXE's at the same
  patterns (e.g. `push <data-symbol>` and `mov [<data-symbol>], eax`).
  Both should generate `src=07 flags=00` (32-bit offset, internal,
  non-additive) records pointing to obj 2 (data) with appropriate
  target_off.
- Try the `dos4g` extender (`exe.py --extender dos4g`) instead of
  PMODE/W to isolate whether the issue is loader-specific.
- Try `wdump MP.EXE` (Linux Watcom binary) to inspect FIXUP
  records authoritatively.
- Audit uc386's codegen for any pattern that would generate asm
  NASM can't represent in OMF (e.g. unusual section-relative
  addressing modes).

The CI's "Best-effort MP.EXE runtime checks" step lists every
expected marker so each iteration shows exactly how far execution
got — a definitive "is this still the same bug" oracle.

### Earlier hypothesis log (rejected/superseded)

1. ~~PMODE/W INT 21h passthrough for AH=0x40 doesn't reach the
   redirected RIG.LOG.~~ Rejected: ECHOTEST.EXE prints `hello dos rig`
   via the same path.
2. ~~`_pmodew_start` bridge doesn't initialize the 1.4 MB BSS
   correctly.~~ Rejected: PMODE/W's loader zero-fills BSS at load
   time per the LE bss_size header. The codegen rep stosb is
   redundant — and runs into a separate failure on multi-100KB
   ranges (suspect (2) above), which we now bypass.
3. ~~Some libc symbol the multi-TU MicroPython build references
   isn't provided by exe.py's bridge.~~ Rejected: linker would
   have erred on an unresolved external; the link is clean.

Until that's diagnosed, dos_emu's emulator (the existing
`test_micropython_smoke.py` suite — 94/94 passing) remains the
authoritative validation that the binary works. The DOSBox-X
rig is shipped as the on-ramp to real-hardware testing.

## Source attribution

`NE2000.COM` is the Crynwr 11.4.3 NE2000 packet driver, MIT-style
distribution terms (see `COPYING.DOC` shipped alongside the source
in the original Crynwr package). Fetched from the Internet Archive
mirror at `lan-packet-drivers-for-ms-dos/drvlan/ne2000.zip`.
