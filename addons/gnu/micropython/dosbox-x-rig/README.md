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

Suspected root causes (in priority order):

1. PMODE/W INT 21h passthrough for AH=0x40 (write) doesn't
   reach the redirected `RIG.LOG` even though smaller test
   programs (`echo.exe`, `factor.exe`) work via the same
   bridge under DOSBox 0.74-3.
2. `_pmodew_start` bridge doesn't initialize MP.EXE's 1.4 MB
   BSS (libc heap + MP static heap + globals) in a way the
   binary expects.
3. Some libc symbol the multi-TU MicroPython build references
   isn't provided by exe.py's bridge.

To distinguish (1) — a rig-wide bridge regression — from (2)/(3),
the CI workflow now drops `ECHOTEST.EXE` (a printf-only build of
`addons/gnu/echo`) into this directory and runs it as a baseline
before `MP.EXE`. If `hello dos rig` lands in `RIG.LOG` between
the `before-echo-baseline` / `after-echo-baseline` markers, the
INT 21h AH=40 passthrough works under this DOSBox-X config and
the issue is MP.EXE-specific (suspect 2 or 3). If it doesn't,
the problem is the rig itself (suspect 1) and `MP.EXE` was never
going to print regardless of its internals. The CI gate now
requires the baseline to print — without it, MP.EXE-runtime
diagnostics are meaningless.

Until that's diagnosed, dos_emu's emulator (the existing
`test_micropython_smoke.py` suite — 94/94 passing) remains the
authoritative validation that the binary works. The DOSBox-X
rig is shipped as the on-ramp to real-hardware testing.

## Source attribution

`NE2000.COM` is the Crynwr 11.4.3 NE2000 packet driver, MIT-style
distribution terms (see `COPYING.DOC` shipped alongside the source
in the original Crynwr package). Fetched from the Internet Archive
mirror at `lan-packet-drivers-for-ms-dos/drvlan/ne2000.zip`.
