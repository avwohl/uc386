# Draft DOSBox-X bug report

Draft for upstream DOSBox-X
(https://github.com/joncampbell123/dosbox-x). Findings below were
uncovered while debugging why a 484 KB PMODE/W-bound MicroPython
binary failed under DOSBox-X but ran cleanly under classic DOSBox
0.74-3 and QEMU+FreeDOS.

We have a reliable reproducer (the binary plus a specific
[autoexec] layout) but have NOT yet isolated a minimal C-level
repro — see "Bisect" below.

---

## Title

PMODE/W: `INT 21h AH=0x40` writes zero bytes when called from a
specific call-site context (deep call chain into a >256 KB code
object), with autoexec-content sensitivity — works under classic
DOSBox 0.74-3 and FreeDOS, fails under DOSBox-X

## Environment

- DOSBox-X versions reproduced on:
  - `2026.05.02 SDL2` (Homebrew bottle, macOS Darwin 25.4.0)
  - `apt install dosbox-x` on Ubuntu 22.04 GitHub Actions runner
- Configuration (minimal trigger):
  ```ini
  [cpu]
  cputype = pentium
  core    = normal
  cycles  = max
  [dosbox]
  memsize = 16

  [autoexec]
  mount C /path/to/dir
  C:
  MP.EXE > MPOUT.TXT
  exit
  ```
- DOS extender: PMODE/W v1.33 (bundled by Open Watcom V2's
  `wlink system pmodew option stub=$WATCOM/binw/pmodew.exe`)
- CPU emulator: `core=normal` (interpreter)

## Reliable reproducer

1. Pull `MP.EXE` from a recent
   [`mp-rig.yml`](https://github.com/avwohl/uc386/actions/workflows/mp-rig.yml)
   CI run's `mp-rig-artifacts` (484211 bytes; built via uc386 +
   Open Watcom V2 wlink).
2. Save the dosbox-x.conf above into a directory containing
   MP.EXE.
3. Run: `dosbox-x -silent -exit -conf dosbox-x.conf`
4. Inspect `MPOUT.TXT`.

**Expected** (classic DOSBox / QEMU+FreeDOS / real hardware): MPOUT.TXT
is 657 bytes ending at `>>> ` (the MicroPython REPL prompt):

```
[bridge-entered]
[bridge-argv-done]
[bridge-pre-jump]
[bridge-post-fpu]
... (15 more diagnostic markers from MP.EXE's bridge stub)
[bridge-pre-call-main]
[mp-main-entered]
[mp-before-mp-init]
[mp-after-mp-init]
[mp-before-repl]
MicroPython uc386-triage on 2026-05-01; uc386-dos with i386
Type "help()" for more information.
>>>
```

**Actual** under DOSBox-X: MPOUT.TXT is exactly 481 bytes,
truncated immediately after `[bridge-pre-call-main]\n`. The
`[mp-main-entered]\n` that should follow does NOT appear.

The `[mp-main-entered]\n` is produced by MicroPython's `_main`
calling a tiny `_write(fd, buf, count)` wrapper that does
`INT 21h AH=0x40 BX=1 EDX=buf ECX=count`. Earlier in the bridge
stub, IDENTICAL `INT 21h AH=0x40` calls (with the same fd, same
buffer, same count) succeed — see the `[direct-write]=[mp-main-entered]`,
`[stack-write]=[mp-main-entered]`, `[mainlike]=[mp-main-entered]`
markers in MPOUT.TXT.

## Bisect

Reproducible vs. non-reproducible variations of the autoexec
content (everything else identical). All under DOSBox-X
2026.05.02 with the config above; same MP.EXE binary.

| autoexec content                                    | Output            |
|-----------------------------------------------------|-------------------|
| `MP.EXE > X / exit`                                 | 481 bytes (BUG)   |
| `MP.EXE > X / dir / exit`                           | 481 bytes (BUG)   |
| `MP.EXE > X / echo Y / exit`                        | 481 bytes (BUG)   |
| `echo Z / MP.EXE > X / exit`                        | 481 bytes (BUG)   |
| `echo Z / MP.EXE > X / echo Y / exit`               | 657 bytes (OK)    |
| `MP.EXE > X` (no `exit`)                            | 657 bytes (OK)    |
| `dosbox-x -c "MP.EXE > X" -c "exit"` (no [autoexec])| 657 bytes (OK)    |

The bug fires when:

- MP.EXE is invoked through DOSBox-X's `[autoexec]` section, AND
- the autoexec contains `exit` somewhere after MP.EXE, AND
- there is at most one intervening command between MP.EXE and
  `exit`, AND
- there is no `echo` command before MP.EXE.

It does NOT fire when invoked via `-c` command-line flags or when
there is no `exit` (so DOS sits at the prompt after MP.EXE).

This pattern strongly suggests a state-dependent emulation defect
where the COMMAND.COM stack contents (which depend on the autoexec
command queue) overlap something PMODE/W relies on during INT 21h
reflection, OR where autoexec-driven shell processing changes the
real-mode SS:SP / segment layout that PMODE/W's `INT 21h AH=0x40`
path interacts with.

## What's been ruled out

Across 30+ in-binary diagnostic iterations:

- LE FIXUP records ARE applied at runtime (`[main+7]=000227a3` is
  the section-relative offset `0xc7a3` plus obj 2's runtime
  base).
- The data section IS loaded correctly (`[str-bytes]=2d706d5b` is
  little-endian "[mp-").
- `_main`'s instruction stream IS intact at runtime.
- The `call rel32` from `_main` lands at the correct `_write`
  address (`[write_addr]=001c6573`, bytes there match `_write`'s
  prologue: `55 89 e5 8b 5d 08 8b 55 0c 8b 4d 10 b4 40 cd 21`).
- Direct `INT 21h AH=0x40` from the bridge stub (low CS:EIP) with
  the same args works (`[direct-write]=[mp-main-entered]`).
- A bridge-side clone of `_write` with cdecl args works
  (`[stack-write]=[mp-main-entered]`).
- Calling user.obj's `_write` directly from the bridge via
  indirect call works (`[user-write]=[mp-main-entered]`).
- A bridge-side clone of `_main`'s exact byte sequence (enter / 3
  pushes / call) calling the bridge's `_write` clone works
  (`[mainlike]=[mp-main-entered]`).

The ONE single difference between a working bridge → `_write`
call and the failing `_main` → `_write` call is the call site
(and thus the return address on the stack at the time of
`INT 21h`).

## Minimization attempts

Standalone NASM-only repros at
[`addons/gnu/micropython/dosbox-x-rig/pmwbug/`](https://github.com/avwohl/uc386/tree/main/addons/gnu/micropython/dosbox-x-rig/pmwbug)
do NOT yet trigger the bug:

- `pmwbug.asm` — single-`_start` with two `INT 21h AH=0x40` call
  sites, one at low CS:EIP and one at high CS:EIP after 768 KB of
  NOP padding, sharing one `_writer` helper.
- `pmwbug_user.asm` — depth-3 call chain (`_start → _main →
  _write`) where `_main` and `_write` are byte-for-byte matches
  of MP.EXE's emitted prologs, padded to ~2 MB into the code
  segment, plus 1 MB of BSS, built via the same exe.py bridge
  shape (2-obj LE) as MP.EXE.

Neither minimization fires the bug under DOSBox-X with any
autoexec layout. So the trigger requires more than caller CS:EIP,
call depth, or code-object size. Outstanding hypotheses for what
specifically about MP.EXE is necessary:

- MicroPython's `mp_init` does FPU operations before `_main`'s
  marker write — FPU state may be relevant.
- MP.EXE has hundreds of cross-obj fixups; relocation table
  layout may matter.
- Stack-canary or specific page-table layout of the larger
  binary may be required.

## Source-side suspects

We did a read-only walkthrough of DOSBox-X's INT 21h dispatch
([`src/dos/dos.cpp`](https://github.com/joncampbell123/dosbox-x/blob/master/src/dos/dos.cpp)).
Three areas look suspicious:

1. **Per-INT-21h register-save preamble** at `dos.cpp:1054-1066`:
   ```cpp
   if (((reg_ah != 0x50) && ... && reg_ah<0x6c)) {
       DOS_PSP psp(dos.psp());
       psp.SetStack(RealMake(SegValue(ss),reg_sp-18));
       /* Save registers */
       real_writew(SegValue(ss), reg_sp - 18, reg_ax);
       real_writew(SegValue(ss), reg_sp - 16, reg_bx);
       /* ... 7 more 16-bit writes through reg_sp - 2 ... */
   }
   ```
   This writes 18 bytes to `(SS<<4) + reg_sp - 18`. If `reg_sp`
   is a 32-bit value and DOSBox-X's `reg_sp` is the truncated
   16-bit form, the write target depends on whether
   `cpu.stack.big` is set. For PMODE/W's mode-switched real-mode
   handler entry, this should be the real-mode 16-bit stack —
   but if PMODE/W keeps a 32-bit real-mode stack (`big`) and
   `(real_SS<<4) + (real_SP - 18)` happens to land in PMODE/W's
   transfer buffer, the AX/BX/CX/DX/SI/DI/BP/DS/ES values would
   overwrite the user's data BEFORE `MEM_BlockRead` at `:2152`
   reads it.

2. **AH=0x40 buffer read** at `dos.cpp:2152`:
   ```cpp
   MEM_BlockRead(SegPhys(ds)+reg_dx,dos_copybuf,towrite);
   ```
   Uses `reg_dx` (16-bit) and `SegPhys(ds)` (real-mode segment
   base). PMODE/W must arrange for `(DS<<4) + DX` to point at a
   real-mode-addressable bounce buffer copy of the user data.
   If PMODE/W's bounce-buffer setup is in any way dependent on
   call-site state (which DPMI hosts sometimes do for
   "fast-path" optimizations), a high-call-depth caller may
   trigger a different setup path.

3. **64K-truncation logic at `dos.cpp:2143-2148`**:
   ```cpp
   if (((uint32_t)towrite+(uint32_t)reg_dx) > 0xFFFFUL && (reg_dx & 0xFU) != 0U) {
       uint16_t nuwrite = (uint16_t)(0x10000UL - (reg_dx & 0xF));
       if (nuwrite > towrite) nuwrite = towrite;
       LOG_MSG("INT 21h WRITE warning: ...");
       towrite = nuwrite;
   }
   ```
   With `reg_dx & 0xF` non-zero AND
   `reg_dx + towrite > 0xFFFF`, this caps `towrite`. If
   PMODE/W's bounce buffer happens to be at a non-aligned offset
   near the end of a real-mode segment and the user's count
   pushes past the boundary, this could truncate to zero or near
   zero.

The "writes zero bytes" symptom in our reproducer is consistent
with EITHER (a) `dos_copybuf` containing zeros at `:2168`'s
`DOS_WriteFile` call (because the source was either zero-initialized
memory OR was overwritten by `:1057-1066`'s register save), OR
(b) `towrite` being capped to zero by the truncation logic at
`:2143-2148`.

## Diagnostic infrastructure

The
[`pmwbug` workflow](https://github.com/avwohl/uc386/blob/main/.github/workflows/dosbox-x-bug.yml)
runs MP.EXE under DOSBox-X / classic DOSBox / QEMU+FreeDOS in CI
and uploads all three outputs. The
[instrumented-build workflow](https://github.com/avwohl/uc386/blob/main/.github/workflows/dosbox-x-instrument.yml)
patches `dos.cpp` to log every AH=0x40 call's full register state
plus the first 16 bytes pulled into `dos_copybuf`, builds DOSBox-X
from source, and runs the failing case so the EXACT bytes /
addresses DOSBox-X observes during the failing call are captured.
A successful run of that workflow gives the upstream maintainer
the decisive data to localize the bug.

## Why this matters

This bug blocks a multi-MB DOS application (the uc386 MicroPython
port) from running under DOSBox-X, even though it runs cleanly
under classic DOSBox 0.74-3, FreeDOS-on-QEMU, and is expected to
run on real DOS hardware (FreeDOS in VMware on Windows is the
production target). The DOSBox-X rig was set up specifically to
test the binary's NE2000+SLIRP networking path under emulation,
since classic DOSBox doesn't have NE2000 support. The bug forces
a fall-back to either real hardware or QEMU+FreeDOS for network
validation, which is a meaningful testing-cost regression.

The fact that the bug is autoexec-content-sensitive also suggests
something subtle about COMMAND.COM stack / shell state interacting
with PMODE/W's INT 21h reflection — fixing it likely fixes a
broader class of DOS-extender bugs in DOSBox-X, not just this
one binary.
