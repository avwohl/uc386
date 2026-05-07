# Draft DOSBox-X bug report

This is a draft bug report for upstream DOSBox-X
(https://github.com/joncampbell123/dosbox-x). The findings below
were uncovered while debugging why a 484 KB PMODE/W-bound
MicroPython binary failed under DOSBox-X but ran cleanly under
classic DOSBox 0.74-3.

We have NOT isolated a minimal C-level reproducer yet; the bug
report below describes the observation and points at our existing
diagnostic infrastructure for anyone reproducing it.

---

## Title

PMODE/W: `INT 21h AH=0x40` writes NUL bytes (instead of the
buffer at EDX) when called from inside a >256 KB code object —
works under classic DOSBox 0.74-3, fails under DOSBox-X

## Environment

- DOSBox-X version: 2026.05.02 SDL2 (Homebrew bottle on macOS;
  also reproduced on `apt install dosbox-x` in the GitHub
  Actions Ubuntu 22.04 runner)
- Configuration:
  ```ini
  [cpu]
  cputype = pentium
  core    = normal
  cycles  = max
  [dosbox]
  memsize = 16
  ```
- DOS extender: PMODE/W v1.33 (bundled by Open Watcom V2's
  `wlink system pmodew option stub=$WATCOM/binw/pmodew.exe`)

## Observed behavior

- The binary's bridge stub (in the .obj at the START of obj 1)
  successfully calls `INT 21h AH=0x40` with `BX=1`, `EDX=<flat
  addr in obj 2>`, `ECX=N` and the bytes from EDX appear in the
  output file (DOS shell `>>` redirect). Multiple such calls
  work end-to-end, including writes from a stack-arg `_write`
  clone and from an indirect-call to user.obj's `_write`.
- The same binary's `main()` (in user.obj at runtime VA
  ~0x1c36a4) calls user.obj's `_write` (at VA ~0x1c6573) with
  identical args. Inside `_write`, `INT 21h AH=0x40` runs with
  the same registers (BX=1, EDX=same buffer addr, ECX=18). But
  the bytes written to the redirected file are NUL (0x00), not
  the expected ASCII string. Subsequent writes from `_main`
  similarly produce NULs and eventually the program hits an
  invalid-opcode exception (DOSBox-X log: `CPU:Illegal
  Unhandled Interrupt Called 6`).

## Comparison with classic DOSBox 0.74-3

The IDENTICAL binary, run under `apt install dosbox` (DOSBox
0.74-3) with the same `cputype=pentium core=normal cycles=max
memsize=16` config, prints the expected output:

```
[mp-main-entered]
[mp-before-mp-init]
[mp-after-mp-init]
[mp-before-repl]
MicroPython uc386-triage on 2026-05-01; uc386-dos with i386
Type "help()" for more information.
>>> import lwip, uc386_net
>>> lwip.reset()
```

So the DOSBox-X-specific bug only manifests for some calls to
`INT 21h AH=0x40` from within the binary. Whatever the
triggering condition is (caller VA, call depth, code-object
size, instruction-cache state — we haven't pinned it down), it
doesn't fire under classic DOSBox.

## Reproduction (current state of art)

The diagnostic infrastructure lives at
https://github.com/avwohl/uc386 in
`addons/gnu/micropython/dosbox-x-rig/` and
`addons/harness/exe.py`. The CI workflow at
`.github/workflows/mp-rig.yml` reproduces the bug end-to-end:

1. Build `micropython.bin` via uc386 (a pure-Python C23
   compiler — ~14 min on M1, ~85 min on a CI runner).
2. Bind it into MP.EXE with PMODE/W via Open Watcom V2's
   wlink (`addons/harness/exe.py`).
3. Run under DOSBox-X with the rig's config — observe NUL
   bytes after `[bridge-pre-call-main]` in `RIG.LOG`.
4. Run the SAME MP.EXE under classic DOSBox 0.74-3 (the
   "Run MP.EXE under classic DOSBox 0.74-3" step in the same
   workflow) — observe the full MicroPython REPL banner in
   `MPDOS.LOG`.

CI run `25479809993` shows both outputs side-by-side:
https://github.com/avwohl/uc386/actions/runs/25479809993

A successful CI run downloads MP.EXE as a build artifact, so
you can pull it and run it locally without building MicroPython
from scratch. The bridge stub embedded in MP.EXE prints
diagnostic markers `[bridge-entered]`, `[bridge-pre-call-main]`,
`[main+0]=000004c8`, `[main+7]=000227a3`, `[str-bytes]=2d706d5b`,
etc. — each marker confirms a specific bit of execution
reached that point with expected register/memory state. The
last-seen marker pinpoints where execution diverges.

## What's been ruled out

Diagnostic markers across 30+ iterations of the rig confirmed:
- LE FIXUP records ARE applied at runtime (`[main+7]=000227a3`
  is the section-relative offset 0xc7a3 + obj 2's runtime base).
- The data section IS loaded correctly (`[str-bytes]=2d706d5b`
  is little-endian "[mp-").
- `_main`'s instruction stream IS intact at runtime.
- The `call rel32` from `_main` lands at the correct `_write`
  address (`[write_addr]=001c6573`, bytes there match `_write`'s
  prologue exactly).
- Direct `INT 21h AH=0x40` from the bridge with the same args
  works (`[direct-write]=[mp-main-entered]`).
- A bridge-side clone of `_write` with cdecl args works
  (`[stack-write]=[mp-main-entered]`).
- Calling user.obj's `_write` directly from the bridge via
  indirect call works (`[user-write]=[mp-main-entered]` from
  CI run `25477796902`).

The ONE single difference between a working call to user.obj's
`_write` and the failing `_main` → `_write` call is the runtime
address of the CALLER (or, equivalently, the return address
on the stack at the time of `INT 21h`). We have not been able
to identify what about that difference triggers the bug from
outside DOSBox-X — interactive debugging from inside the
emulator would be the next step.

## Why this matters

This bug blocks a multi-MB DOS application (the uc386
MicroPython port) from running under DOSBox-X, even though it
runs cleanly under classic DOSBox and is expected to run on
real DOS hardware. The DOSBox-X rig was set up specifically to
test the binary's NE2000+SLIRP networking path under emulation,
since classic DOSBox doesn't have NE2000 support — so the bug
forces a fall-back to either real hardware or QEMU+SeaBIOS for
network validation, which is a significant testing-cost regression.
