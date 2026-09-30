# Size — measured, not asserted

The "tiny output" claim, checked against the period reference
compiler instead of asserted. Every column below was **reproduced
on one macOS/arm64 host** by `python -m addons.harness.compare`
(Open Watcom V2 has no native macOS build, so its DOS-hosted
`wcc386`/`wlink` run under DOSBox-X via `addons/harness/
watcom_dosbox.py`; DJGPP is the gcc-12.2 osx cross under Rosetta).
Bytes of the on-disk executable; full table in
[`addons/results.md`](../addons/results.md):

| program | uc386 .bin | uc386 .exe | Watcom | DJGPP |
|---------|-----------:|-----------:|-------:|------:|
| true    |         18 |     32,847 |  5,420 | 147,914 |
| echo    |        264 |     32,911 | 11,286 | 150,212 |
| factor  |      2,022 |     32,981 | 20,538 | 179,614 |
| wc      |      1,861 |     32,992 | 20,158 | 179,092 |

Reading this honestly:

- **`.bin` is not a DOS program.** It has no MZ header and runs
  only under `uc386.dos_emu`/a custom loader. It is the right
  metric for *codegen+DCE tightness* (and there uc386 is in a
  class of its own — tens of bytes), but it is not what you ship.
- **`.exe` is what you ship**, and it carries a **~32.8 KB DOS/32A
  extender floor** — every `.exe` in the table is that floor plus a
  few hundred bytes of program. Against that real-DOS artifact,
  **Open Watcom is smaller: ~6× on `true`, ~1.6× on `wc`/`factor`**
  (its DOS/4GW clib + mature linker beat our extender floor); the
  two converge as real code grows. uc386 beats **DJGPP ~4.5–5.5×**.
- **The floor is a deliberate correctness trade.** `--extender=pmodew`
  halves it (~16.8 KB), but PMODE/W's real-mode call path hangs on
  any DOS call that touches a physical sector, so a PMODE/W build
  cannot do disk I/O on real DOS. DOS/32A is the default because a
  working `.exe` beats a smaller broken one; PMODE/W stays available
  for programs that only touch stdout.
- So: uc386's *code generation* is extremely compact; its current
  *DOS packaging* is not yet competitive with Watcom's. Both
  statements are true and the table shows which is which — no
  single "390× smaller" headline.
