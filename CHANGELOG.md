# Changelog

Notable changes to uc386. Releases before 0.2.5 are described on the
[GitHub releases page](https://github.com/avwohl/uc386/releases).

## 0.2.5 — 2026-08-20

No change to uc386's own source since 0.2.4. This release exists to raise the
dependency floors, so that a fresh `pip install uc386` resolves the sibling
versions that carry the fixes below rather than the older ones pip would
otherwise be free to choose.

### Changed

- `upeep386` floor raised to `>=0.2.1`. 0.2.1 fixes two peephole faults, both
  of which reached ordinary uc386 output and both of which were found by
  bisecting an addons harness whose failures disappeared under
  `--no-peephole`:

  The optimizer collapsed `fld tword [esi]` followed by `fmulp st1, st0`
  into `fmul tword [esi]`. `fld` has an m80fp form; no x87 arithmetic
  instruction does — FADD/FMUL/FSUB/FDIV take m32fp or m64fp only. The
  result was an unencodable instruction, and NASM rejected the whole file
  with "invalid operand sizes", so every build that pulled in the printf
  float path failed to assemble.

  Separately, a `call` was assumed to clobber eax/ecx/edx, which holds for
  cdecl but not for the bundled hand-written libc, whose hot helpers take
  arguments in registers. The argument setup was folded away as dead and the
  callee read whatever the register happened to hold, so printing a number
  faulted on a wild address. Deadness is now derived per callee from the asm
  being optimized instead of from a fixed allowlist.

- `uc_core` floor raised to `>=0.4.1` (the `<0.5` cap is unchanged). 0.4.1 is
  four preprocessor and frontend fixes: `__has_include` / `__has_include_next`
  now exist, stringification no longer drops backslashes, a function-like
  macro's opening `(` may sit on the next line, and the DOS qualifier eraser
  no longer eats a struct, union or enum tag that happens to be spelled
  `near`, `far` or `huge`.

- `uplox` floor raised to `>=3.3.0`, matching what `uc_core` 0.4.1 requires.

Nothing here changes uc386's own command line, output format, or generated
code beyond what those dependencies change.
