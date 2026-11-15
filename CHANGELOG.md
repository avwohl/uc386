# Changelog

Notable changes to uc386. Releases before 0.2.5 are described on the
[GitHub releases page](https://github.com/avwohl/uc386/releases).

## Unreleased

### Changed

- `pyproject.toml` no longer has the `<0.5` upper limit on `uc_core`. It has
  floors only. The `pypi-bounds` workflow resolves from PyPI each week, and
  finds a `uc_core` release that breaks uc386.
- CI uses Python `3.x` (the newest release) in place of 3.12 and 3.13, and
  the latest DJGPP release in place of v3.4.

## 0.2.6 — 2026-11-05

No change to uc386's own source since 0.2.5 (two README edits only). This
release raises the `uc_core` floor, so that a fresh `pip install uc386`
resolves uc_core 0.4.2 and its fixes rather than 0.4.1.

### Changed

- `uc_core` floor raised to `>=0.4.2` (the `<0.5` cap is unchanged). 0.4.2 is
  the 38 commits since 0.4.1, mostly preprocessor and parser fixes:
  `__VA_OPT__` and `#__VA_ARGS__`, a self-named macro and an alias call
  through an argument, the GNU comma paste, `__COUNTER__`, `_Pragma` and
  pragma lines in `-E` text, a macro parameter replaced only in identifier
  tokens; a label before `asm(...)`, a case or default label before a
  declaration or a statement word; a block use of a tag that declares the
  tag, a parameter that hides a file-scope enumerator; `volatile` kept for
  typedef, struct, union and enum types. The uc_core CHANGELOG has each one.

Measured on 2026-11-05: in a fresh Python 3.12 venv, `pip install .` took
uc_core 0.4.2, upeep386 0.2.1 and uplox 3.3.1 from PyPI, and the suite with
`unicorn` installed gave 498 passed, 1 skipped (the MicroPython test, which
needs a built `micropython.bin`) - the same as against uc_core 0.4.1.

Nothing here changes uc386's own command line, output format, or generated
code beyond what uc_core changes.

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
