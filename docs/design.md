# Goal and design

## Goal

Compile representative public-source DOS games **unmodified**:

- Descent (Parallax, 1995 — Watcom)
- Duke Nukem 3D / Build engine (3D Realms, 1996 — Watcom)
- Rise of the Triad (Apogee, 1994 — Watcom)
- Heretic / Hexen (Raven, 1994–95 — Watcom)

These all share one compiler (Watcom C/C++) and one memory model
(flat 32-bit under DOS/4GW). That's the target.

**Non-goals:** 16-bit real-mode with near/far/huge memory models
(Wolf3D-era code). uc386 will *parse* the 16-bit keywords so that
shared period headers don't choke, but won't honor their semantics —
all pointers are 32-bit flat.

## Design

The uc80/uc386 family shares a single C23 frontend
([uc_core](https://github.com/avwohl/uc_core), itself uplox-driven).
This project contributes only:

- `main.py` — driver (CLI, I/O, embedding, post-processing)
- `codegen.py` — x86-32 NASM code generator
- `lib/i386_dos_libc.asm` — the DOS libc, plus `lib/include/` headers
- `runtime.py` — placeholder for Python-side runtime bindings (the
  real libc is the `.asm` above)
- `dos_emu.py` — i386 emulator harness for testing flat-binary output
- `dos_emu_netsim.py` — simulated network for the INT 0x83 packet-driver shim
- `dosiz_run.py` — alternate harness dispatching to `../dosiz`
  (in-process dosbox-staging, full DPMI 0.9)
- `harness.py` — selects between the two via `UC386_HARNESS`
- `addons/harness/` — the `.asm` → `.obj` → MZ+LE `.exe` pipeline
  (source checkout only; not shipped on PyPI)

The NASM-text peephole optimizer, the assembly-level dead-code
eliminator, and the libc symbol splitter used to live here; they were
factored out into [upeep386](https://github.com/avwohl/upeep386) and
are now a dependency rather than part of this repo.

Every front-end improvement (new C23 feature, AST optimization, DOS-era
syntax tolerance) lands in uc_core and benefits both targets
automatically.
