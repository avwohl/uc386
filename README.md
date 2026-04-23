# uc386

C23 compiler targeting the Intel 386 (i386 / x86-32) processor under a
DOS extender — specifically the **flat 32-bit Watcom / DOS/4GW-era** C
that early-to-mid-1990s PC games were written in.

**Status: skeleton only.** The frontend (parsing, preprocessing,
AST-level optimization) is fully functional via
[uc_core](https://github.com/avwohl/uc_core). The x86 code generator
is a stub.

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
([uc_core](https://github.com/avwohl/uc_core)). This project
contributes only:

- `main.py` — driver (CLI, I/O, embedding, post-processing)
- `codegen.py` — x86-32 code generator (stub)
- `runtime.py` — DOS/DPMI runtime bindings (stub)

Every front-end improvement (new C23 feature, AST optimization, DOS-era
syntax tolerance) lands in uc_core and benefits both targets
automatically.

## Install

```
pip install -e .
```

## Roadmap

### Phase 0 — hello world (current)
Emit enough assembly for `int main(){return 0;}` and a DOS INT 21h/4Ch
exit. Pick an assembler target (candidates: NASM, MASM, hand-rolled
`um386` paralleling `um80`).

### Phase 1 — syntactic tolerance for DOS-era cruft
Parse-and-ignore the non-standard keywords/pragmas that period headers
use. In flat-32 these are mostly no-ops — we just need the parser to
not choke on them. Lands in **uc_core** (shared with uc80). Includes:

- **Type qualifiers to ignore**: `near`, `far`, `huge`, `__near`,
  `__far`, `__huge`, `_cs`, `_ds`, `_es`, `_ss`, `_seg`, `__based(...)`
- **Calling-convention keywords**: `__cdecl`, `__pascal`, `__stdcall`,
  `__fastcall`, `__syscall`, `__watcall` (plus bare and `_`-prefixed
  variants). Accepted; all compile to the same ABI in Phase 1.
- **Function attributes**: `__interrupt`, `interrupt`, `__loadds`,
  `__saveregs`, `__export`
- **Pragmas to drop**: `hdrstop`, `hdrfile`, `warn`, `warning`,
  `intrinsic`, `function`, `check_stack`, `code_seg`, `data_seg`,
  `alloc_text`, `disable_message`, `argsused`, `inline`, `library`,
  `startup`, `exit`. (`pack` stays honored.)

### Phase 2 — Watcom real (the big one)
The survey says `#pragma aux` is the single feature that unlocks
Descent, Duke3D, ROTT, and Heretic/Hexen. It has two forms:

1. Describe calling convention for a named function:
   `#pragma aux f parm [eax] [edx] value [eax] modify [ecx];`
2. Define an inline-asm function body:
   `#pragma aux f = "add eax, edx" parm [eax] [edx] value [eax];`

Also in this phase: `__watcall` as a real ABI (first 4 args in
`EAX/EDX/EBX/ECX`), `_asm { }` Intel-syntax inline blocks.

### Phase 3 — optional gcc-compat
If we want Doom's public source (Linux port, DJGPP-style) or Quake
(also DJGPP): GCC-style `asm(...)`, `__attribute__((...))`, GAS `.S`
input.

### Phase 4 — integer codegen
32-bit int, 16-bit short, 32-bit pointer. Reuses
`uc_core.ast_optimizer` once TypeConfig lands in uc_core.

### Phase 5 — libc subset
printf / putchar / puts / file I/O via DOS INT 21h.

### Phase 6 — testing
Via [dosemu](https://github.com/avwohl/dosemu) or similar.

## Related Projects

- [cpmdroid](https://github.com/avwohl/cpmdroid) - Z80/CP/M emulator for Android with RomWBW HBIOS compatibility and VT100 terminal
- [cpmemu](https://github.com/avwohl/cpmemu) - CP/M 2.2 emulator with Z80/8080 CPU emulation and BDOS/BIOS translation to Unix filesystem
- [dosemu](https://github.com/avwohl/dosemu) - MS-DOS emulator for Linux: dosbox-staging CPU + cpmemu-style syscall translation (intended test host for uc386)
- [qxDOS](https://github.com/avwohl/qxDOS) - DOS emulator for iPad and Mac — DOSBox-based with SwiftUI interface
- [uc80](https://github.com/avwohl/uc80) - C23 compiler targeting Z80 processor and CP/M; sibling backend sharing the uc_core frontend
- [uc_core](https://github.com/avwohl/uc_core) - Shared C23 frontend and AST optimizer used by uc80 and uc386
- [um80_and_friends](https://github.com/avwohl/um80_and_friends) - Microsoft MACRO-80 compatible toolchain for Linux: assembler, linker, librarian, disassembler (the Z80 analogue of what uc386 needs for i386)
- [upeepz80](https://github.com/avwohl/upeepz80) - Z80 peephole optimizer (template for an eventual upeep386)

## License

GPL-3.0-or-later.
