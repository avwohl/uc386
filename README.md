# uc386

C23 compiler targeting the Intel 386 (i386 / x86-32) processor and MS-DOS.

**Status: skeleton only.** The frontend (parsing, preprocessing,
AST-level optimization) is fully functional via
[uc_core](https://github.com/avwohl/uc_core). The x86 code generator
is a stub.

## Design

The uc80/uc386 family shares a single C23 frontend
([uc_core](https://github.com/avwohl/uc_core)). This project
contributes only:

- `main.py` — the driver (CLI, I/O, embedding, post-processing)
- `codegen.py` — x86-32 code generator (stub)
- `runtime.py` — MS-DOS runtime bindings (stub)

Every front-end improvement (new C23 feature, AST optimization, etc.)
lands in uc_core and benefits both targets automatically.

## Sister projects

- [uc_core](https://github.com/avwohl/uc_core) — shared frontend
- [uc80](https://github.com/avwohl/uc80) — Z80 / CP/M backend (reference target)
- [dosemu](https://github.com/avwohl/dosemu) — MS-DOS emulator (intended test host)

## Install

```
pip install -e .
```

## Roadmap

1. Hello world: emit enough assembly for `int main(){return 0;}` and DOS syscall exit (INT 21h, AH=4Ch).
2. Pick an assembler target: NASM vs MASM vs a hand-rolled um386 (paralleling um80).
3. Integer codegen: 32-bit int, 16-bit short, 32-bit pointer. Reuses `uc_core.ast_optimizer` once TypeConfig lands in uc_core.
4. Memory model: flat 32-bit (DPMI) vs real-mode tiny/small/large. Start with tiny (.COM-like) for simplicity.
5. libc subset: printf/putchar/puts via DOS INT 21h.
6. Testing via [dosemu](https://github.com/avwohl/dosemu).

## License

GPL-3.0-or-later.
