# uc386 — Claude operating notes

C23 compiler targeting i386 / MS-DOS (flat-32 Watcom-era). The frontend
lives in [uc_core](https://github.com/avwohl/uc_core) (sibling checkout
expected at `../uc_core`). This repo owns only the driver, the x86-32
codegen, and the DOS runtime bindings.

See `README.md` for the public roadmap (Phase 0–6).

## Layout

- `src/uc386/main.py` — driver: CLI, preprocess → lex → parse → optimize → codegen → write `.asm`
- `src/uc386/codegen.py` — x86-32 NASM emitter
- `src/uc386/runtime.py` — DOS/DPMI runtime bindings (stub)
- `tests/test_smoke.py` — end-to-end pipeline checks

## Toolchain

- Python 3.12 (system Python 3.9 is too old — uc_core uses `dataclass(kw_only=True)`).
- Working venv at `.venv/` with `uc_core` and `uc386` installed editable.
  - Create: `python3.12 -m venv .venv && .venv/bin/pip install pytest -e ../uc_core -e .`
- Run tests: `.venv/bin/pytest tests/`
- Run driver: `.venv/bin/python -m uc386.main examples/hello.c -o /tmp/hello.asm`
- Assembler target: NASM Intel syntax (`bits 32`, `section .text`).

## Codegen contract (current)

- Output is a single `.asm` text file in NASM syntax.
- Entry point `_start` calls `_main`, then exits via `INT 21h` AH=4Ch with AL = main's return.
- Functions get a standard `push ebp / mov ebp, esp / ... / mov esp, ebp / pop ebp / ret` frame.
- Falling off the end of any function leaves EAX = 0 (correct for `main` per C99; deterministic for others until full codegen lands).
- Anything beyond `return <int-literal>;` (or empty body / bare `return;`) raises `CodegenError`.

## Session log

- **2026-04-25 — Phase 0**: Replaced codegen stub with NASM emitter for `int main` + integer-literal returns. Picked NASM as the assembler target. 8 smoke tests cover structure, exit-code path, fall-through-zero, and rejection of non-literal returns / missing main.
