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
- Functions get a standard `push ebp / mov ebp, esp / sub esp, N / ... / mov esp, ebp / pop ebp / ret` frame.
- Falling off the end of any function leaves EAX = 0 (correct for `main` per C99; deterministic for others until full codegen lands).
- `int` locals supported: addressed as `[ebp - N]`, allocated in a single up-front pass over the function body. All slots are 4 bytes regardless of declared size — type-aware sizing comes when `short`/`char` codegen lands.
- Expressions: integer literals, identifier reads, unary `+ - ~ !`, binary `+ - * / % & | ^ << >> == != < > <= >=`, and assignment `=` (lvalue must be an identifier). No calls, casts, `&&`/`||`, or pointer/array ops yet.
- Stack-machine evaluation: left → EAX → push, right → EAX → ECX, pop EAX, op. Comparisons land via `cmp` + `setCC al` + `movzx eax, al`. Division/modulo via `cdq` + `idiv ecx`. Right shift is `sar` (signed); will branch to `shr` when type info reaches codegen.

## Session log

- **2026-04-25 — Phase 0**: Replaced codegen stub with NASM emitter for `int main` + integer-literal returns. Picked NASM as the assembler target.
- **2026-04-25 — Phase 4 slice 1**: `int` locals with integer-literal initializers and identifier reads in returns. Frame layout: locals stacked at `[ebp - 4]`, `[ebp - 8]`, etc. Single-pass collection in the prologue. 5 new tests; 13 total passing.
- **2026-04-25 — Phase 4 slice 2**: Arithmetic, bitwise, shift, comparison binary ops; unary `- + ~ !`; assignment to identifiers; ExpressionStmt. Stack-machine eval. 21 new tests; 34 total passing.
