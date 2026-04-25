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
- Expressions: integer literals, identifier reads (with array decay), unary `+ - ~ ! ++ -- & *`, binary `+ - * / % & | ^ << >> == != < > <= >= && ||`, assignment `=` to an identifier / `*p` / `arr[i]`, compound assignment (Identifier lvalue only), ternary `?:`, and array indexing `arr[i]` (read or write). No casts, no struct/union access yet.
- Pointer arithmetic obeys C scaling rules. `_FuncCtx` carries a parallel `types` map; `_type_of(expr, ctx)` does best-effort static type inference (Identifier → declared type, `&x` → pointer-to, `*p` → pointee, `+`/`-` → propagate pointer-ness, Index → element type, others → int). `_size_of` knows the i386 sizes for `char`/`short`/`int`/`long`/`long long`/`void`/pointer/array. `_is_pointer_like` collapses `PointerType` and `ArrayType` into a single "pointer-like" predicate so array names participate in the same arithmetic and dereference paths as real pointers. `+` and `-` route through `_add_sub`, which handles ptr±int (scale the int), int+ptr (symmetric), and ptr-ptr (subtract then unscale). `++`/`--` on a pointer slot emit `add/sub dword [...], sizeof(*ptr)` instead of `inc/dec`. `+` of two pointers and `int - ptr` are rejected. Scaling uses `shl`/`sar` for power-of-two sizes, `imul`/`idiv` otherwise.
- Arrays: `int arr[N]` allocates `N * sizeof(elem)` on the frame; the slot's lowest byte is `arr[0]`. `_collect_locals` passes the actual `_size_of(var_type)` to `alloc_local` so non-int slots get full-width frames. Array names decay to addresses in expression context (`Identifier` of `ArrayType` lowers to `lea`, not `mov`). `arr[i]` lowers via `_index_address` (eval array → push, eval index → scale by element size → pop+add) followed by `mov eax, [eax]` for loads or `pop ecx; mov [ecx], eax` for stores. `&arr[i]` reuses `_index_address` without the deref. Assignment to an array name, `++`/`--` on an array name, and `int arr[N] = {...}` initializers all raise — array-init waits on InitializerList lowering. Sub-word loads/stores still come with `char`/`short` codegen, so element types are int-sized today.
- Control flow: `if`/`else`, `while`, `do`/`while`, `for`, `break`, `continue`. Labels are function-local (NASM `.LN_*`), generated via a per-function counter. Loop targets live on a stack on the function context so `break`/`continue` resolve to the innermost loop.
- Stack-machine evaluation: left → EAX → push, right → EAX → ECX, pop EAX, op. Comparisons land via `cmp` + `setCC al` + `movzx eax, al`. Division/modulo via `cdq` + `idiv ecx`. Right shift is `sar` (signed); will branch to `shr` when type info reaches codegen.
- Locals are allocated in a single recursive pre-pass over the whole function (including nested blocks, if-branches, loop bodies, for-init). Flat scope — redeclaring a name in a nested block raises.
- ABI (current): cdecl. Caller pushes args right-to-left, callee accesses params via `[ebp + 8 + 4*i]`, caller cleans the stack. Return value in EAX. Watcom register call (`__watcall`) is Phase 2 work in uc_core; once it lands we'll switch the default but keep cdecl reachable.
- Address rendering: `_ebp_addr(disp)` produces `[ebp - N]` for negative displacements (locals) and `[ebp + N]` for positive (params). Slot displacements live on `_FuncCtx.slots`.

## Session log

- **2026-04-25 — Phase 0**: Replaced codegen stub with NASM emitter for `int main` + integer-literal returns. Picked NASM as the assembler target.
- **2026-04-25 — Phase 4 slice 1**: `int` locals with integer-literal initializers and identifier reads in returns. Frame layout: locals stacked at `[ebp - 4]`, `[ebp - 8]`, etc. Single-pass collection in the prologue. 5 new tests; 13 total passing.
- **2026-04-25 — Phase 4 slice 2**: Arithmetic, bitwise, shift, comparison binary ops; unary `- + ~ !`; assignment to identifiers; ExpressionStmt. Stack-machine eval. 21 new tests; 34 total passing.
- **2026-04-25 — Phase 4 slice 3**: Control flow — `if`/`else`, `while`, `do`/`while`, `for`, `break`, `continue` — plus `&&` and `||` short-circuit. Per-function label counter; loop stack for break/continue targets. `_collect_locals` made recursive so for-init declarations get slots. 11 new tests; 45 total passing.
- **2026-04-25 — Phase 4 slice 4**: Function parameters + direct calls under cdecl. Refactored slot storage to signed displacements with `_ebp_addr` renderer (negative = local, positive = param). Renamed `alloc` → `alloc_local`, added `alloc_param`. 7 new tests; 52 total passing.
- **2026-04-25 — Phase 4 slice 5**: Compound assignment, prefix/postfix `++`/`--`, ternary `?:`. Compound ops desugar to `lvalue = lvalue OP rvalue` via a synthesized BinaryOp (safe for Identifier lvalues; pointer/array lvalues will need a different lowering). `++`/`--` use `inc dword [ebp - N]` / `dec dword [ebp - N]` with order chosen by `is_prefix`. 12 new tests; 64 total passing.
- **2026-04-25 — Phase 4 slice 6**: Pointers — `&x` (Identifier only) emits `lea`, `*expr` loads through EAX, `*p = rhs` stores through pointer (eval pointer → push, eval rhs → pop ecx, `mov [ecx], eax`). Pointer-typed locals/params share a 4-byte slot. Pointer arithmetic with size scaling deferred until type info is plumbed. Renamed `_check_int_type` → `_check_supported_type`. 5 new tests; 69 total passing.
- **2026-04-25 — Phase 4 slice 7**: String literals + extern declarations. Strings interned per translation unit and emitted as null-terminated `db` lines in `.data` (control bytes split out as numeric segments so NASM stays happy). Bodyless function declarations — produced by the parser as either FunctionDecl(body=None) or VarDecl with a FunctionType — emit `extern _name`. 5 new tests; 74 total passing.
- **2026-04-25 — Phase 4 slice 8**: Pointer arithmetic with size scaling. `_FuncCtx` now carries a parallel `types` map (populated from VarDecl/ParamDecl) and a module-level `_func_return_types` table is built in `generate()`. New `_type_of` does best-effort type inference and `_size_of` resolves i386 pointee sizes. `+`/`-` are routed through `_add_sub` which scales the int side via `shl`/`imul` for ptr+int and unscales the byte difference via `sar`/`idiv` for ptr-ptr; `++`/`--` on a pointer slot emit `add/sub dword [...], sizeof(*ptr)`. `+` of two pointers and `int - ptr` raise. 13 new tests; 87 total passing.
- **2026-04-25 — Phase 4 slice 9**: Arrays — uninitialized locals, indexing, decay. `_check_supported_type` and `_size_of` recurse through `ArrayType`; `_collect_locals` now passes `_size_of(var_type)` to `alloc_local` so a 12-byte `int arr[3]` slot is reserved end-to-end. Identifier eval emits `lea` (not `mov`) when the slot's declared type is an array. New `_index_address` and `_index_load` lower `arr[i]` as base-plus-scaled-index; `_assign` and `_address_of` both gain Index branches so `arr[i] = rhs` and `&arr[i]` work. `_is_pointer_like` collapses `PointerType` / `ArrayType` for arithmetic and dereference paths, so `arr + 2`, `*arr`, `arr[i]`, and `int *p = arr;` all share scaling. Assigning to / incrementing an array, initializing one with `{...}`, and unsized `int a[];` locals raise. 12 new tests; 99 total passing.
