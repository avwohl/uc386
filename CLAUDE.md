# uc386 — Claude operating notes

C23 compiler targeting i386 / MS-DOS (flat-32 Watcom-era). The frontend
lives in [uc_core](https://github.com/avwohl/uc_core) (sibling checkout
expected at `../uc_core`). This repo owns the driver, the x86-32 codegen,
and the DOS runtime bindings; several language-neutral pieces have been
split out into sibling packages (see **Sibling packages** below).

See `README.md` for the public roadmap (Phase 0–6).

## Layout

- `src/uc386/main.py` — driver: CLI, preprocess → lex → parse → optimize → codegen → write `.asm`
- `src/uc386/codegen.py` — x86-32 NASM emitter
- `src/uc386/runtime.py` — DOS/DPMI runtime bindings (stub)
- `tests/test_smoke.py` — end-to-end pipeline checks

## Sibling packages (REQUIRED — bootstrap reads this)

uc386 imports these at runtime; **all must be cloned as siblings and
installed editable** or the driver/tests fail (`ModuleNotFoundError`,
or a collection error in `tests/test_libc_split_integration.py`):

| Sibling | Imported by | Clone from |
|---|---|---|
| `../uc_core` | frontend (lex/parse/AST/const-fold) | `git@github.com:avwohl/uc_core.git` |
| `../uplox` | uc_core dep | `git@github.com:avwohl/uplox.git` |
| `../upeep386` | `codegen.py`, `main.py`, `dos_emu.py` (`PeepholeOptimizer`, `dce`, `optimize`, `parse_libc`) | `git@github.com:avwohl/upeep386.git` |
| `../pyle` | `addons/harness/exe.py` — imported as **`upyle`** (`parse_omf`, `link`, `write_le`, `bind_dos32a_stub` — the OMF→MZ+LE linker for the `.exe` pipeline) | `git@github.com:avwohl/pyle.git` |

⚠️ **The repo is `pyle`; the package is `upyle`.** Directory
`../pyle`, but `pip install upyle` and `import upyle`. The bare name
`pyle` on PyPI belongs to an unrelated project (aljungberg/pyle, a
shell one-liner tool) that installs a top-level `pyle.py`, so sharing
the name would mean the two silently shadow each other on `sys.path`.
Never add a bare `pyle` dependency anywhere, and never `pip install
pyle` — you get the wrong package and an AttributeError on
`parse_omf`.

⚠️ **pyle keeps getting "fixed" the wrong way.** It IS on GitHub at
`github.com/avwohl/pyle` — **clone it.** Do NOT re-recover it from
uc386 git history (`02f6daf^`); that produces a rootless local-only
repo with no remote that has to be redone on every machine. If you
find a `../pyle` with no `origin` remote, replace it with the clone.
(Same applies to upeep386: it was split out of uc386 in `a7e0a18` /
`27b1348` and lives at `github.com/avwohl/upeep386` — clone, don't
copy `peephole.py`/`asm_dce.py`/`libc_split.py` back in.)

## Toolchain

- Python ≥ 3.10 (uc_core uses `dataclass(kw_only=True)`, added in 3.10).
  Linux ships 3.10+ in current LTSes; on macOS install via
  `brew install python@3.12` (Apple's system 3.9 is too old).
- Working venv at `.venv/` with all four siblings + `uc386` + `unicorn`
  installed editable.
  - Clone the siblings first (see table above), then create the venv:
    ```
    python3 -m venv .venv
    .venv/bin/pip install pytest unicorn \
        -e ../uc_core -e ../uplox -e ../upeep386 -e ../pyle -e .
    ```
    (The old `-e ../uc_core -e .`-only command is what left every fresh
    checkout broken — it never installed uplox/upeep386/pyle.)
- Run tests: `.venv/bin/pytest tests/` (expect 460 passed, 1 skipped).
  The peephole/dce/libc_split tests now live in upeep386 — run those
  with `.venv/bin/pytest ../upeep386/tests` (expect 897 passed).
- Run driver: `.venv/bin/python -m uc386.main examples/hello.c -o /tmp/hello.asm`
- Assembler target: NASM Intel syntax (`bits 32`, `section .text`).
- Full per-platform install (brew / apt / dnf), incl. optional bison +
  DJGPP + OpenWatcom for addons and size comparison: see `docs/INSTALL.md`.

## Codegen contract (current)

- Output is a single `.asm` text file in NASM syntax.
- Entry point `_start` calls `_main`, then exits via `INT 21h` AH=4Ch with AL = main's return.
- Functions get a standard `push ebp / mov ebp, esp / sub esp, N / ... / mov esp, ebp / pop ebp / ret` frame.
- Falling off the end of any function leaves EAX = 0 (correct for `main` per C99; deterministic for others until full codegen lands).
- Scalar locals (`int`, `short`, `char` — signed and unsigned) are addressed as `[ebp - N]`, allocated in a single up-front pass. Each slot is rounded up to a 4-byte boundary (`(size + 3) & ~3`) so adjacent ints stay aligned and `char arr[5]` consumes 8 bytes of frame. The byte-payload width is preserved at the access level via `_load_to_eax` / `_store_from_eax`.
- Expressions: integer literals, character literals (`'A'`), identifier reads (with array and function decay), unary `+ - ~ ! ++ -- & *`, binary `+ - * / % & | ^ << >> == != < > <= >= && ||`, assignment `=` to an identifier / `*p` / `arr[i]` / `s.m` / `p->m`, compound assignment (`+= -= *= /= %= &= |= ^= <<= >>=`) to any of those lvalues, ternary `?:`, array indexing `arr[i]` (read or write), struct member access `s.m` and `p->m`, `sizeof` (both `sizeof(type)` and `sizeof(expr)`, evaluated at compile time), and explicit casts `(T)expr`. Struct-to-struct assignment, struct-by-value params, struct return-by-value (caller-provided buffer or per-call temp), unions, bitfields, and designated initializers all land cleanly. Phase 4–5 features that aren't yet supported live in `docs/changes.md`'s slice-by-slice notes.
- Pointer arithmetic obeys C scaling rules. `_FuncCtx` carries a parallel `types` map; `_type_of(expr, ctx)` does best-effort static type inference (Identifier → declared type, `&x` → pointer-to, `*p` → pointee, `+`/`-` → propagate pointer-ness, Index → element type, others → int). `_size_of` knows the i386 sizes for `char`/`short`/`int`/`long`/`long long`/`void`/pointer/array. `_is_pointer_like` collapses `PointerType` and `ArrayType` into a single "pointer-like" predicate so array names participate in the same arithmetic and dereference paths as real pointers. `+` and `-` route through `_add_sub`, which handles ptr±int (scale the int), int+ptr (symmetric), and ptr-ptr (subtract then unscale). `++`/`--` on a pointer slot emit `add/sub dword [...], sizeof(*ptr)` instead of `inc/dec`. `+` of two pointers and `int - ptr` are rejected. Scaling uses `shl`/`sar` for power-of-two sizes, `imul`/`idiv` otherwise.
- Arrays: `int arr[N]` allocates `N * sizeof(elem)` on the frame (rounded up to a 4-byte boundary); the slot's lowest byte is `arr[0]`. `_collect_locals` calls `_resolved_var_type(decl)` first — that fills in inferred sizes for `int arr[] = {...}` and `char s[] = "..."` from the initializer. Array names decay to addresses in expression context (`Identifier` of `ArrayType` lowers to `lea`, not `mov`). `arr[i]` lowers via `_index_address` (eval array → push, eval index → scale by element size → pop+add) followed by a width-correct load via `_load_to_eax`; stores use `_store_from_eax`. `&arr[i]` reuses `_index_address` without the deref. Assignment to an array name and `++`/`--` on an array name still raise. `int arr[N] = {a, b, c}` and `char s[] = "..."` are handled by `_array_init`: per-element stores via `_store_from_eax` (so `char arr[3] = {65, 66, 67}` writes bytes), then `mov <width> [...], 0` zero-fills any unfilled trailing elements. Designated initializers and nested `{}` for multidim arrays still raise.
- Sub-word codegen: `_load_to_eax(addr, ty)` and `_store_from_eax(addr, ty)` are the single chokepoints for slot/element access. Loads use `mov eax, [...]` for 4-byte values, `movsx eax, word [...]` / `movzx eax, word [...]` for shorts, and `movsx eax, byte [...]` / `movzx eax, byte [...]` for chars (zero-extension when `is_signed=False`, sign-extension otherwise). Stores narrow via `mov word [...], ax` or `mov byte [...], al`. The helpers are wired into Identifier read, `_var_init`, `_assign` (Identifier / `*ptr` / `arr[i]`), `_unary *`, `_index_load`, and `_inc_dec` (with `inc/dec byte` or `inc/dec word` for sub-word slots). Integer promotion happens implicitly because every load returns a 32-bit EAX value.
- Control flow: `if`/`else`, `while`, `do`/`while`, `for`, `switch`/`case`/`default`, `break`, `continue`, `goto`/labels. Labels are function-local (NASM `.LN_*`), generated via a per-function counter. `_FuncCtx` carries two parallel stacks — `break_targets` (pushed by both loops and switches) and `continue_targets` (pushed only by loops) — so `continue` inside a switch correctly escapes to the enclosing loop. User-declared labels are pre-walked into a name → NASM-label map so forward `goto`s resolve.
- Stack-machine evaluation: left → EAX → push, right → EAX → ECX, pop EAX, op. Comparisons land via `cmp` + `setCC al` + `movzx eax, al`. Division/modulo via `cdq` + `idiv ecx`. Right shift is `sar` (signed); will branch to `shr` when type info reaches codegen.
- Locals are allocated in a single recursive pre-pass over the whole function (including nested blocks, if-branches, loop bodies, for-init). Flat scope — redeclaring a name in a nested block raises.
- ABI (current): cdecl. Caller pushes args right-to-left, callee accesses params via [ebp + 8 + accumulated-size]; scalars take 4 bytes, struct-by-value takes `sizeof(struct)` rounded up to 4. Caller cleans the stack. Return value in EAX. Struct-returning functions take a hidden first param (`__retptr__` at [ebp+8]) — caller pushes the destination address last so it lands at the leftmost arg slot, callee copies the value into `*__retptr__` and returns the same pointer in EAX so chained struct returns work without temps. Watcom register call (`__watcall`) is Phase 2 work in uc_core; once it lands we'll switch the default but keep cdecl reachable.
- Calls: `_call` dispatches direct vs indirect via `_emit_call`. A direct call is `call _name` when the callee is an Identifier whose name is in `_func_return_types`; otherwise the callee expression evaluates to EAX after args are pushed and we emit `call eax`. Leading `*`s on the callee (`(*fp)()`) are stripped — function-typed values are idempotent under `*` in C. A function name in value position (e.g. `int (*fp)() = helper;` or passing `helper` as an arg) decays to its label as an immediate via `_identifier_load`'s function-name fallback.
- Address rendering: `_ebp_addr(disp)` produces `[ebp - N]` for negative displacements (locals) and `[ebp + N]` for positive (params). Slot displacements live on `_FuncCtx.slots`.
- Globals: top-level VarDecls are registered in `CodeGenerator._globals` (resolved type) and `_global_inits` (init expr if present). Identifier access dispatches via the `_identifier_*` helpers — locals first (`ctx.slots`), globals second — so a function-scope local cleanly shadows a same-named global. Access lowers to `[_name]` (memory) or `_name` (immediate, for `&` and array decay). Initialized globals land in `.data` with `db`/`dw`/`dd` (or `dd v1, v2, ...` for arrays); uninitialized in `.bss` with `resb N`. Init expressions must reduce to compile-time constants — `_const_eval` handles literals, unary `-/+/~/!`, and the standard binary integer ops; references to other identifiers raise.


## Session log

Slice-by-slice notes have moved to `docs/changes.md`.
