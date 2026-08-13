# uc386 — Claude operating notes

C23 compiler targeting i386 / MS-DOS (flat-32 Watcom-era). The frontend
lives in [uc_core](https://github.com/avwohl/uc_core) (sibling checkout
expected at `../uc_core`). This repo owns the driver, the x86-32 codegen,
and the DOS runtime bindings; several language-neutral pieces have been
split out into sibling packages (see **Sibling packages** below).

See `README.md` for the public status, the measured size table, and
the install paths. Released on PyPI — `pip install uc386` (0.2.0).

## Layout

- `src/uc386/main.py` — driver: CLI, preprocess → lex → parse → optimize → codegen → write `.asm`
- `src/uc386/codegen.py` — x86-32 NASM emitter
- `src/uc386/lib/i386_dos_libc.asm` — the DOS libc; `lib/include/` the headers
- `src/uc386/runtime.py` — placeholder for Python-side bindings (still a stub;
  the real libc is the `.asm` above)
- `src/uc386/dos_emu.py` — unicorn-based i386 emulator for flat `.bin` output
- `src/uc386/dos_emu_netsim.py` — simulated network for the INT 0x83 shim
- `src/uc386/dosiz_run.py` — alternate harness dispatching to `../dosiz`
- `src/uc386/harness.py` — picks between the two via `UC386_HARNESS`
- `addons/harness/` — `.asm` → `nasm -f obj` → upyle → MZ+LE `.exe` pipeline
  (`exe.py`), the size comparison (`compare.py`), addon builds (`build.py`),
  release packaging (`package.py`)
- `tests/` — `test_smoke.py` (end-to-end), plus codegen/peephole integration,
  libc split, env block, VFS seed, and MicroPython integration checks

## Sibling packages (REQUIRED — bootstrap reads this)

uc386 imports these at runtime. **All four are published on PyPI**, and
`uc_core` / `uplox` / `upeep386` are declared dependencies, so a plain
`pip install uc386` (or `pip install -e .`) resolves them without any
clone. `upyle` is deliberately *not* declared — the `.exe` pipeline
lives in `addons/`, which isn't packaged — so add it explicitly:
`pip install upyle`.

Clone them as siblings and install editable only when **co-developing**
the frontend or the optimizer alongside uc386. Get that wrong and the
driver/tests fail (`ModuleNotFoundError`, or a collection error in
`tests/test_libc_split_integration.py`):

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

- Python ≥ 3.11, and the floor comes from a **transitive dep**, not our
  own syntax: `src/uc386`, `tests/`, uc_core and upeep386 are all
  3.10-clean, but uc_core → uplox, and uplox declares `>=3.11`. On 3.10
  `pip install uc386` fails at resolution with "no matching
  distributions ... uplox", so 3.10 was never installable regardless of
  whether the code would have run. On macOS install via
  `brew install python@3.12` (Apple's system 3.9 is too old).
  Note `addons/harness/` additionally needs 3.11 outright — it imports
  `tomllib`, which is stdlib only from 3.11.
- Working venv at `.venv/`.
  - **Just working on uc386?** No clones needed — pip resolves the
    frontend and optimizer from PyPI. `upyle` is the one extra:
    ```
    python3 -m venv .venv
    .venv/bin/pip install pytest unicorn upyle -e .
    ```
  - **Co-developing a sibling?** Clone it (see table above) and install
    that one editable on top; the rest can stay on PyPI:
    ```
    .venv/bin/pip install pytest unicorn \
        -e ../uc_core -e ../uplox -e ../upeep386 -e ../pyle -e .
    ```
    (The old `-e ../uc_core -e .`-only command is what left every fresh
    checkout broken — it never installed uplox/upeep386/pyle.)
  - ⚠️ A venv missing `upyle` still passes `pytest tests/` — nothing in
    `tests/` touches the `.exe` path. It fails only when you run
    `addons.harness.exe`. Check with
    `.venv/bin/python -c "import upyle"` before trusting an `.exe` build.
- Run tests: `.venv/bin/pytest tests/` (expect 498 passed, 1 skipped).
  The peephole/dce/libc_split tests now live in upeep386 — run those
  with `.venv/bin/pytest ../upeep386/tests` (expect 897 passed).
- Run driver: `.venv/bin/python -m uc386.main examples/hello.c -o /tmp/hello.asm`
- Build a DOS `.exe`:
  `.venv/bin/python -m addons.harness.exe addons/gnu/true/main.c -o /tmp/true.exe`
  Defaults to the **DOS/32A** extender (since `58c1f79`); `--extender=pmodew`
  halves the size but cannot do disk I/O on real DOS. Needs `nasm` + `upyle`;
  Open Watcom is optional and only required for `causeway`/`dos4g`.
- Assembler target: NASM Intel syntax (`bits 32`, `section .text`).
- Full per-platform install (brew / apt / dnf), incl. optional bison +
  DJGPP + OpenWatcom for addons and size comparison: see `docs/INSTALL.md`.

## Conformance suites

The README's headline numbers (215/220 c-testsuite, 1397/1514
gcc-c-torture) come from these runners. They are **not** part of
`pytest tests/` and need upstream checkouts under `../external/`:

```
git clone https://github.com/c-testsuite/c-testsuite.git ../external/c-testsuite
git clone https://github.com/llvm/llvm-test-suite.git    ../external/llvm-test-suite

.venv/bin/python run_ctests.py --full            # 215/220
.venv/bin/python run_gcc_torture.py --full --kr  # 1397/1514
```

- `--full` = compile + NASM-assemble + run under `dos_emu` + diff
  stdout. Without it both default to `--compile-only`, which is a
  much weaker signal — don't quote a compile-only number as a pass
  rate.
- `--kr` is required for the torture corpus (pre-ANSI + GNU-heavy);
  `run_ctests.py` defaults it on, `run_gcc_torture.py` does not.
- Single test: `run_gcc_torture.py --full --kr -v <name>`.
- `run_fujitsu.py` (needs `../external/compiler-test-suite`) works
  but its results are reported in no doc. `run_sdcc.py` is a
  skeleton that prints a "not ported" message and exits — it is not
  a runner yet.

## Releasing

Two workflows, triggered in sequence by one tag push:

1. Bump `version` in `pyproject.toml`, commit, then push a `v*` tag.
2. `.github/workflows/release.yml` fires on the tag: installs nasm +
   bison + the siblings, runs `pytest tests/`, builds the FOSS and
   games tarballs via `addons.harness.package`, regenerates
   `addons/results.md`, and attaches everything to a GitHub release.
3. `.github/workflows/publish.yml` pushes the wheel + sdist to PyPI via
   trusted publishing (no API token in secrets). ⚠️ **It will not fire
   on its own.** Its `release: published` trigger is suppressed because
   release.yml creates the release with the default `GITHUB_TOKEN`, and
   GitHub does not let workflow-created events trigger further
   workflows. Releases through v0.1.5 only published because a human
   created them by hand; v0.2.0 was the first automated one and it
   stalled. So after the release appears, run:

   ```
   gh workflow run publish.yml --ref v0.2.0
   ```

   (or toggle the release draft off/on under a real account, which does
   emit the event). Verify with
   `curl -s https://pypi.org/simple/uc386/ | grep 0.2.0` — the JSON API
   is CDN-cached and lags. The durable fix is to have release.yml
   create the release with a PAT instead of `GITHUB_TOKEN`.

Keep the `uc_core` upper bound in `pyproject.toml` in step with the
API actually used — see the comment there. CI installs the siblings
from `@main` rather than PyPI, so a bad bound passes CI and only
breaks for `pip install uc386` users.

The `pytest.yml` matrix (3.11–3.13) matches `requires-python` now.
It previously claimed `>=3.10` while testing only 3.11+; the claim was
the wrong half — 3.10 is uninstallable because of uplox's own floor
(see **Toolchain** above), so the floor moved up rather than the matrix
moving down. Keep the two in step if uplox's floor ever changes.

## Codegen owns the exit-time stdio flush

Console output is line-buffered in the libc, but **return-from-main
never passes through a libc function** — `_start_stub` calls `_main`
and falls straight into INT 21h/4Ch. So codegen emits the flush:
`_start_stub` drops a `_FLUSH_HOOK` sentinel, and `generate()` replaces
it once every call site has been lowered (the stub is built *before*
the bodies, so the decision can't be made in place).

It is emitted only when `_uses_buffered_stdio()` is true — i.e. the
program actually calls a buffered-output symbol. An unconditional call
would link the buffering code into every binary, and `true.bin` is 18
bytes precisely because nothing unused is linked in. The flush is
wrapped in `push eax` / `pop eax` so it can't clobber main's exit code.

If you add a libc function that writes to the console, add it to
`_BUFFERED_STDIO_SYMS` or programs calling only that function will lose
their last partial line.

## Trap: libc helpers must take arguments on the stack

`main.py` re-runs the peephole over the **combined** user + libc asm
(the `optimize` call after bundling), and the peephole deletes a
`mov eax, [ebp+8]` that sits immediately before a `call`:

```
_caller:                          _caller:
        mov  eax, [ebp + 8]  =>           call __helper
        call __helper                     leave
```

That is sound for cdecl — EAX is caller-saved and its value is dead
once the callee's return value lands there — but it silently miscompiles
any hand-written libc helper that expects an argument **in a register**.
Nothing in the libc relies on that today, and new helpers must not start:
pass arguments on the stack (`__stdio_flag_ptr` is the worked example).
The peephole rewriting the caller's `add esp, 4` into `pop ecx` is fine
and expected; it does not touch the pushed argument itself.

## Codegen contract (current)

**`--int` / `--long` / `--long-long` / `--ptr` only accept the flat-32
values.** These flags feed the *frontend* — they reach `ASTOptimizer`,
which const-folds `sizeof` — while codegen sizes types from its own
`CodeGenerator._BASIC_SIZES` and never receives them. A non-default
width therefore used to produce a compiler whose `sizeof` contradicted
its own storage layout (`--long 64` → `sizeof(long)==8` with `long`
locals 4 bytes apart), silently overrunning any `sizeof`-driven
`memcpy`/stride.

`main.py` now compares the requested `TypeConfig` against
`_BASIC_SIZES` and exits 1 with a diagnostic rather than miscompiling.
Making a width genuinely work is the larger job: the 64-bit paths key
off the type *name* (`_is_long_long`, codegen.py) rather than the size,
so `long` at 8 bytes would get 8-byte storage and 32-bit arithmetic.
Size-key those predicates first; the guard lifts automatically once
`_BASIC_SIZES` agrees.

- Output is a single `.asm` text file in NASM syntax.
- Entry point `_start` calls `_main`, then exits via `INT 21h` AH=4Ch with AL = main's return.
- Functions get a standard `push ebp / mov ebp, esp / sub esp, N / ... / mov esp, ebp / pop ebp / ret` frame.
- Falling off the end of any function leaves EAX = 0 (correct for `main` per C99; deterministic for others until full codegen lands).
- Scalar locals (`int`, `short`, `char` — signed and unsigned) are addressed as `[ebp - N]`, allocated in a single up-front pass. Each slot is rounded up to a 4-byte boundary (`(size + 3) & ~3`) so adjacent ints stay aligned and `char arr[5]` consumes 8 bytes of frame. The byte-payload width is preserved at the access level via `_load_to_eax` / `_store_from_eax`.
- Expressions: integer literals, character literals (`'A'`), identifier reads (with array and function decay), unary `+ - ~ ! ++ -- & *`, binary `+ - * / % & | ^ << >> == != < > <= >= && ||`, assignment `=` to an identifier / `*p` / `arr[i]` / `s.m` / `p->m`, compound assignment (`+= -= *= /= %= &= |= ^= <<= >>=`) to any of those lvalues, ternary `?:`, array indexing `arr[i]` (read or write), struct member access `s.m` and `p->m`, `sizeof` (both `sizeof(type)` and `sizeof(expr)`, evaluated at compile time), and explicit casts `(T)expr`. Struct-to-struct assignment, struct-by-value params, struct return-by-value (caller-provided buffer or per-call temp), unions, bitfields, and designated initializers all land cleanly. What genuinely remains unimplemented is tracked in `STANDARD_C_BACKLOG.md` — **not** in `docs/changes.md`, which is a historical log closed on 2026-05-04 and lists limitations that have since been fixed.
- Pointer arithmetic obeys C scaling rules. `_FuncCtx` carries a parallel `types` map; `_type_of(expr, ctx)` does best-effort static type inference (Identifier → declared type, `&x` → pointer-to, `*p` → pointee, `+`/`-` → propagate pointer-ness, Index → element type, others → int). `_size_of` knows the i386 sizes for `char`/`short`/`int`/`long`/`long long`/`void`/pointer/array. `_is_pointer_like` collapses `PointerType` and `ArrayType` into a single "pointer-like" predicate so array names participate in the same arithmetic and dereference paths as real pointers. `+` and `-` route through `_add_sub`, which handles ptr±int (scale the int), int+ptr (symmetric), and ptr-ptr (subtract then unscale). `++`/`--` on a pointer slot emit `add/sub dword [...], sizeof(*ptr)` instead of `inc/dec`. `+` of two pointers and `int - ptr` are rejected. Scaling uses `shl`/`sar` for power-of-two sizes, `imul`/`idiv` otherwise.
- Arrays: `int arr[N]` allocates `N * sizeof(elem)` on the frame (rounded up to a 4-byte boundary); the slot's lowest byte is `arr[0]`. `_collect_locals` calls `_resolved_var_type(decl)` first — that fills in inferred sizes for `int arr[] = {...}` and `char s[] = "..."` from the initializer. Array names decay to addresses in expression context (`Identifier` of `ArrayType` lowers to `lea`, not `mov`). `arr[i]` lowers via `_index_address` (eval array → push, eval index → scale by element size → pop+add) followed by a width-correct load via `_load_to_eax`; stores use `_store_from_eax`. `&arr[i]` reuses `_index_address` without the deref. Assignment to an array name and `++`/`--` on an array name still raise. `int arr[N] = {a, b, c}` and `char s[] = "..."` are handled by `_array_init`: per-element stores via `_store_from_eax` (so `char arr[3] = {65, 66, 67}` writes bytes), then `mov <width> [...], 0` zero-fills any unfilled trailing elements. Designated initializers (`[2]=9`, `.field=x`) and nested `{}` for multidim arrays both work — a brace-elider handles the auto-AST `[StringLiteral]` shapes, and `offsetof` designators (`offsetof(S, in.b[2])`) resolve too.
- Sub-word codegen: `_load_to_eax(addr, ty)` and `_store_from_eax(addr, ty)` are the single chokepoints for slot/element access. Loads use `mov eax, [...]` for 4-byte values, `movsx eax, word [...]` / `movzx eax, word [...]` for shorts, and `movsx eax, byte [...]` / `movzx eax, byte [...]` for chars (zero-extension when `is_signed=False`, sign-extension otherwise). Stores narrow via `mov word [...], ax` or `mov byte [...], al`. The helpers are wired into Identifier read, `_var_init`, `_assign` (Identifier / `*ptr` / `arr[i]`), `_unary *`, `_index_load`, and `_inc_dec` (with `inc/dec byte` or `inc/dec word` for sub-word slots). Integer promotion happens implicitly because every load returns a 32-bit EAX value.
- Control flow: `if`/`else`, `while`, `do`/`while`, `for`, `switch`/`case`/`default`, `break`, `continue`, `goto`/labels. Labels are function-local (NASM `.LN_*`), generated via a per-function counter. `_FuncCtx` carries two parallel stacks — `break_targets` (pushed by both loops and switches) and `continue_targets` (pushed only by loops) — so `continue` inside a switch correctly escapes to the enclosing loop. User-declared labels are pre-walked into a name → NASM-label map so forward `goto`s resolve.
- Stack-machine evaluation: left → EAX → push, right → EAX → ECX, pop EAX, op. Comparisons land via `cmp` + `setCC al` + `movzx eax, al`. Division/modulo via `cdq` + `idiv ecx`. Right shift is `sar` (signed); will branch to `shr` when type info reaches codegen.
- Locals are allocated in a single recursive pre-pass over the whole function (including nested blocks, if-branches, loop bodies, for-init). Block scoping works, including shadowing: `int x = 1; { int x = 2; }` gives each `x` its own slot and the inner one wins inside the block (verified — prints `inner=2` / `outer=1`).
- ABI (current): cdecl. Caller pushes args right-to-left, callee accesses params via [ebp + 8 + accumulated-size]; scalars take 4 bytes, struct-by-value takes `sizeof(struct)` rounded up to 4. Caller cleans the stack. Return value in EAX. Struct-returning functions take a hidden first param (`__retptr__` at [ebp+8]) — caller pushes the destination address last so it lands at the leftmost arg slot, callee copies the value into `*__retptr__` and returns the same pointer in EAX so chained struct returns work without temps. Watcom register call (`__watcall`) is Phase 2 work in uc_core; once it lands we'll switch the default but keep cdecl reachable.
- Calls: `_call` dispatches direct vs indirect via `_emit_call`. A direct call is `call _name` when the callee is an Identifier whose name is in `_func_return_types`; otherwise the callee expression evaluates to EAX after args are pushed and we emit `call eax`. Leading `*`s on the callee (`(*fp)()`) are stripped — function-typed values are idempotent under `*` in C. A function name in value position (e.g. `int (*fp)() = helper;` or passing `helper` as an arg) decays to its label as an immediate via `_identifier_load`'s function-name fallback.
- Address rendering: `_ebp_addr(disp)` produces `[ebp - N]` for negative displacements (locals) and `[ebp + N]` for positive (params). Slot displacements live on `_FuncCtx.slots`.
- Globals: top-level VarDecls are registered in `CodeGenerator._globals` (resolved type) and `_global_inits` (init expr if present). Identifier access dispatches via the `_identifier_*` helpers — locals first (`ctx.slots`), globals second — so a function-scope local cleanly shadows a same-named global. Access lowers to `[_name]` (memory) or `_name` (immediate, for `&` and array decay). Initialized globals land in `.data` with `db`/`dw`/`dd` (or `dd v1, v2, ...` for arrays); uninitialized in `.bss` with `resb N`. Init expressions must reduce to compile-time constants — `_const_eval` handles literals, unary `-/+/~/!`, and the standard binary integer ops; references to other identifiers raise.


## Session log

Slice-by-slice notes live in `docs/changes.md` — a **historical log
closed on 2026-05-04**. It is not maintained and not a statement of
current capability; use `git log` for anything later.
