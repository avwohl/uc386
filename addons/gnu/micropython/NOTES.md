# MicroPython port — status: **skeleton + triage** (2026-05-01)

**Upstream**: https://github.com/micropython/micropython
**License**: MIT
**Effort**: multi-day. MicroPython is ~100 K LoC of C; a uc386 port
needs an upstream-style `ports/uc386-dos/` directory plus a fixed-
region GC heap and a thin HAL backed by INT 21h. Today this addon
is a **triage skeleton** — it answers the prerequisite question
"how much of the platform-independent core compiles cleanly via
uc386?" before we sink time in a real port.

## Build

```sh
./fetch.sh    # clones micropython upstream into upstream/
./build.sh    # per-file triage of upstream/py/*.c through uc386
              # writes build/<name>.asm on PASS, build/<name>.err
              # on FAIL; build/triage.txt is the per-source ledger,
              # build/errors.txt the histogram.
```

## Triage result (latest run)

```
== py/ triage: 131 pass / 1 fail / 132 total ==
```

That's **99 % of the platform-independent core** compiling clean
through uc386 → NASM-ready .asm in one pass. The single remaining
failure is `objmodule.c`'s `MICROPY_REGISTERED_MODULES` — that's
a port-specific macro listing the modules the port wants to
register, empty in our triage stubs. A real port (e.g.
`ports/uc386-dos/mpconfigport.h`) supplies it as part of the
normal build setup. **Effectively all of py/ that can compile
without port-specific config does compile.**

The setup:

- Stub `genhdr/moduledefs.h`, `genhdr/mpversion.h`,
  `genhdr/root_pointers.h` (empty headers — real builds emit these
  from the source tree).
- Auto-generate a triage `genhdr/qstrdefs.generated.h` by grepping
  `MP_QSTR_*` references out of `upstream/py/` and emitting the
  matching `QDEF0(...)` macro invocations. Approximates upstream's
  `tools/makeqstrdefs.py` over-inclusively (any MP_QSTR_x pattern
  becomes a qstr, even if it's only a comment in real source) but
  keeps the enum in `py/qstr.h` complete enough that downstream
  refs resolve.
- Synthetic `int main()` so uc386's "every TU needs `main`" check
  accepts library sources.

The single remaining failure:

| Class                                                                              | Count | Cause                                                                                                          |
|------------------------------------------------------------------------------------|-------|----------------------------------------------------------------------------------------------------------------|
| `__static_objmodule__mp_builtin_module_table.key: got Identifier MICROPY_REGISTERED_MODULES` | 1     | Port-specific `MICROPY_REGISTERED_MODULES` macro is empty in stubs; real port supplies it.                     |

## Bug surfaced (and fixed)

The `pp->m`-on-a-typedef case surfaced a real bug in the
**uc_core AST optimizer's copy-propagation path**. The shape that
tripped it up:

```c
void f(void *data) {
    struct printer *pr = data;     // legal C: void* → struct*
    if (pr->flag) { ... }           // ← optimizer rewrote pr → data
}
```

`_types_compatible_for_copy` happily propagated `pr = data` because
both sides are PointerType. But replacing `pr` with `data` loses the
declared `struct printer *` type — `_type_of(data)` returns
`PointerType(void)`, which uc386's `->` lowering rejects.

**Fix** (in `uc_core/src/uc_core/ast_optimizer.py`): refuse copy
propagation between two PointerTypes when either side's pointee is
`void`, or when the pointee kinds differ (one BasicType, one
StructType, etc.). Equivalent pointers (e.g. `int *` to `int *`)
still propagate.

**Triage progression**:
- 95/132 with empty qstrdefs (most failures were downstream of
  missing MP_QSTR enum entries, not separate bugs).
- 115/132 once the synthetic qstr table was in place.
- 117/132 with the uc_core copy-prop fix lifting the 2 `pp->m`
  failures.
- 130/132 once `_const_eval` learned `TernaryOp` (lifted the 12
  packed-flag `.sig` failures from `MP_OBJ_FUN_MAKE_SIG`'s
  `(takes_kw) ? 1 : 0` ternary).
- 131/132 (current) once `_resolved_var_type` learned to const-
  eval enum-constant designators (lifted the
  `[SCOPE_GEN_EXPR] = ...`-style array-size mis-inference).

## Next steps for a runnable image

The triage proves the core is reachable. To land an actual
`micropython.bin`:

1. **Run upstream's `tools/makeqstrdefs.py`** to emit the real
   `genhdr/qstrdefs.generated.h` (correct hash + len fields,
   minus the over-inclusion the grep heuristic ships).
2. **Compiler fixes for MicroPython idioms** — all already shipped
   as part of this slice:
   - **uc_core**: copy-propagation refuses propagation across
     `void *` and across pointee-kind boundaries (was rewriting
     `void *data → struct *p` propagations and losing struct
     type for later `p->m`).
   - **uc386 const-eval**: `TernaryOp` + comparison + `&&`/`||`
     now fold (lifted the `MP_OBJ_FUN_MAKE_SIG`'s `.sig` family).
   - **uc386 array sizing**: `_resolved_var_type` const-evals
     enum-constant designators when inferring an unsized array's
     length (lifted the `static const T arr[] = { [ENUM] = … }`
     class).
3. **Write `ports/uc386-dos/`** — a thin port with:
   - `mpconfigport.h` (start from `ports/minimal/`)
   - `main.c` calling `mp_init` / `pyexec_friendly_repl` with
     a fixed-region heap.
   - `mphalport.c` — `mp_hal_stdout_tx_strn` → INT 21h AH=09;
     `mp_hal_stdin_rx_chr` → INT 21h AH=01; `mp_hal_ticks_ms`
     → INT 1Ah BIOS time.
   - GC-aware setjmp/longjmp; uc386 already lowers them via
     libc, but the GC root scan needs to know where the stack
     range is — the port shim wires that up via
     `MP_STATE_THREAD(stack_top)`.
4. **Compile + link multi-file** through uc386, using the same
   pattern as the doom port (single uc386 invocation over the
   whole TU set). Existing multi-file affordances (file-scope
   `static` mangling, structural anonymous-struct identity, etc.)
   are already in place.

## Build artefacts

`build/` contains per-source `.asm` (one per PASS), `.err`
(stderr per FAIL), and the two roll-ups `triage.txt` and
`errors.txt`. None of these ship in the release tarball — they
are dev-side intermediate output.

## License

MIT — see `upstream/LICENSE` after running `./fetch.sh`. The thin
uc386 port shim, when written, inherits GPL-3.0 from the parent
uc386 repo (matches the convention used by the in-tree GNU
utility addons).
