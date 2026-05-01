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
== py/ triage: 117 pass / 15 fail / 132 total ==
```

That's **89 % of the platform-independent core** compiling clean
through uc386 → NASM-ready .asm in one pass. The setup:

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
- `--no-ast-optimize` to dodge the uc_core alias-propagation bug
  (see "Bug surfaced" below).

The 15 remaining failures fall into:

| Class                                                                              | Count | Cause                                                                                                          |
|------------------------------------------------------------------------------------|-------|----------------------------------------------------------------------------------------------------------------|
| `global <obj>.sig: float init must be a constant expression`                       | 12    | `MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN` packs flags+min+max into an Identifier-bearing init that const-eval rejects. |
| `__static_scope__scope_simple_name_table: initializer index 6 out of range`        | 1     | Array of 6 `const char *` initialized with 7 entries — likely a parser quirk in our upstream version.           |
| `__static_objmodule__mp_builtin_module_table.key: got Identifier MICROPY_REGISTERED_MODULES` | 1     | Port-specific `MICROPY_REGISTERED_MODULES` macro is empty in stubs; real port supplies it.                     |
| Other                                                                              | 1     | Additional `.sig` flavour outside the dominant 12.                                                              |

## Bug surfaced (and worked around)

The `pp->m`-on-a-typedef case actually surfaced a real bug in the
**uc_core AST optimizer**, not in uc386 codegen. The shape that
trips it up:

```c
void f(void *data) {
    struct printer *pr = data;
    if (pr->flag) { ... }   // ← optimizer rewrites pr → data, type-of(data) is `void *`, error
}
```

The optimizer propagates `pr = data` and replaces later `pr`
references with `data`. That loses the declared `struct printer *`
type, so the codegen's `_type_of(pr)` returns `PointerType(void)`
instead, which the `->` lowering rejects. The triage build script
passes `--no-ast-optimize` to dodge this; a real port would want
the bug fixed in uc_core (the optimizer must respect declared types
of the propagation target).

**Earlier baselines**:
- 95/132 with empty qstrdefs (most failures were downstream of
  missing MP_QSTR enum entries, not separate bugs).
- 115/132 once the synthetic qstr table was in place.
- 117/132 (current) with `--no-ast-optimize` lifting the 2
  `pp->m` failures.

## Next steps for a runnable image

The triage proves the core is reachable. To land an actual
`micropython.bin`:

1. **Run upstream's `tools/makeqstrdefs.py`** to emit the real
   `genhdr/qstrdefs.generated.h` (correct hash + len fields,
   minus the over-inclusion the grep heuristic ships).
2. **Fix the remaining uc386/uc_core issues** that remain after
   the qstr table is correct:
   - `MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN`-style packed-flag
     bitfield-style init (the `.sig` family — 12 failures).
     Likely the `(MP_OBJ_FUN_FLAG_* | n_min)` shape includes
     an Identifier that needs to evaluate to a constant.
   - **uc_core optimizer alias-propagation type bug** — see
     "Bug surfaced" above. Currently worked around in build.sh
     via `--no-ast-optimize`; real port should fix it upstream.
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
