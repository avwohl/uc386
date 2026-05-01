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
== py/ triage: 95 pass / 37 fail / 132 total ==
```

That's **72 % of the platform-independent core** compiling clean
through uc386 → NASM-ready .asm in one pass — using empty stubs
for the four upstream-generated headers (`qstrdefs.generated.h`,
`moduledefs.h`, `mpversion.h`, `root_pointers.h`) and a synthetic
`int main()` so uc386's "every TU needs main" check accepts library
sources. The 37 failures bucket two ways:

| Class                                                          | Count | Cause                                                                                                |
|----------------------------------------------------------------|-------|------------------------------------------------------------------------------------------------------|
| `unknown identifier MP_QSTR_*`                                 | 16    | Real qstrdefs needs `tools/makeqstrdefs.py` over the source tree first; stub is empty.               |
| `global mp_type_X.name: float init must be a constant expression (got Identifier)` | 21    | uc386 const-eval bug — `mp_type_X.name = MP_QSTR_X` (integer ID) routes through the float-init path. |

## Next steps for a runnable image

The triage proves the core is reachable. To land an actual
`micropython.bin`:

1. **Run upstream's qstr generator** to emit a real
   `genhdr/qstrdefs.generated.h` from the source tree. This
   resolves the 16 `MP_QSTR_*` identifier failures.
2. **Fix the const-eval Identifier-as-int regression in uc386
   codegen** (or, as a workaround, dynamically initialise the
   `mp_type_X` struct fields in module-init rather than at
   global scope). This resolves the remaining 21.
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
