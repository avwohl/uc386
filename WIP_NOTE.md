# WIP — resume notes

Last known state: clean tree on origin/main at commit `b92f8ae` (peephole:
slot_constant_propagation first-reference check).

## Status of test suites

- 1268 unit tests passing
- 220/220 c-testsuite (--full)
- 1514/1514 gcc-c-torture (--full)
- pr70602 benchmark: 471 → 52 lines (89% reduction)

## What was about to start: slice 42

While probing for the next pattern, looked at `_sum` from a sample probe:

```c
int g = 5;
int sum() {
    int total = 0;
    for (int i = 0; i < g; i++)
        total += g;
    return total;
}
```

The codegen emits `mov eax, [_g]` twice (once for `i < g` cmp, once for
`total += g`). The second load is redundant — `_g` isn't written between
them and EAX still holds [_g]'s value. But the existing
`redundant_eax_load` pass only tracks ebp-relative memory, not labels.

Idea for slice 42: extend `_run_redundant_reg_load` to also track
label-only memory (`[_glob]`). Aliasing concerns:
- Same-label store: aliases (already handled by textual-equality check).
- Different-label store: disjoint.
- `mov [reg], val` (register-base store): could alias if reg happens to
  hold the global's address. Look up `lea reg, _glob` per-function (similar
  to `_compute_addr_taken_per_line` but for label addresses).
- `mov [ebp ± N], val` (frame store): disjoint from globals.
- Calls: clobber globals conservatively (already handled by `op == "call"`
  invalidation).

Hooks needed:
- New helper `_is_label_mem(text)` — matches `[_label]` (no offset/SIB).
- New helper `_compute_label_addr_taken_per_line(lines)` — returns set of
  global labels whose address is taken via `lea reg, _label`.
- Extend `_mem_disjoint_with_taken` (or sister helper) to handle label vs
  other-memory aliasing.
- Track `[_glob]` strings in `reg_mem` alongside `[ebp ± N]` strings (the
  textual-equality check already works for both).

Other slice 42 candidates considered:
- Tail call optimization for noreturn calls (`call _abort` → `jmp _abort`):
  saves 0 bytes (both rel32). Not useful unless target is rel8-reachable.
- BSS-empty section elimination: `_bss_zero_start: _bss_zero_end:` labels
  emitted even when no globals exist. Could detect and drop the section
  entirely. Modest savings.
- Function-level constant return: if every code path in a function returns
  the same constant, replace with `mov eax, IMM; ret`. More complex —
  needs CFG analysis.

## Resume command

On the new machine:

```bash
cd /Users/wohl/src/uc386  # or equivalent
git pull
.venv/bin/pytest tests/ -q  # confirm green
```

Then continue with slice 42 (label-memory tracking in
`_run_redundant_reg_load`).

Per autonomous-loop directive:
- Implement, verify (unit + c-testsuite + torture --full), commit, push,
  delete this WIP_NOTE.md, continue to slice 43.
