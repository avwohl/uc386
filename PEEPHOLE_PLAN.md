# uc386 — Optimization Plan

Roadmap for matching uc80's "smallest binary" model on x86-32. This is a
working plan, not a contract — order and scope will shift as we hit
concrete wins.

## Why now

uc386's codegen currently emits NASM directly, line by line, with no
post-pass. The frontend (uc_core's `ast_optimizer.py`) does a fair amount
already — strength reduction, constant folding, copy propagation, dead
store elimination — but everything below the AST is verbatim from the
emitter. As a result the asm has visible stack-machine slack: every
binop pushes the left operand, evaluates the right into EAX, then
recovers the left into ECX via pop. See `/tmp/probe_misc31h.asm` —
identical with and without `-O` because there's nothing to compare it
against.

The uc80 numbers are the target. Per its README:
- 47/47 of the Fujitsu test suite smaller than z88dk-SDCC
- Aggregate uc80 binary size 46% of z88dk's
- `hello world (puts)` is **256 bytes** vs. z88dk's 5,172
- Minimal binary is **128 bytes**

That gap comes from a stack of optimizations, not a single silver
bullet. uc386 needs the same stack.

## What uc80 does (model to copy)

1. **AST-level optimization** — already shared with uc386 via
   `uc_core/ast_optimizer.py`. Nothing to do here.
2. **Whole-program codegen** — `uc80 a.c b.c c.c -o prog.mac` merges
   ASTs into one TranslationUnit before codegen. Enables:
   - **Dead function elimination** at the AST level (a function that
     no live function calls is dropped before emission).
   - **Function inlining** at call sites.
   - **Interprocedural constant propagation**.
   - **Shared storage** — non-recursive functions skip the
     stack-frame dance and use static scratch instead. Big win on Z80
     because frame setup is several instructions; smaller win on i386
     but still real.
3. **Embedded runtime** — libc lives as `.mac` files split by function;
   the driver embeds only the functions actually called (transitively).
   Plus all the EXTRN references the codegen itself emits.
4. **Assembly DCE** — `asm_dce.py` parses the assembled output into
   labeled blocks, walks reachability from `_main` + PUBLIC labels, and
   drops unreachable blocks plus their referenced data. Catches
   functions whose only callers were themselves dropped, plus
   unreachable arms of switch statements that the AST optimizer didn't
   see.
5. **Peephole** — `upeepz80` package, separate repo. Pattern-based
   rewriter on the asm text. Multiple passes to fixed point.
6. **Printf auto-detection** — scans format strings in printf-family
   calls (`printf`, `fprintf`, `sprintf`, `snprintf`, `vprintf`,
   `vfprintf`, `vsprintf`) and only links the handlers actually used.
   `--printf int` / `--printf float` / `#pragma printf int` etc. let
   you override. Special case: a `printf` whose format string has no
   specifiers gets rewritten to `puts` (or printed inline as raw bytes).

## What uc386 has today

- `src/uc386/codegen.py` — direct NASM emitter, no post-passes.
- `src/uc386/main.py` — single-file driver, basic CLI.
- `src/uc386/runtime.py` — stub (the actual libc lives in
  `lib/i386_dos_libc.asm`, currently embedded as one big blob).
- `lib/i386_dos_libc.asm` — monolithic. Every uc386-compiled binary
  pulls in printf, malloc, qsort, the FPU helpers, `__builtin_*`
  intrinsics, file I/O, signal — even if main is `return 0;`.

So we're starting from zero for items 2–6 above.

## Phases

### Phase A — Inline peephole

Build the optimizer inline in uc386 first, let it grow concrete
patterns, then extract to `upeep386` once the API has shape. The
lesson from `upeepz80` is that the abstractions (instruction parsing,
pattern DSL, statistics, multi-pass driver) only become clear after
you've written ~30 patterns and can see what they have in common.
Designing the package boundary up front means guessing.

Layout when it lands:
- `src/uc386/peephole.py` — line-based asm rewriter.
- `tests/test_peephole.py` — pattern-by-pattern tests with concrete
  before/after snippets.
- Wire-up in `codegen.py`: post-process the asm text just before the
  driver writes it.
- CLI flag: `--no-peephole` to disable.

#### Patterns (initial, ordered by visible-asm impact)

The current asm has a few signature shapes worth attacking first.
Numbers below come from `/tmp/probe_misc31h.asm`.

**P1: Dead `xor eax, eax` after unconditional `jmp .epilogue`.**
Every function ends with:
```
        mov     eax, 0
        jmp     .epilogue
        xor     eax, eax     ; ← dead
.epilogue:
```
Trivial pattern: any instruction between an unconditional jump and a
label is dead. One match per function; saves 2 bytes each.

**P2: `mov eax, simple_src; push eax; mov eax, simple_src2; pop ecx; mov [ecx], eax`.**
This is the canonical "store through computed pointer" pattern. The
push/pop pair is wasteful when the middle is a simple `mov eax, IMM`
or `mov eax, [...]` that doesn't read ECX:
```
        mov     eax, [ebp - 4]
        push    eax              ; save addr
        mov     eax, 100         ; value
        pop     ecx              ; restore addr → ECX
        mov     [ecx], eax
```
→
```
        mov     ecx, [ebp - 4]
        mov     eax, 100
        mov     [ecx], eax
```
Saves 2 instructions per match. Conditions: addr's source doesn't
reference ECX (true for `[ebp - N]`, immediates, labels — fails for
`[ecx]` and `[ecx + N]`); middle doesn't write ECX (fails for `call`,
`shl/shr/sar cl`, nested binop with same shape).

**P3: Stack-machine collapse for binops with simple right side.**
```
        mov     eax, A         ; left
        push    eax
        mov     eax, B         ; right (single instruction, no ECX touch)
        mov     ecx, eax
        pop     eax
        add     eax, ecx
```
→
```
        mov     ecx, B
        mov     eax, A
        add     eax, ecx
```
Saves 3 instructions. Same conditions as P2 plus: B doesn't depend on
the previous EAX (typical when B is a literal or memory load).

**P4: Redundant `mov reg, EAX` followed by reg overwrite.**
```
        mov     ecx, eax
        ...                    ; doesn't read ECX
        mov     ecx, X         ; clobbers
```
→ drop the first mov.

**P5: `add eax, 0` / `imul eax, 1` / `shl eax, 0`.**
The AST optimizer should be catching most of these but defensively
sweep at the asm level.

**P6: `mov eax, IMM` followed by `test eax, eax; jz LABEL`.**
Resolves at compile time. Either drop the test+jz (when IMM != 0) or
replace with unconditional `jmp LABEL` (when IMM == 0). Catches cases
where the AST optimizer left a constant test in place (e.g. after
inlining).

**P7: Tail-call: `call X; ret` → `jmp X`.** Standard. Saves the call
overhead and one instruction.

**P8: Sign-extend redundancy.** `movsx eax, al` immediately after
something that wrote EAX with a signed value already in the high
bits. Detection requires a small dataflow window.

**P9: Local label threading.** `jmp .L1` where `.L1:` is an
unconditional `jmp .L2` — thread directly to `.L2`.

#### Multi-pass driver

Patterns can enable each other. P1 plus P7 plus label threading often
compound. Run patterns in a fixed order until no rewrites occur (cap
at 10 passes to bound runtime; uc80's experience says 3–4 passes is
typical).

#### Statistics

Each pattern increments a counter so we can see what's actually
firing. Print under `-v` like uc80 does:
```
peephole optimizations:
  dead_after_jmp: 12
  push_pop_collapse: 47
  stack_collapse_binop: 31
  ...
```

### Phase B — Assembly DCE

After peephole, before final write. NASM-flavored cousin of
`uc80/src/asm_dce.py`. Different parsing rules:
- Sections are `section .text` / `section .data` / `section .bss`
  rather than CSEG/DSEG/COMMON.
- Public labels declared via `global _name` rather than
  `PUBLIC _name`.
- Externs are `extern _name` not `EXTRN`.
- Local labels start with `.` and are scoped to the previous
  global. The reachability walker has to track which global a `.L_*`
  belongs to.
- Jump tables are `dd _name, _name2, ...` in `.data` (or wherever
  computed-goto lookup tables live) — successor analysis must follow
  them.
- Address-taken: `mov eax, _name` for function pointers, `lea eax,
  [_name]` for arrays.

Entry points: `_start` (driver-emitted entry) plus any `global _name`
the user marked. In single-file mode `_main` is implicitly an entry.

Worth noting: the AST optimizer's dead-code passes don't catch
unreachable blocks within a function (e.g., `if (0) { ... }` after
inlining a constant condition), and they don't catch functions whose
only callers were themselves dropped. Asm DCE picks both up.

### Phase C — Multi-file / whole-program

Today: `uc386 file.c -o file.asm` takes one .c file.
Goal: `uc386 a.c b.c c.c -o prog.asm` merges them, emits one .asm.

The shape from uc80's main.py:
1. Loop over input files. For each:
   a. Preprocess.
   b. Lex.
   c. Parse.
   d. Append to `asts: list[TranslationUnit]`.
2. Merge: `merged = TranslationUnit(declarations=sum(a.declarations for a in asts, []))`.
3. AST-optimize the merged tree.
4. Codegen once.

Frontend cleanup: each file has its own `#include`s and `#pragma`s.
The preprocessor instance has to be per-file (so `#define` in a.c
doesn't leak into b.c) but the printf-feature set merges across
files. Identical `static` symbols in different files have to remain
distinct — uc_core needs to mangle `static` names with the file
basename.

CLI flag: `--no-whole-program` falls back to per-file emission for
tools that need separate compilation. uc80 has the same toggle.

Default: whole-program ON when there's >= 1 input file. Single-file
case is just the trivial whole-program.

### Phase D — Embedded runtime with selective inclusion

Today: `lib/i386_dos_libc.asm` is one big blob, copied verbatim into
every output.

Goal: split into per-function units, embed only the functions
transitively reachable from the user's code.

The work:
1. **Split `lib/i386_dos_libc.asm`** into per-function `.asm` files
   (one per global symbol the function exports). uc80 has
   `lib/split_libc.py` doing exactly this from a monolithic source —
   port the script.
2. **Build a `RuntimeLibrary` parser** matching uc80's
   `runtime.py` — extract `name → AsmFunction(source, deps, externs)`
   tuples. NASM-flavored: track `global` for exports, `extern` for
   deps, `call _name` / `mov reg, _name` for transitive references.
3. **Driver integration**: after codegen, scan the output for
   referenced runtime symbols, walk transitively, embed only those
   functions.
4. **Data section sharing**: many libc functions share scratch buffers
   (`__sret_buf`, `__tmp_qword`, etc.). Emit only the data labels
   referenced by the embedded function set.

Important: this comes AFTER asm DCE in the pipeline. Asm DCE can drop
codegen-emitted blocks that became unreachable post-inlining; that
in turn can drop runtime functions whose only caller was dropped.
Run-DCE-then-runtime would miss those cases, so the order is:
codegen → peephole → asm DCE on user code → embed runtime → asm DCE
on runtime → peephole on full → final write.

### Phase E — Printf auto-detection + puts rewrite

Two parts.

**Part 1: scan format strings, link only needed handlers.**

Port uc80's `_auto_detect_printf_features` from
`/Users/wohl/src/uc80/src/codegen.py:2672`. The scanner walks the AST
looking for calls to `printf` / `fprintf` / `sprintf` / `snprintf` /
`vprintf` / `vfprintf` / `vsprintf` with literal format strings, and
collects `{int, long, llong, float}` features per the conversions
seen.

Non-literal format → fall back to `all` (link everything).

A program with only `%d` / `%s` / `%c` doesn't need the float
formatting machinery. On i386 the savings are smaller than Z80
(printf's float path is dos_emu-side anyway), but the libc-side
narrow-int handling still has overhead.

The split here happens at the libc level: `_printf` is one entry
point; the dispatch on conversion letter routes to per-conversion
handlers (`__pf_int`, `__pf_long`, `__pf_long_long`, `__pf_float`).
Asm DCE then drops the unused handlers.

**Part 2: puts rewrite.**

When a `printf` call has a literal format string with NO conversion
specifiers (only `%%` and literal text), and the trailing character is
`\n`, rewrite to `puts("...")` (which calls `puts` and which is much
smaller than a full `printf`). When there's no trailing `\n`, rewrite
to `fputs(stdout, "...")` or just direct INT 21h depending on what's
cheapest.

The rewrite happens at the AST level (after the auto-detect scan).
uc80 does it in codegen as a special case in the printf call site.

CLI: `--printf int|long|float|all` (additive, multiple -P flags
stack). `#pragma printf int` in source. Both add to the auto-detected
set; explicit `--printf` overrides auto-detection.

### Phase F — Shared storage for non-recursive functions

uc80 detects functions that are never called recursively (directly or
indirectly) and gives them static scratch instead of stack frames.
On Z80 that saves the IX/IY frame setup; on i386 it saves the
`push ebp; mov ebp, esp; sub esp, N; ... mov esp, ebp; pop ebp`
boilerplate (5 instructions, ~10 bytes).

The analysis is a strongly-connected-component pass over the call
graph. Functions in their own SCC of size 1 with no self-edge are
non-recursive; they can use a single static scratch area sized to
the function's max stack usage. Functions in the same SCC that don't
call each other recursively can share scratch.

This is more invasive than the other phases — touches frame layout
in codegen — and the i386 win is smaller. Defer until A–E land.

### Phase G — Function inlining

uc80 inlines small functions at their call sites. Heuristics: leaf,
short, called few times, not recursive. The AST optimizer is the
right place for this (it's already what handles `__builtin_constant_p`
and friends). Lives in `uc_core/ast_optimizer.py` so both uc80 and
uc386 get it.

Already partly in place: `gnu_inline` `extern inline` functions
(slice 2026-04-28) inline at the call site by AST substitution. That
machinery generalizes to any small leaf function.

### Phase H — Interprocedural constant propagation

If a function is only ever called with a constant argument for some
parameter, propagate the constant into the body. Combined with
inlining and DCE this catches a lot of "configuration" calls. Lives
in `uc_core` like inlining.

### Phase I — Extract upeep386

Once the inline peephole has 30+ patterns and the API has stabilized:
- Extract to `upeep386` Python package.
- Move pattern source there.
- uc386 imports it like uc80 imports upeepz80.
- API mirrors upeepz80's: `optimize(asm_text) -> str`,
  `PeepholeOptimizer().stats`.

## Pattern-shopping checklist

When working on phase A, the actually-impactful patterns come from
looking at real asm. Do this iteratively:
- Pick a representative test (smoke test, ctest, torture sample).
- Compile both with and without peephole.
- Diff. Look for the biggest unchanged ugly-spot.
- Find a pattern that catches it.
- Verify all suites still pass.
- Repeat.

uc80's experience: ~80% of the wins come from ~10 patterns. The long
tail of pattern fragments mostly catches things the AST optimizer
already handles upstream.

## Test discipline

Every pattern gets:
- A unit test in `tests/test_peephole.py` with a concrete input/output
  asm snippet.
- A negative test for at least one near-miss case (e.g., the pattern
  doesn't fire when the middle has a `call`).
- The pre-existing torture / c-testsuite / smoke runs must stay 100%
  green. The bar is no regressions; the peephole is value-additive.

## What this DOESN'T plan to do

- **Register allocation.** The current EAX-everywhere stack machine
  with ECX as the spill is a known limitation but rewriting codegen
  to a real graph-coloring allocator is out of scope. Some peephole
  patterns (P3, P4) will recover a fraction of the win.
- **Loop unrolling / vectorization.** Out of scope for a DOS target.
- **Schedule.** Reordering for instruction-level parallelism. The
  386 doesn't reward it enough to bother.
- **SSA-based optimization.** Way too much machinery for the wins
  available at this layer; the AST optimizer covers the cases that
  matter.

## Order of attack

1. **Phase A** — inline peephole, P1 + P2 + P3 first, see how far they
   take us.
2. **Phase B** — asm DCE.
3. **Phase E** — printf auto-detection + puts rewrite (cheap big win
   for "hello world"-class binaries).
4. **Phase D** — split runtime + selective embed.
5. **Phase C** — multi-file. (Hold until D so we have something to
   share across translation units.)
6. **Phase G** — inlining (in uc_core).
7. **Phase H** — interprocedural const-prop (in uc_core).
8. **Phase F** — shared storage. Lowest priority on i386.
9. **Phase I** — extract upeep386.

A and B alone should knock the typical binary down by 20–30%
(eyeballing the slack in `/tmp/probe_misc31h.asm`). E + D should
shrink the floor for trivial programs by 90%+, matching uc80's
"hello world = 256 bytes" result.
