// uc386-dos MicroPython port config.
//
// Mirrors `upstream/ports/minimal/mpconfigport.h` (MICROPY_CONFIG_ROM_LEVEL_MINIMUM
// + REPL + GC + external import) but flips the knobs that don't fit
// the uc386 / DOS host:
//
//   - MICROPY_MODULE_FROZEN_MPY = 0 — we don't run mpy-tool.py, so
//     the frozen-module symbols (mp_frozen_mpy_content,
//     mp_frozen_names, mp_qstr_frozen_const_pool) don't exist; the
//     minimal port's =1 setting requires them as externs.
//   - MICROPY_QSTR_EXTRA_POOL stays unset (without frozen MPY there
//     is no second pool).
//   - MICROPY_MIN_USE_STDOUT = 1 — route mp_hal_stdin/stdout through
//     read(STDIN_FILENO, …) / write(STDOUT_FILENO, …); uc386's libc
//     turns those into INT 21h DOS syscalls.

#include <stdint.h>

// Use the setjmp/longjmp-backed NLR (non-local return) machinery.
// The default x86 NLR (`upstream/py/nlrx86.c`) is GCC inline asm,
// which uc_core / uc386 doesn't compile — `nlr_push`/`nlr_jump`
// would silently expand to no-ops, leaving uninitialized garbage in
// the nlr_buf and crashing the parser at the first exception. The
// setjmp path goes through uc386's libc `_setjmp`/`_longjmp`
// (lib/i386_dos_libc.asm), which is real i386 asm (saves 6 dwords:
// ebx/esi/edi/ebp/esp/eip). Pair this with the fix in
// `lib/include/setjmp.h` that widens jmp_buf to 24 bytes — the old
// 6-byte declaration would have caused setjmp to buffer-overflow.
#define MICROPY_NLR_SETJMP                (1)

// CORE_FEATURES baseline + selective EXTRA opt-ins. A previous
// experiment bumped ROM_LEVEL to EXTRA_FEATURES wholesale, which
// quietly broke the value-print path — `print(1)` hung the REPL
// (cause not yet root-caused; some EXTRA-default code path trips
// uc386 codegen). The selective approach below lets us cherry-pick
// the high-value EXTRA features without that fallout.
#define MICROPY_CONFIG_ROM_LEVEL          (MICROPY_CONFIG_ROM_LEVEL_CORE_FEATURES)

// EXTRA-gated features we explicitly opt into. Each addition was
// proven safe in isolation by a smoke-test pass.
#define MICROPY_PY_BUILTINS_MEMORYVIEW     (1)
#define MICROPY_PY_BUILTINS_COMPILE        (1)
#define MICROPY_PY_BUILTINS_INPUT          (1)
#define MICROPY_PY_BUILTINS_NEXT2          (1)
#define MICROPY_PY_COLLECTIONS_DEQUE       (1)
#define MICROPY_PY_MATH_CONSTANTS          (1)
#define MICROPY_PY_MATH_FACTORIAL          (1)
#define MICROPY_PY_MATH_ISCLOSE            (1)
#define MICROPY_PY_ALL_SPECIAL_METHODS     (1)
#define MICROPY_PY_ALL_INPLACE_SPECIAL_METHODS (1)
#define MICROPY_PY_REVERSE_SPECIAL_METHODS (1)
#define MICROPY_PY_BUILTINS_BYTES_HEX      (1)
#define MICROPY_PY_BUILTINS_STR_CENTER     (1)
#define MICROPY_PY_BUILTINS_STR_PARTITION  (1)
#define MICROPY_PY_BUILTINS_STR_SPLITLINES (1)
#define MICROPY_PY_BUILTINS_FROZENSET      (1)
#define MICROPY_PY_FSTRINGS                (1)
#define MICROPY_PY_FUNCTION_ATTRS          (1)
#define MICROPY_PY_DELATTR_SETATTR         (1)
#define MICROPY_REPL_EMACS_KEYS            (1)
#define MICROPY_REPL_AUTO_INDENT           (1)
#define MICROPY_KBD_EXCEPTION              (1)
#define MICROPY_PY_COLLECTIONS_DEQUE_ITER  (1)
#define MICROPY_PY_COLLECTIONS_DEQUE_SUBSCR (1)
#define MICROPY_PY_BUILTINS_BYTEARRAY_SLICE_ASSIGN (1)

#define MICROPY_ENABLE_COMPILER           (1)
#define MICROPY_ENABLE_GC                 (1)
#define MICROPY_HELPER_REPL               (1)
#define MICROPY_MODULE_FROZEN_MPY         (0)
#define MICROPY_ENABLE_EXTERNAL_IMPORT    (1)

#define MICROPY_ALLOC_PATH_MAX            (256)
#define MICROPY_ALLOC_PARSE_CHUNK_INIT    (16)

#define MICROPY_PY_SYS_MODULES            (0)
#define MICROPY_PY_SYS_EXIT               (0)
#define MICROPY_PY_SYS_PATH               (0)
#define MICROPY_PY_SYS_ARGV               (0)

// CORE_FEATURES (when we can re-enable it) default-enables
// MICROPY_PY_IO, which references `mp_builtin_open_obj` — a
// port-supplied symbol the uc386-dos port doesn't define (no VFS).
// Pre-set to 0 so a future ROM-level bump doesn't pull in
// `open()` / `io` machinery.
#define MICROPY_PY_IO                     (0)

// Selective opt-ins from EXTRA_FEATURES while staying at the
// CORE_FEATURES ROM level. These were originally MINIMUM-level
// opt-ins for the pre-CORE_FEATURES build; left in for two
// reasons: (a) they're explicit about what we use, and (b) they
// double as on-by-default at CORE_FEATURES so the macros are
// idempotent.
#define MICROPY_PY_BUILTINS_MIN_MAX       (1)
#define MICROPY_PY_BUILTINS_REVERSED      (1)
#define MICROPY_PY_BUILTINS_ENUMERATE     (1)
#define MICROPY_PY_BUILTINS_FILTER        (1)
#define MICROPY_PY_BUILTINS_PROPERTY      (1)

// EXTRA_FEATURES-gated extras worth pulling in to the
// CORE_FEATURES-baseline port. `OrderedDict` and `errno` are
// common-enough idioms that the small static-init cost is worth
// the import surface.
//
// errno requires:
//  - build.sh to pre-emit the EPERM/ENOENT/... qstrs (the module's
//    globals table uses `MP_QSTR_##e` token paste over its X-macro
//    list, which our grep-based gen_qstrdefs.py can't see otherwise).
//  - MICROPY_USE_INTERNAL_ERRNO=1 so MP_EPERM resolves to the
//    upstream's hardcoded MP_##e values rather than to the system's
//    EPERM macros from <errno.h>. uc386's libc errno.h ships only
//    a Linux subset (no EOPNOTSUPP / EADDRINUSE / ECONN* / EHOST* /
//    EALREADY / EINPROGRESS), so the system path leaves several
//    MP_##e references as bare Identifiers that fail const-eval.
#define MICROPY_PY_ERRNO                  (1)
#define MICROPY_USE_INTERNAL_ERRNO        (1)
#define MICROPY_PY_COLLECTIONS_ORDEREDDICT (1)

// Float support — uc386 lowers double through the x87 FPU and
// uc386's libc (lib/i386_dos_libc.asm) provides sin/cos/atan/
// atan2/exp/log/log10/pow/sqrt/ceil/floor/fabs/copysign/isnan/
// isinf/signbit in raw 387 asm. Other math functions
// (tan/asin/acos/fmod/trunc/ldexp/frexp/modf) are added
// alongside this opt-in. DOUBLE rather than FLOAT so the libc's
// `sin`/`cos`/... unsuffixed names match what micropython calls
// (FLOAT_IMPL_FLOAT calls `sinf` / `cosf` / ... which we don't
// have).
#define MICROPY_FLOAT_IMPL                (MICROPY_FLOAT_IMPL_DOUBLE)

// Pull in the hyperbolic / inverse-hyperbolic / log2 / expm1 /
// erf / erfc / tgamma / lgamma surface modmath.c gates on this
// flag. uc386's libc provides FPU-backed implementations for
// sinh/cosh/tanh/asinh/acosh/atanh/log2/expm1, an Abramowitz
// polynomial for erf/erfc, and NaN stubs for tgamma/lgamma
// (full Lanczos approximation is a future slice).
#define MICROPY_PY_MATH_SPECIAL_FUNCTIONS  (1)

// `time` module — `time.ticks_ms`, `time.sleep_ms`, etc. wired
// through INT 1Ah AH=0 BIOS tick counter (~18.2 Hz, ~55 ms/tick)
// in lib/i386_dos_libc.asm and uc386-dos/mphal_uc386dos.c.
//
// Default-off at CORE_FEATURES because upstream's ROM-level
// numbering puts BASIC_FEATURES (20) ABOVE CORE_FEATURES (10),
// so `MICROPY_PY_TIME = AT_LEAST_BASIC_FEATURES` evaluates to 0.
// Opt in here so `import time` works at our CORE_FEATURES
// baseline.
#define MICROPY_PY_TIME                    (1)

typedef long mp_off_t;

#define MICROPY_HW_BOARD_NAME "uc386-dos"
#define MICROPY_HW_MCU_NAME   "i386"

// Use the same STDOUT path the minimal port uses on linux/darwin —
// uc386's libc turns read(STDIN)/write(STDOUT) into INT 21h DOS calls.
#define MICROPY_MIN_USE_STDOUT (1)
// 256 KB GC heap. Bumped from 64 KB after the first runnable port:
// arbitrary input echoes back but the parser/compile path hits an
// UC_ERR_READ_UNMAPPED — heap exhaustion during parse-tree allocation
// is one likely cause. dos_emu maps a much wider data region than
// 64 KB so the additional fixed-region heap is fine.
#define MICROPY_HEAP_SIZE      (262144)

#define MP_STATE_PORT MP_STATE_VM
