// uc386-dos MicroPython port config — ROM_LEVEL = EXTRA_FEATURES.
//
// Started life as a clone of `upstream/ports/minimal/mpconfigport.h`
// (MINIMUM ROM level + REPL + GC + external import). Walked up the
// ROM ladder slice by slice until we hit EXTRA_FEATURES wholesale,
// fixing each new failure as it surfaced (qstr-hash field width,
// X-macro qstrs for errno, IEEE-754 inf/nan via __builtin_*,
// static-pool placement for special-method dunders, ...). See
// `addons/gnu/micropython/NOTES.md` for the per-slice diary.
//
// Port-incompatible defaults explicitly turned off below.

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

// EXTRA_FEATURES baseline. Lights up `compile()` / `eval()` /
// `exec()`, `input()`, `memoryview`, `MICROPY_PY_ALL_SPECIAL_METHODS`
// (`__add__` / `__radd__` / `__iadd__` / `__and__` / etc on class
// instances), `collections.deque`, math constants (pi/e/tau/inf/
// nan), `math.factorial`, `math.isclose`, `bytes.hex` /
// `bytes.fromhex`, `str.center` / `partition` / `splitlines`,
// `frozenset`, **f-strings**, function attribute access
// (`f.__name__`), `delattr`/`setattr`, REPL Emacs keys + auto-
// indent + Ctrl-C → KeyboardInterrupt, deque iter+subscr,
// bytearray slice-assign, plus dozens of smaller surface knobs.
#define MICROPY_CONFIG_ROM_LEVEL          (MICROPY_CONFIG_ROM_LEVEL_EXTRA_FEATURES)

// Port-incompatible defaults — all of these are EXTRA-default but
// need port-supplied helpers we don't ship today.
//
//   - MICROPY_STACK_CHECK: needs `mp_stack_set_top/_limit` calls in
//     main(); ports/minimal/main.c doesn't make them, every check
//     fails, and the stack-overflow raise path infinite-loops.
//   - MICROPY_PY_UCTYPES: pulls in `extmod/moductypes.c` which
//     we don't include in build_port.sh's source list; the missing
//     `mp_module_uctypes` extern crashes the NASM link.
//   - (was MICROPY_PY_TIME_TIME_TIME_NS — now enabled below)
//   - MICROPY_PY_BUILTINS_HELP: needs port-supplied help text
//     table; we don't ship one.
//   - MICROPY_MODULE___FILE__: needs source-path tracker.
#define MICROPY_STACK_CHECK               (0)
#define MICROPY_PY_UCTYPES                (0)
#define MICROPY_PY_BUILTINS_HELP          (0)
#define MICROPY_MODULE___FILE__           (0)

// `time.time()` / `time.localtime()` / `time.gmtime()` /
// `time.mktime()` — wired to the DOS RTC via INT 21h AH=0x2A
// (date) + AH=0x2C (time-of-day) in
// lib/i386_dos_libc.asm:_dos_get_datetime, then converted to
// seconds-since-epoch by upstream's shared/timeutils. The port
// shim lives in `uc386-dos/modtime_uc386dos.c` and is
// `#include`'d into extmod/modtime.c via the
// MICROPY_PY_TIME_INCLUDEFILE hook (provides
// `mp_time_time_get` + `mp_time_localtime_get`).
//
// MICROPY_TIMESTAMP_IMPL = 1 (UINT) forces `mp_timestamp_t` to
// be `mp_uint_t` (32-bit on i386). The default for
// MICROPY_EPOCH_IS_2000 is LONG_LONG, which we can't safely
// return from `time.time()` without longlong int support
// (`mp_obj_new_int_from_ll` is a stub in LONGINT_IMPL_NONE that
// always raises OverflowError). 32-bit unsigned seconds-since-
// 2000 covers through year 2136 — adequate for a DOS port.
//
// Caveat: TIME_TIME_NS gates BOTH `time.time()` AND
// `time.time_ns()` in upstream modtime.c. Calling
// `time.time_ns()` will raise `OverflowError("small int
// overflow")` because it routes through
// `mp_obj_new_int_from_ull(mp_hal_time_ns())` and the ull stub
// rejects everything. Lighting it up cleanly requires
// MICROPY_LONGINT_IMPL_LONGLONG.
#define MICROPY_TIMESTAMP_IMPL            (1)
#define MICROPY_PY_TIME_TIME_TIME_NS      (1)
#define MICROPY_PY_TIME_GMTIME_LOCALTIME_MKTIME (1)
#define MICROPY_PY_TIME_INCLUDEFILE       "uc386-dos/modtime_uc386dos.c"

// `open()` + `import xxx` (loading `xxx.py` from disk) wired through
// uc386's libc INT 21h file syscalls. We provide port-supplied
// `mp_builtin_open_obj` / `mp_import_stat` / `mp_lexer_new_from_file`
// in `uc386-dos/file_uc386dos.c` (no full VFS — just enough for
// flat .py imports and read/write file objects). MICROPY_PY_IO=1
// pulls in modio.c (io.IOBase, BytesIO, StringIO) and exposes
// `open` in the builtins table.
#define MICROPY_PY_IO                     (1)

// `sys`-module sub-features. We have basic `sys` (sys.platform,
// sys.implementation, sys.maxsize) but no sys.modules cache /
// sys.exit / sys.path / sys.argv — those need port-supplied
// state.
#define MICROPY_PY_SYS_MODULES            (0)
#define MICROPY_PY_SYS_EXIT               (0)
#define MICROPY_PY_SYS_PATH               (0)
#define MICROPY_PY_SYS_ARGV               (0)

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
#define MICROPY_USE_INTERNAL_ERRNO        (1)

// Float — uc386 lowers `double` through the x87 FPU and uc386's
// libc (lib/i386_dos_libc.asm) ships sin/cos/tan/asin/acos/atan/
// atan2/exp/log/log10/log2/expm1/pow/sqrt/floor/ceil/trunc/fmod/
// modf/fabs/copysign/signbit/isnan/isinf/isfinite/nan/nearbyint/
// ldexp/frexp/sinh/cosh/tanh/asinh/acosh/atanh/erf/erfc plus NaN
// stubs for tgamma/lgamma in raw 387 asm. DOUBLE rather than
// FLOAT so the libc's `sin`/`cos`/... unsuffixed names match what
// micropython calls (FLOAT_IMPL_FLOAT calls `sinf` / `cosf` /
// ... which we don't have).
#define MICROPY_FLOAT_IMPL                (MICROPY_FLOAT_IMPL_DOUBLE)
#define MICROPY_PY_MATH_SPECIAL_FUNCTIONS (1)

// `time` module — `time.ticks_ms`, `time.sleep_ms`, etc. wired
// through INT 1Ah AH=0 BIOS tick counter (~18.2 Hz, ~55 ms/tick)
// in lib/i386_dos_libc.asm and uc386-dos/mphal_uc386dos.c.
// Default-off at our ROM_LEVEL because upstream's ROM-level
// numbering puts BASIC_FEATURES (20) ABOVE CORE_FEATURES (10),
// so `MICROPY_PY_TIME = AT_LEAST_BASIC_FEATURES` evaluates to 0.
#define MICROPY_PY_TIME                   (1)

#define MICROPY_ENABLE_COMPILER           (1)
#define MICROPY_ENABLE_GC                 (1)
#define MICROPY_HELPER_REPL               (1)

// We don't run mpy-tool.py, so the frozen-module symbols
// (mp_frozen_mpy_content, mp_frozen_names,
// mp_qstr_frozen_const_pool) don't exist; the minimal port's =1
// setting requires them as externs.
#define MICROPY_MODULE_FROZEN_MPY         (0)
#define MICROPY_ENABLE_EXTERNAL_IMPORT    (1)

#define MICROPY_ALLOC_PATH_MAX            (256)
#define MICROPY_ALLOC_PARSE_CHUNK_INIT    (16)

typedef long mp_off_t;

#define MICROPY_HW_BOARD_NAME "uc386-dos"
#define MICROPY_HW_MCU_NAME   "i386"

// Use the same STDOUT path the minimal port uses on linux/darwin —
// uc386's libc turns read(STDIN)/write(STDOUT) into INT 21h DOS calls.
#define MICROPY_MIN_USE_STDOUT (1)

// 256 KB GC heap. Bumped from 64 KB after the first runnable port:
// arbitrary input echoes back but the parser/compile path hits an
// UC_ERR_READ_UNMAPPED — heap exhaustion during parse-tree
// allocation is one likely cause. dos_emu maps a much wider data
// region than 64 KB so the additional fixed-region heap is fine.
#define MICROPY_HEAP_SIZE      (262144)

#define MP_STATE_PORT MP_STATE_VM
