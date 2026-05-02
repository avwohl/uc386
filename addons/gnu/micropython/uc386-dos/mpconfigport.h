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

// CORE_FEATURES. The earlier "every named-builtin NameErrors at
// runtime" regression turned out to be a missing qstr-hash:
// CORE_FEATURES sets `MICROPY_QSTR_BYTES_IN_HASH = 1`, which adds
// a `hashes[]` array to each `qstr_pool_t` and gates
// `qstr_find_strn`'s post-binary-search filter on
// `pool->hashes[at] == str_hash`. Our `gen_qstrdefs.py` was
// emitting `0` for every QDEF1's hash field, so every static
// lookup missed and `print` / `min` / `__name__` raised
// NameError. Fix: gen_qstrdefs.py now computes the djb2 hash
// (mirroring upstream's `tools/makeqstrdata.py:compute_hash`,
// including the `(hash & mask) or 1` zero-fix) via build.sh's
// `--bytes-hash 1`. CORE_FEATURES then boots cleanly and unlocks
// a full builtins surface — `bytearray` / `slice` / `set` types,
// most `MICROPY_PY_BUILTINS_*` defaults, qstr-named error
// messages, and the rest of the gates that default-enable here.
#define MICROPY_CONFIG_ROM_LEVEL          (MICROPY_CONFIG_ROM_LEVEL_CORE_FEATURES)

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

// Selective opt-ins from CORE_FEATURES while staying at the
// MINIMUM ROM level. These builtins are pure-functional with no
// module-state dependency, so they pull in self-contained .c
// files (modbuiltins.c, objreversed.c, objenumerate.c, objfilter.c,
// objproperty.c) without touching the builtins-init code-path that
// broke under a full ROM-level bump.
#define MICROPY_PY_BUILTINS_MIN_MAX       (1)
#define MICROPY_PY_BUILTINS_REVERSED      (1)
#define MICROPY_PY_BUILTINS_ENUMERATE     (1)
#define MICROPY_PY_BUILTINS_FILTER        (1)
#define MICROPY_PY_BUILTINS_PROPERTY      (1)

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
