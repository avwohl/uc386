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

#define MICROPY_CONFIG_ROM_LEVEL          (MICROPY_CONFIG_ROM_LEVEL_MINIMUM)

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
