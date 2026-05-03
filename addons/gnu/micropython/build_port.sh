#!/bin/sh
# Multi-TU build of MicroPython through uc386 — produces a runnable
# `build/micropython.bin` that boots to the REPL banner under
# `uc386.dos_emu.run`.
#
# Sources:
#   - upstream/py/*.c                        (132)
#   - upstream/shared/libc/printf.c
#                       /string0.c
#                       /__errno.c
#                       /abort_.c
#   - upstream/shared/readline/readline.c
#   - upstream/shared/runtime/pyexec.c
#                            /stdout_helpers.c
#                            /interrupt_char.c
#                            /sys_stdio_mphal.c
#   - upstream/ports/minimal/main.c          (REPL driver)
#   - upstream/ports/minimal/uart_core.c     (mp_hal_stdin/stdout)
#
# Includes:
#   - lib/include                            (uc386 libc)
#   - upstream                               (py/, shared/)
#   - uc386-dos                              (mpconfigport.h, mphalport.h)
#   - build                                  (genhdr/qstrdefs.generated.h,
#                                             moduledefs.h, mpversion.h,
#                                             root_pointers.h — emitted
#                                             by ./build.sh during the
#                                             per-file triage pass)
#
# Defines:
#   -D__linux__=1                            triggers the minimal port's
#                                            MICROPY_MIN_USE_STDOUT path
#                                            (read(STDIN_FILENO)/write(...))
#                                            — uc386 lowers those into
#                                            INT 21h DOS syscalls.
#
# Wall-clock: ~14 minutes on the dev Mac (Apple Silicon, Python 3.12,
# uc386 in-process). Multi-TU is heavy because uc_core's preprocessor
# runs once per TU.
#
# Outputs:
#   build/micropython.asm    — NASM-syntax assembly (1.5 MB+)
#   build/micropython.bin    — flat 32-bit DOS bin (170 KB)
#
# Run:
#   .venv/bin/python -c "from pathlib import Path; \
#       from uc386.dos_emu import run; \
#       r = run(Path('addons/gnu/micropython/build/micropython.bin'), \
#               timeout_seconds=10.0); \
#       print(r.stdout)"
#
# The REPL banner ("MicroPython uc386-triage on ...; uc386-dos with i386")
# should appear, followed by `>>> ` and the boot waits on stdin.
set -eu

cd "$(dirname "$0")"

if [ ! -d upstream ]; then
    echo "micropython: run ./fetch.sh first." >&2
    exit 1
fi

# build.sh emits the genhdr/* stubs (qstrdefs.generated.h,
# moduledefs.h, mpversion.h, root_pointers.h) that py/*.c expects.
# Run it first if those aren't here yet.
if [ ! -f build/genhdr/qstrdefs.generated.h ]; then
    echo "micropython: stub headers missing; running ./build.sh first." >&2
    ./build.sh > /dev/null
fi

REPO="$(cd ../../.. && pwd)"
if [ -n "${PYTHON:-}" ]; then
    :
elif [ -x "$REPO/.venv/bin/python" ]; then
    PYTHON="$REPO/.venv/bin/python"
else
    PYTHON="$(command -v python3.12 || command -v python3 || command -v python)"
fi
INCLUDE="$REPO/lib/include"

SOURCES_FILE="build/_port_sources.txt"
{
    find upstream/py -name '*.c' | sort
    echo upstream/shared/libc/__errno.c
    echo upstream/shared/libc/abort_.c
    echo upstream/shared/libc/printf.c
    echo upstream/shared/libc/string0.c
    echo upstream/shared/readline/readline.c
    echo upstream/shared/runtime/pyexec.c
    echo upstream/shared/runtime/stdout_helpers.c
    echo upstream/shared/runtime/interrupt_char.c
    echo upstream/shared/runtime/sys_stdio_mphal.c
    echo upstream/shared/timeutils/timeutils.c
    echo upstream/extmod/modtime.c
    echo upstream/ports/minimal/main.c
    echo upstream/ports/minimal/uart_core.c
    echo uc386-dos/mphal_uc386dos.c
    echo uc386-dos/file_uc386dos.c
} > "$SOURCES_FILE"

n_sources="$(wc -l < "$SOURCES_FILE")"
echo "micropython: compiling $n_sources sources via uc386 (multi-TU; ~14 min) …"

# tr+xargs delivery so the source list survives without arg-list overflow.
# `set -e` would abort on uc386 failure; we want to surface it explicitly.
set +e
tr '\n' '\0' < "$SOURCES_FILE" \
    | xargs -0 "$PYTHON" -m uc386.main \
        -I "$INCLUDE" \
        -I upstream \
        -I uc386-dos \
        -I build \
        -D__linux__=1 \
        -o build/micropython.asm
rc=$?
set -e
if [ $rc -ne 0 ]; then
    echo "micropython: uc386 returned $rc; not assembling." >&2
    exit $rc
fi

asm_size="$(wc -c < build/micropython.asm | tr -d ' ')"
echo "micropython: wrote build/micropython.asm ($asm_size bytes)"
echo "micropython: assembling via nasm …"
# `-w-error=label-redef-late`: same convergence warning we suppress for
# DOOM. Long programs hit short/long jump promotion edge cases NASM 3.x
# now warns about; the binary's correct after convergence.
nasm -f bin -w-error=label-redef-late \
    build/micropython.asm -o build/micropython.bin
bin_size="$(wc -c < build/micropython.bin | tr -d ' ')"
echo "micropython: built build/micropython.bin ($bin_size bytes)"
ls -l build/micropython.bin
