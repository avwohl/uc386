"""Regression: errno must be populated, and strerror/perror must use it.

`errno` existed but was written in exactly two places in the whole libc
(`_open` and `_creat`, both hardcoding ENOENT), so `strerror` returned a
single fixed string — "error" — for every input, and `perror` printed a
hardcoded ": error" suffix carrying no information beyond "something
failed".

The INT 21h error codes were always there; they were just discarded at
each call site. Now every failure path translates the DOS code through
`__set_errno_dos` (DOS 6 invalid handle -> EBADF, DOS 5 access denied ->
EACCES, DOS 2/3 -> ENOENT, ...), and `strerror` maps errno to a real
message.

Also covers `setvbuf`, which used to claim success for a buffering mode
it never implemented.
"""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from uc386.codegen import CodeGenerator
from uc_core.frontend import parse
from uc_core.preprocessor import Preprocessor

REPO_ROOT = Path(__file__).resolve().parent.parent
INCLUDE = REPO_ROOT / "src" / "uc386" / "lib" / "include"


def _compile_to_asm(src: str, tmp_path: Path) -> Path:
    src_path = tmp_path / "prog.c"
    src_path.write_text(src)
    asm_path = tmp_path / "prog.asm"
    pp = Preprocessor(include_paths=[str(INCLUDE)])
    unit = parse(pp.preprocess(src, str(src_path)), str(src_path))
    asm_path.write_text(CodeGenerator().generate(unit))
    return asm_path


@pytest.fixture
def run_prog(tmp_path: Path):
    pytest.importorskip("unicorn")
    from uc386.dos_emu import assemble_and_run

    def _run(src: str, **kw):
        return assemble_and_run(
            _compile_to_asm(src, tmp_path), timeout_seconds=5.0, **kw
        )

    return _run


def test_strerror_maps_errno_to_distinct_messages(run_prog):
    """Every value used to give the same string."""
    src = dedent(r"""
        #include <stdio.h>
        #include <errno.h>
        #include <string.h>
        int main(void) {
            printf("%s|%s|%s|%s|%s\n",
                   strerror(0), strerror(ENOENT), strerror(EACCES),
                   strerror(EBADF), strerror(999));
            return 0;
        }
    """)
    res = run_prog(src)
    assert res.error is None
    assert res.stdout == (
        "No error|No such file or directory|Permission denied|"
        "Bad file descriptor|Unknown error\n"
    )


def test_fopen_failure_sets_errno(run_prog):
    """`if (!f) perror(path);` is the universal idiom and it reported
    nothing, because fopen never set errno."""
    src = dedent(r"""
        #include <stdio.h>
        #include <errno.h>
        int main(void) {
            errno = 0;
            FILE *f = fopen("nope.txt", "r");
            printf("%s %d\n", f ? "ok" : "NULL", errno);
            return 0;
        }
    """)
    res = run_prog(src)
    assert res.error is None
    assert res.stdout == "NULL 2\n"          # ENOENT


def test_perror_reports_the_actual_error(run_prog):
    src = dedent(r"""
        #include <stdio.h>
        int main(void) {
            FILE *f = fopen("nope.txt", "r");
            if (!f) perror("nope.txt");
            return 0;
        }
    """)
    res = run_prog(src)
    assert res.error is None
    assert res.stderr == "nope.txt: No such file or directory\n"


def test_dos_error_codes_map_to_distinct_errnos(run_prog):
    """DOS 6 (invalid handle) -> EBADF, DOS 5 (access denied) -> EACCES.
    Both used to leave errno untouched entirely."""
    src = dedent(r"""
        #include <stdio.h>
        #include <errno.h>
        int read(int, void *, int);
        int main(void) {
            char buf[8];
            errno = 0;
            int n = read(99, buf, 4);      /* invalid handle */
            printf("%d %d\n", n, errno);
            errno = 0;
            int c = fgetc(stdout);         /* write-only stream */
            printf("%d %d\n", c, errno);
            return 0;
        }
    """)
    res = run_prog(src)
    assert res.error is None
    assert res.stdout == "-1 9\n-1 13\n"    # EBADF, EACCES


def test_setvbuf_honors_every_mode(run_prog):
    """setvbuf originally returned 0 ("honored") while ignoring the
    request entirely. It then briefly refused _IOFBF/_IOLBF honestly,
    because no buffering layer existed. Console output is buffered now,
    so all three modes are real and all three succeed. An invalid mode
    still fails."""
    src = dedent(r"""
        #include <stdio.h>
        int main(void) {
            int a = setvbuf(stdout, 0, _IOFBF, 512);
            int b = setvbuf(stdout, 0, _IOLBF, 512);
            int c = setvbuf(stdout, 0, _IONBF, 0);
            int d = setvbuf(stdout, 0, 99, 0);
            printf("%d %d %d %d\n", a, b, c, d ? 1 : 0);
            return 0;
        }
    """)
    res = run_prog(src)
    assert res.error is None
    assert res.stdout == "0 0 0 1\n"
