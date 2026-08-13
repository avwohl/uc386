"""Regression: console output is buffered, and nothing is lost.

Every character of console output used to be its own INT 21h AH=0x02, so
printing 2,000 bytes cost 2,000 DOS calls. On real hardware each of those
is a full interrupt dispatch plus a Ctrl-C poll, which dominates any
output-heavy program.

Output is now line-buffered through `__stdio_putc_con`, flushed on '\\n',
when the 1024-byte buffer fills, by fflush/fclose, and at exit. The
exit-time flush is emitted by codegen (`_start_stub`) only for programs
that actually reference stdio, so a program that never prints keeps its
minimal binary.

The correctness tail is the point of these tests: buffering is only
acceptable if *nothing* is silently dropped.
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


def test_output_without_trailing_newline_survives_exit(run_prog):
    """THE correctness tail. Line-buffered output means a final line with
    no '\\n' is still in the buffer when main returns; without the
    exit-time flush it would be silently discarded."""
    src = dedent(r"""
        #include <stdio.h>
        int main(void) {
            printf("first line\n");
            printf("tail with no newline");
            return 0;
        }
    """)
    res = run_prog(src)
    assert res.error is None
    assert res.stdout == "first line\ntail with no newline"


def test_output_larger_than_the_buffer_is_complete(run_prog):
    """Crossing the 1024-byte boundary must not drop or duplicate."""
    src = dedent(r"""
        #include <stdio.h>
        int main(void) {
            for (int i = 0; i < 3000; i++) putchar('x');
            return 0;
        }
    """)
    res = run_prog(src)
    assert res.error is None
    assert res.stdout == "x" * 3000


def test_fflush_forces_output_out(run_prog):
    """fflush used to return 0 with nothing to flush; it must now
    actually drain the buffer."""
    src = dedent(r"""
        #include <stdio.h>
        int main(void) {
            printf("before");
            fflush(stdout);
            printf("after");
            return 0;
        }
    """)
    res = run_prog(src)
    assert res.error is None
    assert res.stdout == "beforeafter"


def test_exit_flushes(run_prog):
    """exit() bypasses the codegen return-from-main path, so it carries
    its own flush."""
    src = dedent(r"""
        #include <stdio.h>
        #include <stdlib.h>
        int main(void) {
            printf("pending, no newline");
            exit(3);
        }
    """)
    res = run_prog(src)
    assert res.error is None
    assert res.stdout == "pending, no newline"
    assert res.exit_code == 3


def test_handle_writes_interleave_correctly_with_console_output(run_prog):
    """printf goes through the console buffer; fputs/fwrite go straight
    to a handle. Without a flush in the handle writers the buffered text
    would surface *after* them."""
    src = dedent(r"""
        #include <stdio.h>
        int main(void) {
            printf("one ");
            fputs("two ", stdout);
            printf("three ");
            fwrite("four", 1, 4, stdout);
            return 0;
        }
    """)
    res = run_prog(src)
    assert res.error is None
    assert res.stdout == "one two three four"


def test_unbuffered_mode_still_produces_identical_output(run_prog):
    """setvbuf(_IONBF) restores byte-at-a-time output; only the DOS-call
    count should differ, never the bytes."""
    src = dedent(r"""
        #include <stdio.h>
        int main(void) {
            setvbuf(stdout, 0, _IONBF, 0);
            printf("a%db\n", 42);
            printf("no newline tail");
            return 0;
        }
    """)
    res = run_prog(src)
    assert res.error is None
    assert res.stdout == "a42b\nno newline tail"


def test_program_that_never_prints_gets_no_flush_and_stays_minimal(tmp_path):
    """The flush is only worth emitting for programs that print. An
    unconditional call would drag the buffering code into every binary,
    and `true.bin` is 18 bytes precisely because nothing unused links in.
    """
    asm = _compile_to_asm("int main(void) { return 0; }\n", tmp_path)
    text = asm.read_text()
    assert "__stdio_flush_con" not in text
    assert "call    _main" in text


def test_printing_program_does_get_the_exit_flush(tmp_path):
    asm = _compile_to_asm(
        '#include <stdio.h>\nint main(void){ printf("x"); return 0; }\n',
        tmp_path,
    )
    text = asm.read_text()
    assert "call    __stdio_flush_con" in text
    # ...and it must not clobber main's return value on the way to 4Ch.
    idx = text.index("call    __stdio_flush_con")
    window = text[idx - 60:idx + 60]
    assert "push    eax" in window and "pop     eax" in window
