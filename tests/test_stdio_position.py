"""Regression: stdio position + stream-state must be real, not stubbed.

`fseek` / `ftell` / `rewind` / `clearerr` / `feof` / `ferror` used to be
no-op stubs (`i386_dos_libc.asm`, "no-op stubs" banner). They linked
cleanly and then returned plausible-but-wrong answers, which is worse
than failing:

  - `feof()` was hardwired to 0, so `while (!feof(f))` never terminated.
  - `fseek()` returned success without seeking.
  - `ftell()` always returned 0, silently corrupting any
    save-position/restore-position logic.

FILE* is the raw DOS handle, so seeking is just INT 21h AH=0x42 via the
real `_lseek`; what was actually missing was per-stream EOF/error state,
now held in the `_stdio_flags` table (one byte per handle).

These tests pin the behaviour end to end under dos_emu rather than
asserting on emitted asm, so they stay honest if the implementation
moves.
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


def test_read_to_eof_terminates_and_sets_feof(run_prog):
    """The headline bug: with feof() stubbed to 0 this loop never ended.

    Counts bytes with an explicit EOF-sentinel loop first (which always
    worked), then asserts feof() agrees and ferror() does not fire — the
    two must be distinguishable.
    """
    src = dedent(r"""
        #include <stdio.h>
        int main(void) {
            FILE *f = fopen("data.txt", "r");
            if (!f) { printf("OPENFAIL\n"); return 2; }
            int n = 0;
            while (fgetc(f) != EOF) n++;
            printf("%d %d %d\n", n, feof(f) ? 1 : 0, ferror(f) ? 1 : 0);
            fclose(f);
            return 0;
        }
    """)
    res = run_prog(src, vfiles_init={b"data.txt": b"ABCDEFGHIJ"})
    assert res.error is None
    assert res.timed_out is False
    assert res.stdout == "10 1 0\n"


def test_feof_driven_loop_terminates(run_prog):
    """`while (!feof(f))` is the idiom that used to hang outright."""
    src = dedent(r"""
        #include <stdio.h>
        int main(void) {
            FILE *f = fopen("data.txt", "r");
            int n = 0, c;
            while (!feof(f)) {
                c = fgetc(f);
                if (c == EOF) break;
                n++;
            }
            printf("%d\n", n);
            fclose(f);
            return 0;
        }
    """)
    res = run_prog(src, vfiles_init={b"data.txt": b"12345"})
    assert res.error is None
    assert res.timed_out is False
    assert res.stdout == "5\n"


def test_ftell_tracks_position_and_fseek_moves_it(run_prog):
    """ftell() reported 0 forever; fseek() didn't move the position.

    Each ftell() is printed on its own statement so the result can't
    depend on argument evaluation order.
    """
    src = dedent(r"""
        #include <stdio.h>
        int main(void) {
            FILE *f = fopen("data.txt", "r");
            printf("start=%ld\n", ftell(f));
            fgetc(f); fgetc(f); fgetc(f);
            printf("after3=%ld\n", ftell(f));
            fseek(f, 4, SEEK_SET);
            printf("seek4=%ld\n", ftell(f));
            printf("byte=%c\n", fgetc(f));
            fseek(f, 0, SEEK_END);
            printf("size=%ld\n", ftell(f));
            fclose(f);
            return 0;
        }
    """)
    res = run_prog(src, vfiles_init={b"data.txt": b"ABCDEFGHIJ"})
    assert res.error is None
    assert res.stdout == "start=0\nafter3=3\nseek4=4\nbyte=E\nsize=10\n"


def test_rewind_clears_eof_and_rereads(run_prog):
    """rewind() must reset the position AND clear EOF (C11 7.21.9.5)."""
    src = dedent(r"""
        #include <stdio.h>
        int main(void) {
            FILE *f = fopen("data.txt", "r");
            while (fgetc(f) != EOF) { }
            printf("eof_before=%d\n", feof(f) ? 1 : 0);
            rewind(f);
            printf("eof_after=%d pos=%ld\n", feof(f) ? 1 : 0, ftell(f));
            printf("first=%c\n", fgetc(f));
            fclose(f);
            return 0;
        }
    """)
    res = run_prog(src, vfiles_init={b"data.txt": b"XYZ"})
    assert res.error is None
    assert res.stdout == "eof_before=1\neof_after=0 pos=0\nfirst=X\n"


def test_fseek_clears_eof_but_clearerr_is_what_clears_error(run_prog):
    """fseek() clears EOF (C11 7.21.9.2p5); clearerr() clears both."""
    src = dedent(r"""
        #include <stdio.h>
        int main(void) {
            FILE *f = fopen("data.txt", "r");
            while (fgetc(f) != EOF) { }
            printf("eof=%d\n", feof(f) ? 1 : 0);
            fseek(f, 0, SEEK_SET);
            printf("after_seek_eof=%d\n", feof(f) ? 1 : 0);
            clearerr(f);
            printf("after_clearerr=%d %d\n",
                   feof(f) ? 1 : 0, ferror(f) ? 1 : 0);
            fclose(f);
            return 0;
        }
    """)
    res = run_prog(src, vfiles_init={b"data.txt": b"Q"})
    assert res.error is None
    assert res.stdout == "eof=1\nafter_seek_eof=0\nafter_clearerr=0 0\n"


def test_eof_is_per_stream_not_global(run_prog):
    """`_stdio_flags` is indexed by handle, so hitting EOF on one open
    file must not make a second, independent stream report EOF."""
    src = dedent(r"""
        #include <stdio.h>
        int main(void) {
            FILE *a = fopen("a.txt", "r");
            FILE *b = fopen("b.txt", "r");
            while (fgetc(a) != EOF) { }
            printf("a_eof=%d b_eof=%d\n",
                   feof(a) ? 1 : 0, feof(b) ? 1 : 0);
            printf("b_first=%c\n", fgetc(b));
            fclose(a); fclose(b);
            return 0;
        }
    """)
    res = run_prog(src, vfiles_init={b"a.txt": b"aa", b"b.txt": b"bb"})
    assert res.error is None
    assert res.stdout == "a_eof=1 b_eof=0\nb_first=b\n"
