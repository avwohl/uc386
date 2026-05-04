"""End-to-end test for the DOS env-block walk: compiles a tiny C
program that calls `getenv` / `dos_argv0` / `dos_env_iter` and runs
it through dos_emu's fake PSP + env block.

These exercise:
- INT 21h AH=0x62 (Get PSP) → dos_emu's PSP_SEG response.
- PSP[0x2C] = env_seg, env block populated from `env=` kwarg.
- libc's `_getenv` walking entries (case-sensitive match + '=' check).
- libc's `_dos_argv0` skipping the env terminator + count word.
- libc's `_dos_env_iter` for indexed environ enumeration.
"""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from uc386.codegen import CodeGenerator
from uc_core.lexer import Lexer
from uc_core.parser import Parser
from uc_core.preprocessor import Preprocessor


REPO_ROOT = Path(__file__).resolve().parent.parent
INCLUDE = REPO_ROOT / "lib" / "include"


def _compile_to_asm(src: str, tmp_path: Path) -> Path:
    """Run preprocess → lex → parse → codegen in-process so the test
    doesn't rely on a `python` executable on PATH (xcode-select on
    macOS shadows it)."""
    src_path = tmp_path / "prog.c"
    src_path.write_text(src)
    asm_path = tmp_path / "prog.asm"
    pp = Preprocessor(include_paths=[str(INCLUDE)])
    pp_text = pp.preprocess(src, str(src_path))
    tokens = list(Lexer(pp_text, str(src_path)).tokenize())
    unit = Parser(tokens).parse()
    asm_path.write_text(CodeGenerator().generate(unit))
    return asm_path


@pytest.fixture
def run_with_env(tmp_path: Path):
    pytest.importorskip("unicorn")
    from uc386.dos_emu import assemble_and_run

    def _run(src: str, **kwargs):
        asm = _compile_to_asm(src, tmp_path)
        return assemble_and_run(asm, timeout_seconds=5.0, **kwargs)

    return _run


def test_getenv_hit_and_miss(run_with_env):
    src = dedent(r"""
        #include <stdio.h>
        extern const char *getenv(const char *);
        int main(void) {
            const char *p = getenv("PATH");
            printf("PATH=%s\n", p ? p : "(null)");
            const char *m = getenv("NOPE");
            printf("NOPE=%s\n", m ? m : "(null)");
            return 0;
        }
    """)
    res = run_with_env(src, env={"PATH": "C:\\DOS", "TERM": "ansi"})
    assert res.error is None
    assert res.exit_code == 0
    assert "PATH=C:\\DOS" in res.stdout
    assert "NOPE=(null)" in res.stdout


def test_getenv_empty_block(run_with_env):
    """No `env=` kwarg → env block is just the double-NUL terminator
    plus the program-path tail. getenv() always returns NULL."""
    src = dedent(r"""
        #include <stdio.h>
        extern const char *getenv(const char *);
        int main(void) {
            const char *p = getenv("ANYTHING");
            printf("p=%s\n", p ? p : "NULL");
            return 0;
        }
    """)
    res = run_with_env(src)
    assert res.error is None
    assert res.exit_code == 0
    assert "p=NULL" in res.stdout


def test_getenv_no_partial_prefix_match(run_with_env):
    """getenv("PA") must NOT match "PATH=...". The '=' check guards
    against entries whose key happens to start with the query."""
    src = dedent(r"""
        #include <stdio.h>
        extern const char *getenv(const char *);
        int main(void) {
            const char *p = getenv("PA");
            printf("p=%s\n", p ? p : "NULL");
            return 0;
        }
    """)
    res = run_with_env(src, env={"PATH": "C:\\DOS"})
    assert res.error is None
    assert "p=NULL" in res.stdout


def test_dos_argv0_returns_program_path(run_with_env):
    src = dedent(r"""
        #include <stdio.h>
        extern const char *dos_argv0(void);
        int main(void) {
            const char *a = dos_argv0();
            printf("a0=%s\n", a ? a : "NULL");
            return 0;
        }
    """)
    res = run_with_env(src,
                       env={"FOO": "bar"},
                       program_path="C:\\BIN\\TEST.EXE")
    assert res.error is None
    assert "a0=C:\\BIN\\TEST.EXE" in res.stdout


def test_dos_env_iter_walks_entries(run_with_env):
    src = dedent(r"""
        #include <stdio.h>
        extern const char *dos_env_iter(unsigned);
        int main(void) {
            for (unsigned i = 0; i < 10; i++) {
                const char *e = dos_env_iter(i);
                if (!e) break;
                printf("[%u] %s\n", i, e);
            }
            return 0;
        }
    """)
    res = run_with_env(src,
                       env={"A": "1", "BB": "two", "CCC": "three"})
    assert res.error is None
    assert res.exit_code == 0
    # Insertion order is preserved (dict ordering since Python 3.7).
    assert "[0] A=1" in res.stdout
    assert "[1] BB=two" in res.stdout
    assert "[2] CCC=three" in res.stdout
    # Past the last entry returns NULL → loop ends, [3] not printed.
    assert "[3]" not in res.stdout
