"""Regression: a `vfiles_init` seed must be openable by the name a
program actually passes to `fopen`.

`_dos_path` canonicalizes every INT 21h path to an absolute,
C:-drive form ("data.txt" -> "C:\\data.txt"). Commit 3e30b18 added
that for lookups but kept the *raw* manifest key when seeding the
vfs, so `fopen("data.txt")` of a seeded relative-named file always
missed (-> NULL). That silently broke cat / sbase-cat (the only
manifest addons that fopen a relative vfile). `_canon_vfs_path` is
now the single source of truth for both seeding and lookup; these
tests lock the seed==lookup invariant at the behavior level.
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


_CAT_SRC = dedent(r"""
    #include <stdio.h>
    int main(int argc, char **argv) {
        FILE *fp = fopen(argv[1], "r");
        if (!fp) { fputs("OPENFAIL", stdout); return 2; }
        int c;
        while ((c = fgetc(fp)) != EOF) putchar(c);
        fclose(fp);
        return 0;
    }
""")


def test_seeded_relative_file_is_openable(run_prog):
    """The exact cat/sbase-cat failure mode: open a seeded file by
    its relative name."""
    res = run_prog(
        _CAT_SRC,
        argv=["prog", "data.txt"],
        vfiles_init={b"data.txt": b"hello vfs\n"},
    )
    assert res.error is None
    assert res.exit_code == 0
    assert res.stdout == "hello vfs\n"


def test_seeded_absolute_file_still_openable(run_prog):
    """An absolute-named seed (how git-style code names files) must
    keep working through the same canonicalization."""
    res = run_prog(
        _CAT_SRC,
        argv=["prog", "C:\\sub\\f.txt"],
        vfiles_init={b"C:\\sub\\f.txt": b"abs ok\n"},
    )
    assert res.error is None
    assert res.exit_code == 0
    assert res.stdout == "abs ok\n"
