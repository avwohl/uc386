"""Regression: the width flags must not silently miscompile.

`--int` / `--long` / `--long-long` / `--ptr` feed the *frontend* — they
reach ASTOptimizer, which const-folds `sizeof`. Codegen sizes types from
its own `CodeGenerator._BASIC_SIZES` and never sees that config. So a
non-default width produced a compiler whose `sizeof()` contradicted its
own storage layout, with no diagnostic:

    --long 32 -> sizeof(long)=4   adjacent long locals 4 bytes apart
    --long 64 -> sizeof(long)=8   adjacent long locals 4 bytes apart

Any `sizeof`-driven memcpy, malloc or array stride overran by 2x. The
driver now refuses the combination instead.

The guard compares the requested TypeConfig against codegen's real size
table rather than a hardcoded blocklist, so if codegen ever gains a
width the flag starts working with no change here — these tests are
written to keep passing in that case too.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from uc386.codegen import CodeGenerator

REPO_ROOT = Path(__file__).resolve().parent.parent

_SRC = "int main(void) { long a = 1, b = 2; return (int)(a + b); }\n"


def _compile(tmp_path: Path, *flags: str):
    src = tmp_path / "w.c"
    src.write_text(_SRC)
    return subprocess.run(
        [sys.executable, "-m", "uc386.main", str(src),
         "-o", str(tmp_path / "w.asm"), *flags],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )


def test_default_widths_compile(tmp_path: Path):
    assert _compile(tmp_path).returncode == 0


def test_explicit_flat32_widths_compile(tmp_path: Path):
    """Passing the flags with the values codegen actually implements is
    fine — the guard keys off agreement, not off the flag being present."""
    res = _compile(tmp_path, "--int", "32", "--long", "32",
                   "--long-long", "64", "--ptr", "32")
    assert res.returncode == 0, res.stderr


@pytest.mark.parametrize(
    "flag,value,type_name",
    [("--long", "64", "long"), ("--int", "16", "int")],
)
def test_unsupported_width_is_refused(tmp_path: Path, flag, value, type_name):
    """The point of the guard: refuse rather than miscompile.

    Skips automatically if codegen ever implements the width, since then
    the flag is legitimately supported and refusing would be wrong.
    """
    if CodeGenerator._BASIC_SIZES[type_name] == int(value) // 8:
        pytest.skip(f"codegen now implements {type_name} at {value} bits")
    res = _compile(tmp_path, flag, value)
    assert res.returncode == 1
    assert "not supported by the x86-32 code generator" in res.stderr
    assert flag in res.stderr
    # The diagnostic must say what disagrees, not just that something did.
    assert f"sizeof({type_name})" in res.stderr


def test_refusal_names_the_supported_model(tmp_path: Path):
    """A refusal has to tell you what to do instead."""
    if CodeGenerator._BASIC_SIZES["long"] == 8:
        pytest.skip("codegen now implements 64-bit long")
    res = _compile(tmp_path, "--long", "64")
    assert "Watcom flat-32" in res.stderr
