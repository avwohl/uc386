"""Regression: DOS path canonicalization and handle duplication in dos_emu.

E1 -- DOS canonicalizes every path it is given (AH=60h TRUENAME is the same
operation): a `.` component is dropped, a `..` component removes the one
before it, never above the root. chdir resolves its argument against the
current directory and needs an existing directory. `run(cwd=...)` sets the
start directory.

E2 -- INT 21h AH=45h (DUP) and AH=46h (FORCEDUP). A forced duplicate of a
file onto handle 1 redirects standard output, and the console functions
AH=02h / 09h write through handle 1, so they are redirected too.
"""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from uc386.codegen import CodeGenerator
from uc386.dos_emu import _canon_vfs_path
from uc_core.frontend import parse
from uc_core.preprocessor import Preprocessor

REPO_ROOT = Path(__file__).resolve().parent.parent
INCLUDE = REPO_ROOT / "src" / "uc386" / "lib" / "include"


def test_canon_folds_dot_components():
    assert _canon_vfs_path(b"..\\F.TXT", b"C:\\SUB") == b"C:\\F.TXT"
    assert _canon_vfs_path(b"..", b"C:\\") == b"C:\\"
    assert _canon_vfs_path(b"C:\\a/./b/../c", b"C:\\") == b"C:\\a/c"
    assert _canon_vfs_path(b"x/..", b"C:\\G") == b"C:\\G"
    # A dotted NAME is not a dot component; separators stay as written.
    assert _canon_vfs_path(b".git/config", b"C:\\R") == b"C:\\R\\.git/config"


def _compile_to_asm(src: str, tmp_path: Path) -> Path:
    src_path = tmp_path / "prog.c"
    src_path.write_text(src)
    asm_path = tmp_path / "prog.asm"
    pp = Preprocessor(include_paths=[str(INCLUDE)])
    unit = parse(pp.preprocess(src, str(src_path)), str(src_path))
    asm_path.write_text(CodeGenerator().generate(unit))
    return asm_path


_CHDIR_SRC = dedent(r"""
    #include <stdio.h>
    #include <string.h>
    int mkdir(const char *path, int mode);
    int chdir(const char *path);
    char *getcwd(char *buf, int size);
    int main(void) {
        char top[80], cur[80];
        getcwd(top, sizeof top);
        if (mkdir("SUB", 0) != 0) return 1;
        if (chdir("SUB") != 0) return 2;
        getcwd(cur, sizeof cur);
        if (strcmp(cur, top) == 0) return 3;
        if (chdir("..") != 0) return 4;
        getcwd(cur, sizeof cur);
        if (strcmp(cur, top) != 0) return 5;
        FILE *f = fopen("SUB\\..\\F.TXT", "w");
        if (!f) return 6;
        fputs("x", f);
        fclose(f);
        if (!fopen("F.TXT", "r")) return 7;
        if (chdir("NOSUCH") == 0) return 8;
        return 0;
    }
""")


def test_chdir_dotdot_and_start_directory(tmp_path: Path):
    pytest.importorskip("unicorn")
    from uc386.dos_emu import assemble_and_run

    asm = _compile_to_asm(_CHDIR_SRC, tmp_path)
    res = assemble_and_run(asm, timeout_seconds=5.0)
    assert res.error is None, res.error
    assert res.exit_code == 0
    # Started in a build directory: the same program, `..` reaches the root.
    res2 = assemble_and_run(asm, timeout_seconds=5.0, cwd="C:\\BUILD")
    assert res2.error is None, res2.error
    assert res2.exit_code == 0


_REDIRECT_ASM = dedent(r"""
    bits 32
    org 0
    start:
        mov     edx, fname
        mov     ah, 0x3C            ; create C:\CAP.TXT
        xor     ecx, ecx
        int     0x21
        movzx   esi, ax             ; esi = file handle
        mov     ebx, 1
        mov     ah, 0x45            ; saved = dup(1)
        int     0x21
        movzx   edi, ax
        mov     ebx, esi
        mov     ecx, 1
        mov     ah, 0x46            ; 1 := duplicate of the file
        int     0x21
        mov     edx, msg_in
        mov     ah, 0x09            ; "into$" goes to the file
        int     0x21
        mov     dl, '!'
        mov     ah, 0x02            ; so does AH=02h
        int     0x21
        mov     ebx, edi
        mov     ecx, 1
        mov     ah, 0x46            ; restore handle 1
        int     0x21
        mov     edx, msg_out
        mov     ah, 0x09            ; "out$" reaches stdout
        int     0x21
        mov     ebx, edi
        mov     ah, 0x3E
        int     0x21
        mov     ebx, esi
        mov     ah, 0x3E
        int     0x21
        mov     ebx, 77             ; closing a bad handle sets CF
        mov     ah, 0x3E
        int     0x21
        mov     al, 0
        jnc     .bad
        mov     ax, 0x4C00
        int     0x21
    .bad:
        mov     ax, 0x4C09
        int     0x21
    fname:   db "C:\CAP.TXT", 0
    msg_in:  db "into$"
    msg_out: db "out$"
""")


def test_forcedup_redirects_standard_output(tmp_path: Path):
    pytest.importorskip("unicorn")
    from uc386.dos_emu import assemble_and_run

    asm = tmp_path / "redir.asm"
    asm.write_text(_REDIRECT_ASM)
    res = assemble_and_run(asm, timeout_seconds=5.0, bundle_libc=False)
    assert res.error is None, res.error
    assert res.exit_code == 0
    assert res.stdout == "out"
    assert bytes(res.vfiles.get(b"C:\\CAP.TXT", b"")) == b"into!"
