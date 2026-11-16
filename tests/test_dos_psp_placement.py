"""dos_emu places the PSP + environment block where the image is not
(uc386 E3). It used to write them at the fixed 0xF0000 after loading the
image, silently overwriting any image that reached that address."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

_PROBE = """BITS 32
ORG 0
        mov     ah, 0x62
        int     0x21                ; BX = PSP segment
        movzx   esi, bx
        shl     esi, 4              ; PSP linear address
        movzx   edi, word [esi + 0x2C]
        shl     edi, 4              ; environment linear address
        cmp     bx, 0xF000
        jbe     .bad                ; must have moved past the image
        cmp     byte [edi], 'K'
        jne     .bad                ; the environment is intact
        mov     al, 7
        jmp     .out
.bad:
        mov     al, 1
.out:
        mov     ah, 0x4C
        int     0x21
        times   {size} - ($ - $$) db 0x90
"""


def _build(tmp_path: Path, size: int) -> bytes:
    if shutil.which("nasm") is None:
        pytest.skip("nasm not installed")
    src = tmp_path / "probe.asm"
    out = tmp_path / "probe.bin"
    src.write_text(_PROBE.format(size=size))
    subprocess.run(["nasm", "-f", "bin", "-o", str(out), str(src)],
                   check=True)
    return out.read_bytes()


def test_psp_moves_after_a_large_image(tmp_path):
    pytest.importorskip("unicorn")
    from uc386.dos_emu import run
    binary = _build(tmp_path, 0xF8000)
    res = run(binary, env={"K": "V"}, timeout_seconds=5.0)
    assert res.error is None, res.error
    assert res.exit_code == 7


def test_image_without_room_for_the_psp_is_refused(tmp_path):
    pytest.importorskip("unicorn")
    from uc386.dos_emu import run
    binary = _build(tmp_path, 0xFFF00)
    res = run(binary, env={"K": "V"}, timeout_seconds=5.0)
    assert res.error is not None and "PSP" in res.error
