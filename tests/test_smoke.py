"""Smoke tests: verify the uc_core -> uc386 pipeline is wired up."""

from uc_core.backend import CodeGenerator as CodeGeneratorProtocol
from uc386.codegen import CodeGenerator


def test_backend_implements_protocol():
    assert isinstance(CodeGenerator(), CodeGeneratorProtocol)


def test_end_to_end_stub(tmp_path):
    import subprocess, sys
    src = tmp_path / "hi.c"
    src.write_text("int main(void) { return 0; }\n")
    out = tmp_path / "hi.asm"
    r = subprocess.run(
        [sys.executable, "-m", "uc386.main", str(src), "-o", str(out)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    text = out.read_text()
    assert "STUB" in text
    assert "main" in text
