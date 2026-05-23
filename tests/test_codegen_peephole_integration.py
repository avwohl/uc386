"""Codegen↔peephole integration tests.

These end-to-end tests verify that uc386's CodeGenerator + the
upeep386 peephole optimizer compose as expected on real source
inputs. The pure-peephole pattern tests live in upeep386's own
test suite — what lives here is uc386-specific: when codegen
emits pattern P, peephole should fire optimization Q.

Extracted from uc386/tests/test_peephole.py when peephole.py was
moved to the upeep386 package (2026-05-23).
"""

from uc386.codegen import CodeGenerator
from uc_core.frontend import parse


def test_codegen_runs_peephole_by_default():
    src = "int main(void) { return 0; }"
    unit = parse(src, "test.c")

    gen = CodeGenerator()
    asm = gen.generate(unit)

    assert isinstance(gen.peephole_stats, dict)
    # The codegen-emitted `mov eax, 0` for the implicit return is
    # rewritten to `xor eax, eax` (3 bytes saved). The dead one
    # after `jmp .epilogue` (if any) is dropped.
    assert gen.peephole_stats.get("mov_zero_to_xor", 0) >= 1
    assert "        mov     eax, 0" not in asm


def test_codegen_skips_peephole_when_disabled():
    src = "int main(void) { return 0; }"
    unit = parse(src, "test.c")

    gen = CodeGenerator(peephole=False)
    asm = gen.generate(unit)

    assert "        xor     eax, eax\n.epilogue:" in asm
    assert gen.peephole_stats == {}


def test_dead_cleanup_before_leave_codegen_integration():
    src = (
        "int g(int x) { return x * 2; }\n"
        "int f(int x) { return g(x); }\n"
        "int main(void) { return f(5); }\n"
    )
    tu = parse(src, "test.c")
    cg = CodeGenerator(peephole=True)
    asm = cg.generate(tu)
    fstart = asm.index("_f:")
    fend = asm.index("_main:")
    f_block = asm[fstart:fend]
    assert "pop     ecx" not in f_block


def test_cmp_load_promote_codegen_integration():
    src = (
        "int sum_arr(int *arr, int n) {\n"
        "    int s = 0;\n"
        "    for (int i = 0; i < n; i++) {\n"
        "        s += arr[i];\n"
        "    }\n"
        "    return s;\n"
        "}\n"
        "int main(void) { return 0; }\n"
    )
    tu = parse(src, "test.c")
    cg = CodeGenerator(peephole=True)
    asm = cg.generate(tu)
    assert "mov     ecx, [ebp - 8]" in asm
    assert "cmp     ecx, [ebp + 12]" in asm
    assert asm.count("mov     ecx, [ebp - 8]") == 1


def test_indirect_call_collapse_codegen_integration():
    src = (
        "int dispatch(int (*fp)(int, int), int x, int y) {\n"
        "    return fp(x, y);\n"
        "}\n"
        "int main(void) { return 0; }\n"
    )
    tu = parse(src, "test.c")
    cg = CodeGenerator(peephole=True)
    asm = cg.generate(tu)
    assert "call    dword [ebp + 8]" in asm
    assert "mov     eax, [ebp + 8]\n        call    eax" not in asm
