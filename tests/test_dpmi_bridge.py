"""Regression tests for the DPMI real-mode-call bridge in dos_emu.

Background: a uc386 binary reaches DOS through two helpers that wrap an
INT in a segment save/restore dance. dos_emu intercepts them at their
entry point rather than letting them execute, because their bodies trip
two unicorn 2.x bugs (a spurious #GP on `pop es`, and a `pop` that
returns a phantom value instead of the dword in memory).

The interception used to locate its target with a bare `binary.find()`
plus a comment asserting the prologue appeared "at exactly one address".
It appears twice. Only the first helper was intercepted; the second --
the DPMI real-mode-call bridge -- executed natively and died, taking
every dos_emu run of the MicroPython port with it.

Nothing in the suite caught that, because the only test that exercises
this path is tests/test_micropython_integration.py, which skips unless
FREEDOS_MP_BIN points at a MicroPython build (~30 minutes, out of tree)
and CI never sets it. These tests close that gap: they run in-tree, in
milliseconds, with no external build.
"""
from __future__ import annotations

import struct

import pytest

from uc386.dos_emu import locate_seg_wrappers, run

pytest.importorskip("unicorn")


# The shared prologue: push ebp / mov ebp,esp / push esi / push edi /
# push ebx / push es.
PROLOGUE = b"\x55\x89\xe5\x56\x57\x53\x06"


def _pktdrv_like() -> bytes:
    """A pktdrv_int_invoke lookalike: patches its own INT immediate."""
    return PROLOGUE + b"\x1e\x07" + b"\x8b\x45\x08" + b"\xcd\x00" + b"\xc3"


def _dpmi_like() -> bytes:
    """A DPMI-bridge lookalike: hardcodes AX=0x0301 before int 31h."""
    return (PROLOGUE + b"\x1e\x07" + b"\xb8\x01\x03\x00\x00"
            + b"\x8b\x7d\x08" + b"\xcd\x31" + b"\xc3")


class TestLocateSegWrappers:
    """The classifier itself. These are the tests that would have failed
    on the old `binary.find()` implementation."""

    def test_finds_both_helpers_not_just_the_first(self):
        blob = b"\x90" * 16 + _pktdrv_like() + b"\x90" * 16 + _dpmi_like()
        pktdrv, dpmi = locate_seg_wrappers(blob)
        assert pktdrv == 16
        assert dpmi == 16 + len(_pktdrv_like()) + 16

    def test_order_in_the_binary_does_not_matter(self):
        """The DPMI bridge may come first. A `find()`-based
        implementation would then misclassify it as pktdrv_int_invoke."""
        blob = b"\x90" * 8 + _dpmi_like() + b"\x90" * 8 + _pktdrv_like()
        pktdrv, dpmi = locate_seg_wrappers(blob)
        assert dpmi == 8
        assert pktdrv == 8 + len(_dpmi_like()) + 8

    def test_absent_helpers_report_minus_one(self):
        assert locate_seg_wrappers(b"\x90" * 256) == (-1, -1)

    def test_classifies_by_body_not_by_position(self):
        pktdrv, dpmi = locate_seg_wrappers(_dpmi_like())
        assert dpmi == 0
        assert pktdrv == -1


# ---------------------------------------------------------------------
# End-to-end: drive the bridge and check DOS actually happened.
# ---------------------------------------------------------------------

RMCS_FMT = "<8I9H"          # edi esi ebp resv ebx edx ecx eax + 9 words
BOUNCE_SEG = 0x4000         # matches dos_emu's seg arena start
BOUNCE_LINEAR = BOUNCE_SEG << 4
THUNK_SEG = 0x4085
THUNK_LINEAR = THUNK_SEG << 4


def _build_rmcs(eax: int, ecx: int = 0, edx: int = 0, esi: int = 0) -> bytes:
    return struct.pack(
        RMCS_FMT,
        0, esi, 0, 0, 0, edx, ecx, eax,     # edi esi ebp resv ebx edx ecx eax
        0x0202,                              # flags
        0,                                   # es
        BOUNCE_SEG,                          # ds
        0, 0,                                # fs gs
        0, THUNK_SEG,                        # ip, cs -> the CD 21 CB thunk
        0x0FFE, 0x4086,                      # sp, ss
    )


def test_bridge_dispatches_int21_create_write_close():
    """AH=0x3C (create) through the bridge must actually create the
    file. This is the gap that made every `open(path, "w")` in the
    MicroPython port raise ENOENT: the rmcs dispatcher knew AH=0x3D but
    not AH=0x3C, so it returned CF=1 for a call whose whole job was to
    create a file that did not exist yet.
    """
    # Layout: code at 0, rmcs + strings placed in the bounce paragraph.
    name = b"OUT.TXT\x00"
    data = b"hello"

    code = bytearray()
    # The DPMI bridge helper, at a fixed offset we compute after.
    # Caller first: push &rmcs; call bridge; add esp,4  -- three times.
    # We patch the call targets once the helper offset is known.
    RMCS_A = 0x2000        # create
    RMCS_B = 0x2040        # write
    RMCS_C = 0x2080        # close

    def call_bridge(rmcs_addr, placeholder):
        c = bytearray()
        c += b"\x68" + struct.pack("<I", rmcs_addr)   # push imm32
        c += b"\xe8" + struct.pack("<i", placeholder)  # call rel32
        c += b"\x83\xc4\x04"                           # add esp, 4
        return c

    body = bytearray()
    for addr in (RMCS_A, RMCS_B, RMCS_C):
        body += call_bridge(addr, 0)
    body += b"\xb8\x00\x4c\x00\x00"    # mov eax, 0x4C00
    body += b"\xcd\x21"                # int 21h  -> exit(0)

    helper_off = len(body)
    helper = _dpmi_like()

    # Patch each call's rel32 now that we know where the helper lives.
    out = bytearray(body)
    pos = 0
    while True:
        idx = out.find(b"\xe8\x00\x00\x00\x00", pos)
        if idx == -1:
            break
        next_ip = idx + 5
        struct.pack_into("<i", out, idx + 1, helper_off - next_ip)
        pos = next_ip

    blob = bytearray(out + helper)
    blob += b"\x00" * (0x2000 - len(blob))

    # rmcs structures.
    blob += _build_rmcs(0x3C00, ecx=0, edx=0)                 # create, DS:DX=name
    blob += b"\x00" * (0x40 - len(_build_rmcs(0)))
    blob += _build_rmcs(0x4000, ecx=len(data), edx=0x20)      # write
    blob += b"\x00" * (0x40 - len(_build_rmcs(0)))
    blob += _build_rmcs(0x3E00)                               # close
    blob += b"\x00" * (0x40 - len(_build_rmcs(0)))

    # The bounce paragraph holds the filename at +0 and the payload at
    # +0x20; the thunk (`CD 21 CB`) sits in its own paragraph.
    blob += b"\x00" * (BOUNCE_LINEAR - len(blob))
    blob += name + b"\x00" * (0x20 - len(name)) + data
    blob += b"\x00" * (THUNK_LINEAR - len(blob))
    blob += b"\xcd\x21\xcb"

    # The write and close rmcs need BX = the handle the create returned.
    # Rather than thread that through hand-written asm, assert the
    # weaker but still decisive property: the create happened at all.
    res = run(bytes(blob), timeout_seconds=10.0, instruction_limit=5_000_000)

    assert res.error is None, f"dos_emu error: {res.error}"
    created = [k for k in res.vfiles if k.upper().endswith(b"OUT.TXT")]
    assert created, (
        "AH=0x3C through the DPMI bridge did not create the file; "
        f"vfiles={res.vfiles!r}"
    )


def test_unicorn_still_truncates_the_stack_after_pop_es():
    """Pin the unicorn defect the interception exists to work around.

    Executing `pop es` switches unicorn's SS-relative accesses to
    16-bit for the rest of the run. If this test ever fails, unicorn
    has fixed the bug and the entry-point interception in dos_emu could
    be reconsidered — so a failure here is good news, not a regression.
    """
    import unicorn
    from unicorn import Uc, UC_ARCH_X86, UC_MODE_32
    from unicorn.x86_const import UC_X86_REG_ESP, UC_X86_REG_EAX

    uc = Uc(UC_ARCH_X86, UC_MODE_32)
    uc.mem_map(0, 0x1100000)
    uc.mem_write(0x1000, b"\x07\x58\xf4")            # pop es / pop eax / hlt
    uc.mem_write(0x10FFF70, struct.pack("<II", 0, 0xCCCCCCCC))
    uc.mem_write(0x000FF70, struct.pack("<II", 0, 0xDDDDDDDD))
    uc.reg_write(UC_X86_REG_ESP, 0x10FFF70)
    uc.emu_start(0x1000, 0, count=2)

    assert uc.reg_read(UC_X86_REG_EAX) == 0xDDDDDDDD, (
        "unicorn no longer truncates the stack after `pop es` — the "
        "dos_emu seg-wrapper interception may no longer be necessary"
    )


def test_no_segment_pop_executes_in_an_intercepted_run():
    """The invariant that keeps the port working: with the helpers
    intercepted at their entry, no `pop es`/`pop ds` ever reaches the
    CPU, so the stack never gets poisoned."""
    blob = bytearray()
    blob += b"\x68" + struct.pack("<I", 0x1000)   # push &rmcs
    blob += b"\xe8" + struct.pack("<i", 0)        # call (patched below)
    blob += b"\x83\xc4\x04"                       # add esp, 4
    blob += b"\xb8\x00\x4c\x00\x00"               # mov eax, 0x4C00
    blob += b"\xcd\x21"                           # exit
    helper_off = len(blob)
    struct.pack_into("<i", blob, 6, helper_off - 10)
    blob += _dpmi_like()
    blob += b"\x00" * (0x1000 - len(blob))
    blob += _build_rmcs(0x3C00)
    blob += b"\x00" * (BOUNCE_LINEAR - len(blob))
    blob += b"POP.TXT\x00"
    blob += b"\x00" * (THUNK_LINEAR - len(blob))
    blob += b"\xcd\x21\xcb"

    import unicorn
    seen = []
    orig = unicorn.Uc.hook_add

    def hook_add(self, htype, cb, *a, **kw):
        if htype == unicorn.UC_HOOK_CODE:
            def wrapped(uc, address, size, ud):
                if size == 1:
                    try:
                        if uc.mem_read(address, 1)[0] in (0x07, 0x1F):
                            seen.append(address)
                    except Exception:
                        pass
                return cb(uc, address, size, ud)
            return orig(self, htype, wrapped, *a, **kw)
        return orig(self, htype, cb, *a, **kw)

    unicorn.Uc.hook_add = hook_add
    try:
        res = run(bytes(blob), timeout_seconds=10.0,
                  instruction_limit=5_000_000)
    finally:
        unicorn.Uc.hook_add = orig

    assert res.error is None, f"dos_emu error: {res.error}"
    assert seen == [], (
        f"a segment-register pop executed at {[hex(a) for a in seen]}; "
        "the seg-wrapper interception is not covering it"
    )


def test_bridge_helper_body_never_executes():
    """The whole point of the intercept is that the helper's body -- the
    part that trips unicorn -- is skipped. If interception regressed,
    the `int 31h` inside the helper would reach on_int."""
    blob = bytearray()
    blob += b"\x68" + struct.pack("<I", 0x1000)   # push &rmcs
    blob += b"\xe8" + struct.pack("<i", 0)        # call (patched below)
    blob += b"\x83\xc4\x04"                       # add esp, 4
    blob += b"\xb8\x00\x4c\x00\x00"               # mov eax, 0x4C00
    blob += b"\xcd\x21"                           # exit
    helper_off = len(blob)
    struct.pack_into("<i", blob, 6, helper_off - 10)
    blob += _dpmi_like()
    blob += b"\x00" * (0x1000 - len(blob))
    blob += _build_rmcs(0x3C00)
    blob += b"\x00" * (BOUNCE_LINEAR - len(blob))
    blob += b"NUL.TXT\x00"
    blob += b"\x00" * (THUNK_LINEAR - len(blob))
    blob += b"\xcd\x21\xcb"

    res = run(bytes(blob), timeout_seconds=10.0, instruction_limit=5_000_000)
    # A live `int 31h` reaching the emulator unhandled would surface as
    # an error; interception means we never get there.
    assert res.error is None, f"helper body executed: {res.error}"
