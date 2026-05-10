#!/usr/bin/env python3
"""pyle — Python OMF→MZ+LE linker, replacing Open Watcom's `wlink`.

Takes 32-bit OMF .obj files (from `nasm -f obj`) plus a PMODE/W stub
binary and emits an MZ+LE executable that PMODE/W loads on FreeDOS,
DOSBox-X, QEMU+FreeDOS, and dos_emu.

Why this exists: Open Watcom has no native macOS build, so on a Mac
the existing `addons.harness.exe` pipeline can't produce `.exe` from
`.bin`. pyle removes that hard dependency — only `nasm` (cross-platform)
and pyle (pure Python) need to run locally.

Scope (intentionally narrow):
  - 32-bit USE32 segments only — `_TEXT class=CODE`, `_DATA class=DATA`,
    `_BSS class=BSS`. No 16-bit code, no far jumps, no COMDAT.
  - Fixup kinds: rel32 (self-relative offset, used by `call`/`jmp` rel32)
    and off32 (32-bit absolute offset, used by `mov eax, _label`).
  - Multiple input objects, cross-object PUBDEF/EXTDEF resolution.
  - Single linker pass; produces an MZ+LE with the PMODE/W extender
    stub bundled as the MZ portion (so the .exe self-contains).

The file is mostly format machinery — OMF parsing on the way in, LE
emitting on the way out. Linker logic itself is small: gather text/
data segments in input order, accumulate BSS size, resolve externs
against publics, translate OMF fixups into LE fixup records.
"""
from __future__ import annotations

import argparse
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path


# ----------------------------------------------------------------------
# OMF record parser
# ----------------------------------------------------------------------

# OMF record types we care about. Anything not in this map is skipped.
_R_THEADR     = 0x80
_R_COMENT     = 0x88
_R_MODEND_16  = 0x8A
_R_MODEND_32  = 0x8B
_R_EXTDEF     = 0x8C
_R_PUBDEF_16  = 0x90
_R_PUBDEF_32  = 0x91
_R_LNAMES     = 0x96
_R_SEGDEF_16  = 0x98
_R_SEGDEF_32  = 0x99
_R_GRPDEF     = 0x9A
_R_FIXUPP_16  = 0x9C
_R_FIXUPP_32  = 0x9D
_R_LEDATA_16  = 0xA0
_R_LEDATA_32  = 0xA1


@dataclass
class OmfSegment:
    """One SEGDEF in an .obj.

    `name` and `class_name` are the strings looked up via LNAMES; they
    drive the linker's section bucketing (`_TEXT`/`_DATA`/`_BSS` go to
    the corresponding output objects; `class=CODE` makes the bucket
    executable, etc.). `data` is built up by concatenating LEDATA
    chunks at their declared offsets and zero-filling any gaps; final
    length is `size`.
    """
    name: str
    class_name: str
    size: int
    data: bytearray = field(default_factory=bytearray)


@dataclass
class OmfPublic:
    """A PUBDEF entry — `name` is defined at `seg_idx:offset`."""
    name: str
    seg_idx: int  # 1-based SEGDEF index inside this object
    offset: int


@dataclass
class OmfFixup:
    """One OMF fixup, normalised to the cases we emit.

    `is_self_rel`: True for source-kind 9 (rel32 offset); False for
        source-kind 7 (32-bit absolute offset).
    `seg_idx`:     SEGDEF index this fixup lives in (1-based).
    `offset`:      byte offset within that segment where the patch lands.
    `target`:      either a positive int (1-based SEGDEF index — intra-
                   module, target offset is in `target_offset`) or a
                   negative int (negation of 1-based EXTDEF index, e.g.
                   ``-1`` is the first extern).
    `target_offset`: only meaningful for SEGDEF targets; ignored for EXTDEF.
    """
    is_self_rel: bool
    seg_idx: int
    offset: int
    target: int
    target_offset: int


@dataclass
class OmfObject:
    """Parsed .obj file. Everything indexable is 1-based to match OMF."""
    path: Path
    name: str = ""
    lnames: list[str] = field(default_factory=lambda: [""])  # 1-based
    segments: list[OmfSegment | None] = field(default_factory=lambda: [None])
    publics: list[OmfPublic] = field(default_factory=list)
    externs: list[str] = field(default_factory=lambda: [""])  # 1-based
    fixups: list[OmfFixup] = field(default_factory=list)


def parse_omf(path: Path) -> OmfObject:
    """Parse an OMF .obj into an OmfObject. Raises on malformed input."""
    data = path.read_bytes()
    obj = OmfObject(path=path)

    # FIXUPP records refer back to "the most recent LEDATA" via THREAD
    # state. We track the active segment+offset+length so that each
    # FIXUPP we see can be tied to its data chunk.
    last_segdef_idx = 0       # SEGDEF index of the most recent LEDATA
    last_data_offset = 0      # offset within that segment

    i = 0
    n = len(data)
    while i < n:
        rec_type = data[i]
        rec_len = data[i+1] | (data[i+2] << 8)
        body_end = i + 3 + rec_len - 1   # exclude checksum
        body = data[i+3 : body_end]

        if rec_type == _R_THEADR:
            obj.name = body[1:1+body[0]].decode("latin-1")
        elif rec_type == _R_COMENT:
            pass  # skip — comments don't affect linking
        elif rec_type == _R_LNAMES:
            j = 0
            while j < len(body):
                ln = body[j]; j += 1
                obj.lnames.append(body[j:j+ln].decode("latin-1"))
                j += ln
        elif rec_type in (_R_SEGDEF_16, _R_SEGDEF_32):
            # ACBP byte (1) + length (2 if 16, 4 if 32) + name_idx (1)
            # + class_idx (1) + overlay_idx (1).
            #
            # NB: NASM emits SEGDEF16 with the P-bit (LSB of ACBP) set
            # for USE32 segments — the segment is "USE32 with a 16-bit
            # length field". Code+data segments under 64 KB use this
            # form. Bigger ones promote to SEGDEF32 with a 32-bit length.
            acbp = body[0]
            if rec_type == _R_SEGDEF_32:
                size = struct.unpack_from("<I", body, 1)[0]
                k = 5
            else:
                size = struct.unpack_from("<H", body, 1)[0]
                # ACBP B-bit: if set with 16-bit length field, length
                # 0 actually means 65536. Not commonly hit by NASM but
                # keep the case explicit for safety.
                if (acbp & 0x02) and size == 0:
                    size = 65536
                k = 3
            name_idx = body[k]
            class_idx = body[k+1]
            obj.segments.append(OmfSegment(
                name=obj.lnames[name_idx],
                class_name=obj.lnames[class_idx],
                size=size,
                data=bytearray(size),  # zero-initialised; LEDATA will overwrite
            ))
        elif rec_type == _R_GRPDEF:
            pass  # we don't use group definitions
        elif rec_type == _R_EXTDEF:
            j = 0
            while j < len(body):
                ln = body[j]; j += 1
                name = body[j:j+ln].decode("latin-1"); j += ln
                # type-index byte (always 0 for plain externs)
                _type_idx = body[j]; j += 1
                obj.externs.append(name)
        elif rec_type in (_R_PUBDEF_16, _R_PUBDEF_32):
            # base_group(1 — 0=no group), base_segment(1), base_frame(2 only if seg=0),
            # then repeating: name_len(1) + name + offset(2 or 4) + type_idx(1).
            j = 0
            base_group = body[j]; j += 1
            base_seg = body[j]; j += 1
            if base_seg == 0:
                # absolute frame — record but don't support relocation.
                base_frame = struct.unpack_from("<H", body, j)[0]; j += 2
                _ = base_frame
            wide = (rec_type == _R_PUBDEF_32)
            while j < len(body):
                ln = body[j]; j += 1
                name = body[j:j+ln].decode("latin-1"); j += ln
                if wide:
                    off = struct.unpack_from("<I", body, j)[0]; j += 4
                else:
                    off = struct.unpack_from("<H", body, j)[0]; j += 2
                _type_idx = body[j]; j += 1
                obj.publics.append(OmfPublic(
                    name=name, seg_idx=base_seg, offset=off,
                ))
        elif rec_type in (_R_LEDATA_16, _R_LEDATA_32):
            # seg_idx(1), data_offset(2 or 4), data...
            seg_idx = body[0]
            wide = (rec_type == _R_LEDATA_32)
            if wide:
                data_off = struct.unpack_from("<I", body, 1)[0]
                payload = body[5:]
            else:
                data_off = struct.unpack_from("<H", body, 1)[0]
                payload = body[3:]
            seg = obj.segments[seg_idx]
            if seg is None:
                raise ValueError(
                    f"{path}: LEDATA references undefined segment {seg_idx}"
                )
            seg.data[data_off:data_off+len(payload)] = payload
            last_segdef_idx = seg_idx
            last_data_offset = data_off
        elif rec_type in (_R_FIXUPP_16, _R_FIXUPP_32):
            _parse_fixupp(
                obj, body,
                wide=(rec_type == _R_FIXUPP_32),
                last_segdef_idx=last_segdef_idx,
                last_data_offset=last_data_offset,
            )
        elif rec_type in (_R_MODEND_16, _R_MODEND_32):
            break
        else:
            # Unknown record — skip (OMF defines many we don't use).
            pass

        i += 3 + rec_len

    return obj


def _parse_fixupp(
    obj: OmfObject,
    body: bytes,
    *,
    wide: bool,
    last_segdef_idx: int,
    last_data_offset: int,
) -> None:
    """Decode a FIXUPP record body into OmfFixup entries on `obj`.

    OMF FIXUPPs are conceptually a stream of FIXUP and THREAD subrecords.
    NASM's emitter (output/outobj.c) never emits THREAD subrecords —
    every fixup is self-contained — so we only handle FIXUPs.

    NASM-specific encoding details, reverse-engineered from outobj.c
    `obj_write_fixup` plus empirical checks:

      * The FixData byte's `F` bit (bit 7) is always 0; the FRAME bits
        (6..4) carry the frame method directly, and a frame-index byte
        follows iff method is 0..3 (SEGDEF/GRPDEF/EXTDEF/IMPDEF).
      * The `T` bit (bit 3) is always 0; the target method is derived
        from `P|TARGT` (bits 2..0) — methods 0..3 have a target offset
        following, methods 4..7 don't. A target-index byte ALWAYS
        follows (the index is in the record even when other "thread"
        fields are implicit).
      * NASM auto-attaches `defwrt_type=DEFWRT_SEGMENT` to externs
        declared inside any USE32 segment, so frame method = 0 with a
        frame index pointing to the current segment is the normal
        encoding. We just skip the frame index — the frame doesn't
        affect address computation in our flat 32-bit output.
    """
    i = 0
    while i < len(body):
        b0 = body[i]
        if (b0 & 0x80) == 0:
            # THREAD subrecord — NASM never emits one, but be explicit
            # so future encoders that do don't silently corrupt.
            raise NotImplementedError(
                f"{obj.path}: FIXUPP THREAD subrecord at index {i} "
                f"(byte 0x{b0:02x}) not supported; expected NASM-style "
                f"self-contained FIXUP records only."
            )
        # locat (2 bytes, big-endian-style bit packing):
        #   bit 15: 1 = FIXUP (already checked)
        #   bit 14: M (1 = segment-relative absolute, 0 = self-relative)
        #   bits 13..10: location (4-bit fixup kind)
        #   bits 9..0: data record offset
        b1 = body[i+1]
        i += 2
        is_segrel = bool(b0 & 0x40)
        loc = (b0 >> 2) & 0x0F
        data_off_in_frame = ((b0 & 0x03) << 8) | b1

        if loc != 9:
            raise NotImplementedError(
                f"{obj.path}: FIXUP location {loc} not supported "
                f"(only 32-bit offsets, OMF location 9). "
                f"Source byte: 0x{b0:02x} 0x{b1:02x}."
            )
        is_self_rel = not is_segrel

        # FixData (1 byte): frame method in bits 6..4, target method
        # in bits 2..0 (P|TARGT combined).
        ft_byte = body[i]; i += 1
        frame_method = (ft_byte >> 4) & 0x07
        target_method = ft_byte & 0x07

        # Frame index follows when frame method has an associated index
        # (methods 0..3 = SEGDEF/GRPDEF/EXTDEF/frame-num-with-index).
        # Methods 4..7 have implicit frames (canonic of preceding /
        # canonic of target / target itself / no frame). We don't use
        # the frame value for anything — flat-32 output doesn't need
        # segment frames — so we just skip the index byte.
        if frame_method <= 3:
            i = _skip_index(body, i)

        # Target index always follows; target offset follows iff method
        # is 0..3 (P=0). Methods 4..7 have implicit zero offset.
        target_idx, i = _read_index(body, i)
        if target_method <= 3:
            if wide:
                tgt_off = struct.unpack_from("<I", body, i)[0]; i += 4
            else:
                tgt_off = struct.unpack_from("<H", body, i)[0]; i += 2
        else:
            tgt_off = 0

        # Map target method to (kind, sign-of-target) in our IR:
        #   0, 4 -> SEGDEF target (positive index = our SEGDEF)
        #   2, 6 -> EXTDEF target (negative index)
        if target_method in (0, 4):
            target_signed = target_idx
        elif target_method in (2, 6):
            target_signed = -target_idx
        else:
            raise NotImplementedError(
                f"{obj.path}: target method {target_method} not supported "
                f"(only SEGDEF and EXTDEF targets)."
            )

        # The fixup applies inside the frame data of the most recent
        # LEDATA — so the absolute segment offset is
        # last_data_offset + data_off_in_frame.
        obj.fixups.append(OmfFixup(
            is_self_rel=is_self_rel,
            seg_idx=last_segdef_idx,
            offset=last_data_offset + data_off_in_frame,
            target=target_signed,
            target_offset=tgt_off,
        ))


def _read_index(body: bytes, i: int) -> tuple[int, int]:
    """OMF variable-length index: 1 byte for 0..127, 2 bytes for 128+.

    Format: if high bit of first byte is 0, value = byte. Else value =
    ((byte & 0x7F) << 8) | next_byte.
    """
    b = body[i]
    if (b & 0x80) == 0:
        return b, i+1
    return ((b & 0x7F) << 8) | body[i+1], i+2


def _skip_index(body: bytes, i: int) -> int:
    return _read_index(body, i)[1]


# ----------------------------------------------------------------------
# Linker
# ----------------------------------------------------------------------

# Output object indices (1-based, matching LE convention).
LE_OBJ_TEXT = 1
LE_OBJ_DATA = 2
LE_OBJ_BSS  = 3

# Virtual addresses we assign each output object. Match wlink's layout
# so a side-by-side comparison against an existing wlink-built .exe is
# meaningful: text at 0x10000, data at 0x70000, BSS at 0x80000.
# These move around per-binary in real wlink output (data base is
# determined by where text ends, rounded up); we hardcode for parity
# with the inspected MP.EXE and adjust below if text grows past data.
VA_TEXT_BASE = 0x10000
VA_DATA_BASE = 0x70000
VA_BSS_BASE  = 0x80000

PAGE_SIZE = 4096


@dataclass
class OutputSegment:
    """A piece of an output object — one input segment from one .obj.

    The linker collects these per-output-object, in input order, and
    flattens them into the final image at link time.
    """
    src_obj: OmfObject
    src_seg_idx: int      # 1-based SEGDEF index inside src_obj
    out_obj: int          # LE_OBJ_*
    out_offset: int       # offset within the output object
    size: int


@dataclass
class LinkedImage:
    """Result of linking — what the LE writer consumes."""
    text: bytearray
    data: bytearray
    bss_size: int
    # Symbol table: name -> (out_obj, out_offset). Populated from PUBDEFs
    # after segments are placed.
    symbols: dict[str, tuple[int, int]]
    # Per-output-object fixups: (out_obj, out_offset_in_obj, kind,
    # target_obj, target_offset). Kind is "off32" or "rel32".
    fixups: list[tuple[int, int, str, int, int]]


def _bucket_for_class(class_name: str) -> int | None:
    if class_name == "CODE":
        return LE_OBJ_TEXT
    if class_name == "DATA":
        return LE_OBJ_DATA
    if class_name == "BSS":
        return LE_OBJ_BSS
    return None


def link(objects: list[OmfObject]) -> LinkedImage:
    """Single-pass linker: bucket segments, resolve symbols, translate
    fixups. Input order is preserved within each output object so the
    layout is deterministic."""
    # 1. Place each input segment in its output bucket. Track the
    # mapping (input_obj, input_seg_idx) -> OutputSegment so fixups
    # can locate the right output address.
    out_text: list[OutputSegment] = []
    out_data: list[OutputSegment] = []
    bss_total = 0
    placement: dict[tuple[int, int], OutputSegment] = {}

    text_off = 0
    data_off = 0
    for obj_idx, obj in enumerate(objects):
        for seg_idx, seg in enumerate(obj.segments):
            if seg is None:
                continue
            bucket = _bucket_for_class(seg.class_name)
            if bucket is None:
                # Unknown class — skip with a warning.
                sys.stderr.write(
                    f"pyle: warning: ignoring segment {seg.name!r} "
                    f"(class {seg.class_name!r}) from {obj.path.name}\n"
                )
                continue
            if bucket == LE_OBJ_TEXT:
                placed = OutputSegment(
                    src_obj=obj, src_seg_idx=seg_idx,
                    out_obj=bucket, out_offset=text_off, size=seg.size,
                )
                out_text.append(placed)
                text_off += seg.size
            elif bucket == LE_OBJ_DATA:
                placed = OutputSegment(
                    src_obj=obj, src_seg_idx=seg_idx,
                    out_obj=bucket, out_offset=data_off, size=seg.size,
                )
                out_data.append(placed)
                data_off += seg.size
            else:  # BSS
                placed = OutputSegment(
                    src_obj=obj, src_seg_idx=seg_idx,
                    out_obj=bucket, out_offset=bss_total, size=seg.size,
                )
                bss_total += seg.size
            placement[(id(obj), seg_idx)] = placed

    # 2. Flatten text/data into single buffers.
    text_buf = bytearray(text_off)
    data_buf = bytearray(data_off)
    for placed in out_text:
        seg = placed.src_obj.segments[placed.src_seg_idx]
        text_buf[placed.out_offset:placed.out_offset+seg.size] = seg.data
    for placed in out_data:
        seg = placed.src_obj.segments[placed.src_seg_idx]
        data_buf[placed.out_offset:placed.out_offset+seg.size] = seg.data

    # 3. Build the global symbol table from PUBDEFs.
    symbols: dict[str, tuple[int, int]] = {}
    for obj in objects:
        for pub in obj.publics:
            placed = placement.get((id(obj), pub.seg_idx))
            if placed is None:
                continue
            global_off = placed.out_offset + pub.offset
            if pub.name in symbols:
                raise ValueError(
                    f"pyle: duplicate public symbol {pub.name!r} "
                    f"(from {obj.path.name})"
                )
            symbols[pub.name] = (placed.out_obj, global_off)

    # 4. Translate each input fixup into an output-coordinate fixup.
    #
    # Critical NASM-specific detail: NASM emits SEGDEF-target fixups
    # with OMF method 4 ("SEGDEF, no offset in record") and bakes the
    # offset into the patch site as an addend. EXTDEF-target fixups
    # use method 6 ("EXTDEF, no offset in record") with addend = 0
    # for off32, or whatever displacement makes sense for rel32.
    #
    # The LE FIXUP record's target_off field is what the loader uses;
    # the patch addend is OVERWRITTEN by the loader (for non-additive
    # src_byte 0x07/0x08). So we MUST extract NASM's addend from the
    # patch site and bake it into the LE FIXUP target_off; otherwise
    # all SEGDEF references resolve to (runtime_base + 0) instead of
    # (runtime_base + actual_symbol_offset).
    #
    # For rel32 the addend in the patch is typically 0 (NASM defers
    # the displacement to the linker), so this also handles rel32
    # without special-casing.
    out_fixups: list[tuple[int, int, str, int, int]] = []
    for obj in objects:
        for fix in obj.fixups:
            placed = placement.get((id(obj), fix.seg_idx))
            if placed is None:
                continue
            patch_obj = placed.out_obj
            patch_off = placed.out_offset + fix.offset
            kind = "rel32" if fix.is_self_rel else "off32"

            # Extract the addend NASM wrote at the patch site. Read
            # signed so negative addends (occasionally seen on rel32
            # near a forward reference) round-trip correctly.
            src_seg = obj.segments[fix.seg_idx]
            addend = struct.unpack_from("<i", src_seg.data, fix.offset)[0]

            if fix.target > 0:
                # Intra-module SEGDEF target — placement gives us where
                # this input segment landed in the output object.
                tgt_placed = placement.get((id(obj), fix.target))
                if tgt_placed is None:
                    raise ValueError(
                        f"{obj.path.name}: fixup targets segment "
                        f"{fix.target} which wasn't placed"
                    )
                target_obj = tgt_placed.out_obj
                target_off = (
                    tgt_placed.out_offset
                    + fix.target_offset    # from FIXUPP record (usually 0 for NASM method 4)
                    + addend               # from the patch bytes (where NASM stuffed it)
                )
            else:
                # EXTDEF target — resolve via the global symbol table.
                ext_idx = -fix.target
                ext_name = obj.externs[ext_idx]
                if ext_name not in symbols:
                    raise ValueError(
                        f"{obj.path.name}: undefined extern {ext_name!r}"
                    )
                target_obj, sym_off = symbols[ext_name]
                target_off = sym_off + fix.target_offset + addend

            out_fixups.append((
                patch_obj, patch_off, kind, target_obj, target_off,
            ))

    return LinkedImage(
        text=text_buf, data=data_buf, bss_size=bss_total,
        symbols=symbols, fixups=out_fixups,
    )


# ----------------------------------------------------------------------
# LE writer
# ----------------------------------------------------------------------


# LE flags / constants we need.
LE_FLAG_PER_PROCESS_LIBRARY_INIT = 0x04
LE_FLAG_PROGRAM = 0x0000_0000      # MP.EXE has 0x200 — internal fixups present
LE_FLAG_INTERNAL_FIXUPS = 0x0200

# Object flags (24-byte object table entry, 4th u32):
#   0x0001: readable
#   0x0002: writable
#   0x0004: executable
#   0x0040: 32-bit (BIG)
#   0x2000: zero-filled (no pages on disk for BSS)
LE_OBJ_FLAG_READ = 0x0001
LE_OBJ_FLAG_WRITE = 0x0002
LE_OBJ_FLAG_EXEC = 0x0004
LE_OBJ_FLAG_32BIT = 0x2000


def write_le(
    image: LinkedImage,
    stub: bytes,
    entry_symbol: str,
    out_path: Path,
) -> None:
    """Emit the MZ+LE binary at `out_path`.

    `stub` is the PMODE/W stub bytes (carve from an existing
    wlink-built .exe — see scripts/extract_pmodew_stub.py).
    `entry_symbol` is the name of the program entry point (must be
    in image.symbols and live in the text object).
    """
    if entry_symbol not in image.symbols:
        raise ValueError(f"entry symbol {entry_symbol!r} undefined")
    entry_obj, entry_off = image.symbols[entry_symbol]
    if entry_obj != LE_OBJ_TEXT:
        raise ValueError(
            f"entry symbol {entry_symbol!r} is in object {entry_obj}, "
            f"expected text ({LE_OBJ_TEXT})"
        )

    # 1. Pick virtual base addresses for each object. Mirror wlink's
    # spacing: text at 0x10000, then data at 0x70000 + room, then BSS.
    # If text spills past 0x70000 we shift data up; same for BSS.
    text_size = len(image.text)
    data_size = len(image.data)
    bss_size = image.bss_size

    text_base = VA_TEXT_BASE
    data_base = max(VA_DATA_BASE, _round_up(text_base + text_size, 0x10000))
    bss_base = max(VA_BSS_BASE, _round_up(data_base + data_size, 0x10000))
    bases = {
        LE_OBJ_TEXT: text_base,
        LE_OBJ_DATA: data_base,
        LE_OBJ_BSS:  bss_base,
    }

    # NB: do NOT pre-bake target VAs into the text/data buffers.
    # wlink leaves the source bytes as NASM emitted them (typically 0
    # for off32, displacement-from-next-insn-style for rel32), and the
    # LE loader OVERWRITES them at load time using the LE FIXUP records
    # below. Pre-baking causes addresses to come out as
    # 2*runtime_base + offset on PMODE/W. Verified empirically against
    # wlink-built MP.EXE, where bytes at every fixup site read 0.

    # 3. Build the LE structure incrementally. We compute offsets as
    # we go and patch the header at the end.
    #
    # Layout (offsets relative to LE header start):
    #   0x00..0xC4  : LE header (196 bytes)
    #   0xC4..      : object table (24 * n_obj)
    #                 object page map (4 * n_pages)
    #                 resident name table
    #                 entry table
    #                 fixup page table (4 * (n_pages+1))
    #                 fixup record table
    #   0x?? aligned: [page data area] — text + data pages
    #
    # For simplicity we mirror MP.EXE's structure exactly, then
    # cross-check.

    # Compute the page layout for text + data only (BSS has no pages
    # on disk).
    text_pages = _page_count(text_size)
    data_pages = _page_count(data_size)
    n_pages = text_pages + data_pages
    last_page_size = _last_page_size(text_size, data_size)

    # Build object table (3 entries: text, data, bss).
    # Each entry: virt_size(4), reloc_base(4), flags(4), page_idx(4),
    #             npg(4), reserved(4)
    obj_table = bytearray()
    # Object 1: text
    obj_table += struct.pack(
        "<IIIIIi",
        text_size, text_base,
        LE_OBJ_FLAG_READ | LE_OBJ_FLAG_EXEC | LE_OBJ_FLAG_32BIT
                | (0x40 if False else 0) | (0x05 if False else 0)  # mirror wlink: 0x2045
                | 0x40 | 0x04,  # 0x2045
        1, text_pages, 0,
    )
    # Object 2: data
    obj_table += struct.pack(
        "<IIIIIi",
        data_size, data_base,
        LE_OBJ_FLAG_READ | LE_OBJ_FLAG_WRITE | LE_OBJ_FLAG_32BIT
                | 0x40,  # 0x2043
        1 + text_pages, data_pages, 0,
    )
    # Object 3: bss (no on-disk pages, page_idx=0 per wlink convention
    # — actually wlink puts BSS at page_idx=1+text+data with npg=0,
    # but page_idx=0 also works since npg=0).
    obj_table += struct.pack(
        "<IIIIIi",
        bss_size, bss_base,
        LE_OBJ_FLAG_READ | LE_OBJ_FLAG_WRITE | LE_OBJ_FLAG_32BIT
                | 0x40,  # 0x2043
        1 + text_pages + data_pages, 0, 0,
    )

    # Object page map: one entry per page. Each entry is 4 bytes
    # (3-byte page number + 1-byte flags) — but wlink emits a simpler
    # form where page_number = 1-based page index, flags = 0.
    page_map = bytearray()
    for page_num in range(1, n_pages + 1):
        # 3-byte big-endian page number + 1-byte flags. The "page
        # number" here is the index into the page-data area, 1-based.
        page_map += bytes([
            (page_num >> 16) & 0xFF,
            (page_num >> 8) & 0xFF,
            page_num & 0xFF,
            0,
        ])

    # Resident name table: one entry for the module name (empty here
    # — we can use a placeholder), then a NUL terminator (length 0).
    # Each entry: length(1) + name + ordinal(2 LE).
    module_name = out_path.stem.upper()[:255].encode("ascii", "replace")
    resident_names = bytearray()
    resident_names += bytes([len(module_name)]) + module_name + b"\x00\x00"
    resident_names += b"\x00"  # terminator

    # Entry table: minimal — just terminator (one byte 0x00 = end).
    # The actual entry point is given by eip+init_obj_cs in the LE
    # header, not by entry table ordinals.
    entry_table = b"\x00"

    # Build fixup data per page.
    # `fixups_by_page[page_idx]` is a list of (page_off, kind, target_obj,
    # target_off) where page_idx is 0-based across (text_pages + data_pages).
    fixups_by_page: list[list[tuple[int, str, int, int]]] = [
        [] for _ in range(n_pages)
    ]
    for patch_obj, patch_off, kind, target_obj, target_off in image.fixups:
        if patch_obj == LE_OBJ_TEXT:
            page_base = 0
            obj_off = patch_off
        elif patch_obj == LE_OBJ_DATA:
            page_base = text_pages
            obj_off = patch_off
        else:
            continue  # no fixups in BSS
        page_idx = page_base + (obj_off // PAGE_SIZE)
        page_off = obj_off % PAGE_SIZE
        fixups_by_page[page_idx].append((page_off, kind, target_obj, target_off))

    # Encode each page's fixup records.
    fixup_records = bytearray()
    fixup_page_table = bytearray()
    fixup_page_table += struct.pack("<I", 0)  # page 1 starts at offset 0
    for page_idx in range(n_pages):
        for page_off, kind, target_obj, target_off in fixups_by_page[page_idx]:
            # LE FIXUP source byte:
            #   low nibble = source kind
            #     7 = 32-bit offset
            #     8 = 32-bit self-relative offset
            # NB: wlink emits 0x07 for both flavours and uses the flags
            # byte's bit 0x08 to distinguish; we mirror that to match.
            src_byte = 0x07
            flags = 0x00
            # Bit 0x10 = 32-bit target offset; bit 0x40 = 16-bit
            # object-number. We use 16-bit object number (object < 256
            # is fine but the format specifies 16-bit by default in
            # MP.EXE) and 16-bit target offset where possible.
            # MP.EXE uses 0x00 flags for typical fixups (8-bit obj#,
            # 16-bit target offset). Our objects fit comfortably.
            if kind == "rel32":
                # Self-relative — the source is "kind 8" in the LE spec
                # but MP.EXE uses src_byte=0x07 for both kinds and
                # signals self-relative via... actually MP.EXE doesn't
                # have rel32 cross-page fixups; let's match wlink and
                # use src_byte 0x08 for self-relative.
                src_byte = 0x08

            # Record layout (typical case): src(1) flags(1) src_off(2)
            #                               obj(1) tgt_off(2 or 4)
            # If target_off > 0xFFFF we promote to 32-bit target off.
            if target_off > 0xFFFF:
                flags |= 0x10
                fixup_records += struct.pack(
                    "<BBhBI",
                    src_byte, flags, page_off, target_obj, target_off,
                )
            else:
                fixup_records += struct.pack(
                    "<BBhBH",
                    src_byte, flags, page_off, target_obj, target_off,
                )
        fixup_page_table += struct.pack("<I", len(fixup_records))

    # Now compute layout offsets for the LE header.
    LE_HDR_SIZE = 196
    obj_table_off = LE_HDR_SIZE                                      # 0xC4
    obj_page_map_off = obj_table_off + len(obj_table)
    resident_names_off = obj_page_map_off + len(page_map)
    entry_table_off = resident_names_off + len(resident_names)
    fixup_page_table_off = entry_table_off + len(entry_table)
    fixup_record_table_off = fixup_page_table_off + len(fixup_page_table)
    loader_section_size = (entry_table_off + len(entry_table)) - obj_table_off
    fixup_section_size = len(fixup_page_table) + len(fixup_records)

    # Page data starts after fixups, aligned to 4 bytes (wlink pads).
    pre_data = (fixup_record_table_off + len(fixup_records))
    data_pages_off = _round_up(pre_data, 4)

    # Assemble the LE header (196 bytes).
    le_hdr = bytearray(LE_HDR_SIZE)
    le_hdr[0:2] = b"LE"
    # bytes 2..3: byte/word order = 0,0 (little-endian, both)
    struct.pack_into("<I", le_hdr, 4, 0)         # format level
    struct.pack_into("<H", le_hdr, 8, 2)         # cpu type (2 = 386)
    struct.pack_into("<H", le_hdr, 10, 1)        # os type (1 = OS/2; PMODE/W accepts)
    struct.pack_into("<I", le_hdr, 12, 0)        # module version
    struct.pack_into("<I", le_hdr, 16, LE_FLAG_INTERNAL_FIXUPS)
    struct.pack_into("<I", le_hdr, 20, n_pages)
    struct.pack_into("<I", le_hdr, 24, LE_OBJ_TEXT)   # init obj cs
    struct.pack_into("<I", le_hdr, 28, entry_off)     # eip (offset within text obj)
    struct.pack_into("<I", le_hdr, 32, 0)             # init obj ss (0 = use auto stack)
    struct.pack_into("<I", le_hdr, 36, 0)             # esp (0 = top of stack)
    struct.pack_into("<I", le_hdr, 40, PAGE_SIZE)
    struct.pack_into("<I", le_hdr, 44, last_page_size)
    struct.pack_into("<I", le_hdr, 48, fixup_section_size)
    struct.pack_into("<I", le_hdr, 52, 0)             # fixup checksum
    struct.pack_into("<I", le_hdr, 56, loader_section_size)
    struct.pack_into("<I", le_hdr, 60, 0)             # loader checksum
    struct.pack_into("<I", le_hdr, 64, obj_table_off)
    struct.pack_into("<I", le_hdr, 68, 3)             # number of objects
    struct.pack_into("<I", le_hdr, 72, obj_page_map_off)
    struct.pack_into("<I", le_hdr, 76, 0)             # obj iter data map (we don't use)
    struct.pack_into("<I", le_hdr, 80, 0)             # resource table offset
    struct.pack_into("<I", le_hdr, 84, 0)             # number of resources
    struct.pack_into("<I", le_hdr, 88, resident_names_off)
    struct.pack_into("<I", le_hdr, 92, entry_table_off)
    struct.pack_into("<I", le_hdr, 96, 0)             # module directives
    struct.pack_into("<I", le_hdr, 100, 0)            # number of module directives
    struct.pack_into("<I", le_hdr, 104, fixup_page_table_off)
    struct.pack_into("<I", le_hdr, 108, fixup_record_table_off)
    struct.pack_into("<I", le_hdr, 112, 0)            # imported modules name table
    struct.pack_into("<I", le_hdr, 116, 0)            # number of imported modules
    struct.pack_into("<I", le_hdr, 120, 0)            # imported proc name table
    struct.pack_into("<I", le_hdr, 124, 0)            # per-page checksum table
    # data_pages_offset is FILE-RELATIVE in LE (not LE-header-relative).
    # We add the stub size at write time — pyle places LE right after
    # the stub, so file offset = len(stub) + LE-relative offset.
    struct.pack_into("<I", le_hdr, 128, len(stub) + data_pages_off)
    struct.pack_into("<I", le_hdr, 132, 0)            # number of preload pages
    struct.pack_into("<I", le_hdr, 136, 0)            # non-resident name table
    struct.pack_into("<I", le_hdr, 140, 0)            # non-resident name length
    struct.pack_into("<I", le_hdr, 144, 0)            # non-resident name checksum
    struct.pack_into("<I", le_hdr, 148, 0)            # automatic data obj
    struct.pack_into("<I", le_hdr, 152, 0)            # debug info offset
    struct.pack_into("<I", le_hdr, 156, 0)            # debug info length
    # PMODE/W reads stack size from offset 0xAC (LE-extension /
    # wlink-emitted; some specs document this as `stack_size`). Without
    # it (we write 0), PMODE/W gives the program a degenerate stack and
    # INT 21h calls — which switch to real mode and need a working
    # 32-bit stack for the round-trip — corrupt memory. Empirically
    # confirmed against an early pyle build that printed bytes from the
    # PSP (DOS state at conventional-memory low addresses) instead of
    # the program's data segment. Match wlink's default of 64 KB.
    struct.pack_into("<I", le_hdr, 0xAC, 0x10000)     # stack size
    # bytes 0xB0..0xC0 are reserved — leave 0.

    # Pad pre_data → data_pages_off with zeros.
    pad_after_fixups = bytes(data_pages_off - pre_data)

    # Page data: text pages then data pages, each padded to PAGE_SIZE
    # (except the very last page in the file, which is `last_page_size`
    # bytes per the LE header — matches wlink behaviour).
    page_data = bytearray()
    page_data += image.text
    # Pad text up to a page boundary if data follows.
    if data_size > 0:
        page_data += bytes(_round_up(text_size, PAGE_SIZE) - text_size)
    page_data += image.data
    # Final last-page size is implicit; LE loader uses last_page_size
    # to know how many bytes of the final page are valid.

    # 4. Patch the MZ stub's `new_exe_offset` (at MZ+0x3C) to point to
    # the LE header's start. The PMODE/W stub we carved from MP.EXE
    # already has this field, but it's pointing at MP.EXE's stub-end.
    # If the stub size matches what we use, no change is needed; if it
    # doesn't, we'd have to patch. We use the stub verbatim and require
    # MZ.new_exe_offset == len(stub) — verify and bail if not.
    stub_new_exe = struct.unpack_from("<I", stub, 0x3C)[0]
    if stub_new_exe != len(stub):
        # Patch in place — make a mutable copy.
        stub = bytearray(stub)
        struct.pack_into("<I", stub, 0x3C, len(stub))

    # 5. Write the file.
    with open(out_path, "wb") as f:
        f.write(stub)
        f.write(le_hdr)
        f.write(obj_table)
        f.write(page_map)
        f.write(resident_names)
        f.write(entry_table)
        f.write(fixup_page_table)
        f.write(fixup_records)
        f.write(pad_after_fixups)
        f.write(page_data)


def _round_up(x: int, align: int) -> int:
    return (x + align - 1) & ~(align - 1)


def _page_count(size: int) -> int:
    if size == 0:
        return 0
    return (size + PAGE_SIZE - 1) // PAGE_SIZE


def _last_page_size(text_size: int, data_size: int) -> int:
    if data_size > 0:
        rem = data_size % PAGE_SIZE
        return rem if rem else PAGE_SIZE
    if text_size > 0:
        rem = text_size % PAGE_SIZE
        return rem if rem else PAGE_SIZE
    return 0


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="pyle",
        description="Python OMF→MZ+LE linker (replaces wlink for our use).",
    )
    ap.add_argument(
        "objects", nargs="+", type=Path,
        help="OMF .obj files (from `nasm -f obj`) to link. Order matters: "
             "earlier files take precedence in the section layout.",
    )
    ap.add_argument(
        "-o", "--output", type=Path, required=True,
        help="output .exe path",
    )
    ap.add_argument(
        "--stub", type=Path, required=True,
        help="PMODE/W stub binary (carve from existing wlink-built .exe).",
    )
    ap.add_argument(
        "--entry", default="_pmodew_start",
        help="entry symbol name (must be defined PUBDEF in one of the objects)",
    )
    args = ap.parse_args()

    objects = [parse_omf(p) for p in args.objects]
    image = link(objects)
    stub = args.stub.read_bytes()
    write_le(image, stub, args.entry, args.output)
    print(f"pyle: wrote {args.output} ({args.output.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
