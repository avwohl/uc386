#!/usr/bin/env python3
"""Reverse-mangle MP_QSTR_<sanitized> macro names back to the
original qstr source string.

Upstream's `tools/makeqstrdata.py:qstr_escape` walks each qstr and
replaces every non-`[A-Za-z0-9]` byte with `_<name>_`, where
`<name>` comes from `codepoint2name` (HTML entity names + a small
custom map) or `0x%02x` for the rest. Our triage build greps for
`MP_QSTR_<sanitized>` references in the source; the *macro* name
is what's been sanitized — the original qstr string for `\\n` is
1 byte, but it appears in source as `MP_QSTR__0x0a_` (5-byte
sanitized form). If we emit the sanitized form as the qstr's
4th-field string, `qstr_find_strn("\\n", 1)` misses, AND output
qstrs (e.g. `print()`'s trailing newline) render as the literal
text `_0x0a_` instead of a newline.

This script reads MP_QSTR_<x> tokens on stdin (one per line, may
contain dups), reverses the escape, and emits one
`QDEF1(macro, 0, len, "<orig>")` line per UNIQUE token sorted by
the **original string** (ASCII byte order). Sort key matters:
qstr_find_strn does `strncmp(probe_str, pool->qstrs[mid], n)` —
the comparison key at runtime is the un-escaped string, so the
pool's `is_sorted=true` invariant requires that order. Sorting
by macro name happens to coincide for pure-identifier qstrs
(`print`, `__name__`) but breaks for escaped ones
(`MP_QSTR__0x0a_` lex-orders near `_`, while its actual string
`\\n` = 0x0A would sort before space).
"""
from __future__ import annotations

import sys

# Pull codepoint2name from upstream verbatim — same source of truth
# as makeqstrdata.qstr_escape so the inverse is exact. Caller is
# expected to set sys.path so `upstream.py.makeqstrdata` resolves.
sys.path.insert(0, "upstream/py")
from makeqstrdata import codepoint2name  # type: ignore[import-not-found]

# Inverse map: HTML entity name -> single-character byte string.
# Two filters:
#   1. Codepoint must be < 256 — qstrs are byte sequences, so
#      escapes for high-codepoint Unicode chars never appear in real
#      source. Without this, `_omega_` etc. would false-match.
#   2. The decoded char must NOT itself be an identifier char
#      (`[A-Za-z0-9_]`). Upstream's `qstr_escape` only produces
#      `_<name>_` wrappers for NON-identifier chars (the regex
#      `RE_NO_ESCAPE = r"[A-Za-z0-9_]"` passes identifier chars
#      through unchanged). So `_<name>_` in a macro tail can only
#      have come from escaping a punctuation/whitespace byte —
#      never from an alphanumeric escape. This filter eliminates
#      false matches like `__not__` (a real Python dunder, not an
#      escape of `¬` U+00AC) and `__and__` (likewise, not `∧`).
_IDENT_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
# Restrict to ASCII printable punctuation (32–126, excluding the
# identifier subset). Control chars (`\n`, `\t`, …) and high-byte
# chars (¬ U+00AC, ∧ U+2227, Α U+0391, …) are handled either via
# the `0x%02x` literal path or are simply unreachable in real qstr
# source — `__not__` is a real Python dunder, not an escape for `¬`.
name2char = {
    name: chr(cp)
    for cp, name in codepoint2name.items()
    if 32 <= cp <= 126 and chr(cp) not in _IDENT_CHARS
}


def unescape(macro_tail: str) -> str:
    """Reverse qstr_escape on the part after `MP_QSTR_`.

    Walks left-to-right. Plain `[A-Za-z0-9]` runs pass through.
    A `_<name>_` group decodes to a single byte: `name` is either
    an HTML entity name (`lt`, `gt`, `space`, `hyphen`, ...) or a
    `0x%02x` literal. The leading `_` and trailing `_` framing the
    name come from `qstr_escape`'s `"_" + name + "_"` template.

    Edge case: an underscore that's part of the *original* string
    (e.g. `__name__`) ALSO gets sanitized — but to itself, since
    `_` is in `RE_NO_ESCAPE` upstream:

        RE_NO_ESCAPE = re.compile(r"[A-Za-z0-9_]")

    So plain `_` runs through verbatim. The only `_<name>_` groups
    that exist are the genuine escape sequences. Disambiguating
    works because every escape `name` is at minimum 2 chars
    (`lt`, `gt`, `0x..`) — a lone `_` is never a wrapper.
    """
    out: list[str] = []
    i = 0
    n = len(macro_tail)
    while i < n:
        c = macro_tail[i]
        if c != "_":
            out.append(c)
            i += 1
            continue
        # Found a `_` — could be a literal underscore in the source
        # qstr (e.g. `__name__`) or the start of `_<name>_` where
        # `<name>` is an entry in `codepoint2name` (HTML entity name)
        # or `0xNN` (hex byte literal). Some entity names CONTAIN
        # underscores themselves: `brace_open`, `brace_close`,
        # `paren_open`, `bracket_open`, etc. So a naive
        # `find("_", i+1)` would split `_brace_open_` on the inner
        # `_` and produce `brace` as the candidate name (which
        # doesn't match anything, so we'd silently drop the escape).
        # Iterate forward over EVERY `_` that follows and accept the
        # first candidate that's a known name (longest valid by
        # construction — names don't share prefixes that are also
        # names). Fall back to literal `_` if no closing `_` matches.
        matched = False
        j = i + 1
        while True:
            k = macro_tail.find("_", j)
            if k == -1:
                break
            candidate = macro_tail[i + 1 : k]
            if candidate in name2char:
                out.append(name2char[candidate])
                i = k + 1
                matched = True
                break
            if (
                len(candidate) == 4
                and candidate[:2] == "0x"
                and all(ch in "0123456789abcdef" for ch in candidate[2:])
            ):
                out.append(chr(int(candidate[2:], 16)))
                i = k + 1
                matched = True
                break
            j = k + 1
        if not matched:
            out.append("_")
            i += 1
    return "".join(out)


def c_string(s: str) -> str:
    """Render `s` as a C string literal — escape `"`, `\\`, and any
    byte outside printable ASCII."""
    parts: list[str] = []
    for ch in s:
        b = ord(ch)
        if ch == "\\":
            parts.append("\\\\")
        elif ch == '"':
            parts.append('\\"')
        elif ch == "\n":
            parts.append("\\n")
        elif ch == "\t":
            parts.append("\\t")
        elif ch == "\r":
            parts.append("\\r")
        elif 0x20 <= b < 0x7F:
            parts.append(ch)
        else:
            parts.append(f"\\x{b:02x}")
    return '"' + "".join(parts) + '"'


def main() -> int:
    seen: dict[str, str] = {}  # macro -> original
    for line in sys.stdin:
        macro = line.strip()
        if not macro.startswith("MP_QSTR_"):
            continue
        if macro in seen:
            continue
        # `MP_QSTR_` is 8 chars; the rest is the sanitized qstr.
        seen[macro] = unescape(macro[8:])

    # Sort by the original string (ASCII byte order), which is the
    # runtime binary search key. Tiebreak by macro name for stable
    # output across builds (qstr_escape is 1:1, so ties shouldn't
    # arise in practice — but a deterministic order keeps diffs
    # clean if a duplicate slips through).
    out_w = sys.stdout.write
    for macro, original in sorted(
        seen.items(), key=lambda item: (item[1].encode("utf-8"), item[0])
    ):
        out_w(
            f"QDEF1({macro}, 0, {len(original)}, {c_string(original)})\n"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
