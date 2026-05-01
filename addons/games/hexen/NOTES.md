# Hexen port notes

Hexen — Raven's id-Tech-1-derived engine, separate from Heretic's.

**Upstream**: <https://github.com/id-Software/HEXEN>
**License**: GPL-2.0.

## Status (2026-05-01)

`fetch.sh` works (shares chocolate-doom upstream with Heretic).
`uc386_config` is a symlink to `../heretic/uc386_config` so the
generated `config.h` and `SDL_endian.h` shims are shared.

Triage: every `src/hexen/*.c` we tried bails at the same spot —
chocolate-doom's `PACKED_STRUCT(...)` macro spans multiple lines
with a brace-block argument:

```c
typedef PACKED_STRUCT (
{
    short width; short height; ...
}) patch_t;
```

uc_core's preprocessor doesn't merge subsequent lines when a
function-like macro invocation has unclosed parentheses spanning
several lines AND the inner content has braces. PACKED_STRUCT
stays unexpanded, then the parser hits `typedef PACKED_STRUCT (`
and bails with "Expected type specifier".

`_has_unclosed_macro_call` exists in the preprocessor and *does*
merge subsequent lines for unclosed parens — but it's not robust
to brace-content arguments (or maybe a different fault). Worth a
uc_core ticket. Once that's fixed, Hexen should build through
the same `doom_stubs.c` family Doom uses.
