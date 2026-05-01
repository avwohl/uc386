# Heretic / Hexen port notes

Heretic and Hexen share Doom's engine (Raven Software extended id's
codebase). Public source releases:

- Heretic: <https://github.com/chocolate-doom/chocolate-doom> (chocolate
  port preserves DOS-era semantics)
- Original Raven release: <https://github.com/id-Software/Heretic>

**License**: GPL-2.0 (id Tech 1 codebase since 1999).

Build prospects track Doom's — same engine, same blockers. Once Doom
builds, these are mostly recompiles + per-game asset paths.

`fetch.sh` and `build.sh` are intentionally absent until Doom itself
builds; copying them now without progress on the underlying engine
would just be aspirational.
