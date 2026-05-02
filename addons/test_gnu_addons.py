#!/usr/bin/env python3
"""Smoke test the in-tree GNU addons against their manifests.

Each `addons/gnu/<name>/manifest.toml` describes the addon's
sources, expected stdin/argv/vfiles, and expected stdout/exit
behavior. The build harness already runs this validation when
invoked as `python -m addons.harness.build gnu all`; this test
wraps that path so pytest also catches regressions and reports
each addon as a separate test case.

Skips upstream-fetched GNU addons (awk-bwk, gawk, micropython —
those have their own smoke tests and need separately-built bins).

Total wall-clock today: ~2.6s for 16 in-tree addons (true, false,
yes, echo, wc, dirname, head, basename, cat, factor, open_test,
strtol_test, tail, sbase-cat, sbase-tee, sbase-head).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent

# Make the harness importable.
sys.path.insert(0, str(REPO_ROOT))


def _gather_manifests() -> list[Path]:
    """Return manifest.toml paths for every in-tree GNU addon. The
    upstream-fetched addons (awk-bwk, gawk, micropython) don't have
    a top-level manifest.toml — they ship their own build.sh +
    smoke test."""
    out: list[Path] = []
    gnu_root = REPO_ROOT / "addons" / "gnu"
    for sub in sorted(gnu_root.iterdir()):
        if not sub.is_dir() or sub.name.startswith("_"):
            continue
        m = sub / "manifest.toml"
        if m.exists():
            out.append(m)
    return out


_MANIFESTS = _gather_manifests()


@pytest.mark.parametrize(
    "manifest_path",
    _MANIFESTS,
    ids=[m.parent.name for m in _MANIFESTS],
)
def test_gnu_addon_against_manifest(manifest_path: Path) -> None:
    """Compile + run the addon, compare stdout/exit to the manifest's
    expectations."""
    from addons.harness.build import Manifest, build_and_run

    manifest = Manifest.from_path(manifest_path)
    build_dir = REPO_ROOT / "build" / "addons" / manifest.name
    res = build_and_run(manifest, build_dir=build_dir)

    assert res.compile_error is None, (
        f"{manifest.name}: compile failed: {res.compile_error}"
    )
    assert res.run_error is None, (
        f"{manifest.name}: run failed: {res.run_error}"
    )
    assert not res.timed_out, f"{manifest.name}: timed out"
    if res.expected_stdout is not None:
        assert res.stdout == res.expected_stdout, (
            f"{manifest.name}: stdout mismatch\n"
            f"  expected: {res.expected_stdout!r}\n"
            f"  actual:   {res.stdout!r}"
        )
    assert res.exit_code == res.expected_exit, (
        f"{manifest.name}: exit code {res.exit_code}, "
        f"expected {res.expected_exit}"
    )
