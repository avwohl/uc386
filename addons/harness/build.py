#!/usr/bin/env python3
"""Build/test harness for uc386 addons.

Wraps `python -m uc386.main` + `uc386.dos_emu.assemble_and_run` into a
single command suitable for both the in-tree trivial addons (phase 2)
and downloaded upstream sources (phases 4-5). Each addon directory
contains a `manifest.toml` that tells the harness:

  * what sources to compile (one or many `.c` files)
  * what stdin / argv to pass when running
  * what stdout / exit code to expect (golden test)

The harness is intentionally small: no parallelism, no per-target
build cache — those land later if needed. Target use is `python -m
addons.harness.build` from the repo root inside the `.venv`.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ADDONS_ROOT = REPO_ROOT / "addons"
LIB_INCLUDE = REPO_ROOT / "src" / "uc386" / "lib" / "include"
EXAMPLES_DIR = REPO_ROOT / "examples"


@dataclass
class Manifest:
    """Per-addon build description, parsed from `manifest.toml`."""

    name: str
    sources: list[Path]
    description: str = ""
    expect_stdout: str | None = None
    expect_exit: int = 0
    stdin: bytes = b""
    argv: list[str] = field(default_factory=list)
    vfiles: dict[bytes, bytes] = field(default_factory=dict)
    timeout_seconds: float = 10.0
    instruction_limit: int = 200_000_000
    extra_cflags: list[str] = field(default_factory=list)

    @classmethod
    def from_path(cls, manifest_path: Path) -> "Manifest":
        data = tomllib.loads(manifest_path.read_text())
        base = manifest_path.parent
        # vfiles in TOML: a [vfiles] table mapping name → string content
        # (relative path to a file is also accepted as `{file = "..."}`).
        vfiles_raw = data.get("vfiles", {}) or {}
        vfiles: dict[bytes, bytes] = {}
        for name, val in vfiles_raw.items():
            if isinstance(val, dict) and "file" in val:
                vfiles[name.encode("utf-8")] = (base / val["file"]).read_bytes()
            else:
                vfiles[name.encode("utf-8")] = str(val).encode("utf-8")
        return cls(
            name=data.get("name") or base.name,
            description=data.get("description", ""),
            sources=[base / s for s in data["sources"]],
            expect_stdout=data.get("expect_stdout"),
            expect_exit=int(data.get("expect_exit", 0)),
            stdin=data.get("stdin", "").encode("utf-8"),
            argv=list(data.get("argv", [])),
            vfiles=vfiles,
            timeout_seconds=float(data.get("timeout_seconds", 10.0)),
            instruction_limit=int(data.get("instruction_limit", 200_000_000)),
            extra_cflags=list(data.get("extra_cflags", [])),
        )


@dataclass
class BuildResult:
    name: str
    asm_path: Path | None = None
    asm_bytes: int = 0
    stdout: str = ""
    exit_code: int | None = None
    timed_out: bool = False
    compile_error: str | None = None
    run_error: str | None = None
    expected_stdout: str | None = None
    expected_exit: int = 0

    @property
    def ok(self) -> bool:
        if self.compile_error or self.run_error or self.timed_out:
            return False
        if self.exit_code != self.expected_exit:
            return False
        if self.expected_stdout is not None and self.stdout != self.expected_stdout:
            return False
        return True

    def summary(self) -> str:
        if self.compile_error:
            return f"{self.name}: COMPILE FAIL — {self.compile_error[:200]}"
        if self.run_error:
            return f"{self.name}: RUN FAIL — {self.run_error[:200]}"
        if self.timed_out:
            return f"{self.name}: TIMEOUT"
        status = "PASS" if self.ok else "FAIL"
        bits = [f"{self.name}: {status}"]
        if self.asm_bytes:
            bits.append(f"{self.asm_bytes}B asm")
        if self.exit_code is not None:
            bits.append(f"exit={self.exit_code}")
        if (
            self.expected_stdout is not None
            and self.stdout != self.expected_stdout
        ):
            bits.append(f"stdout={self.stdout!r} (want {self.expected_stdout!r})")
        return " ".join(bits)


def compile_one(
    sources: list[Path],
    out_asm: Path,
    *,
    extra_cflags: list[str] | None = None,
) -> str | None:
    """Run `python -m uc386.main` on `sources`. Returns error string or None."""
    cmd = [
        sys.executable, "-m", "uc386.main",
        *[str(s) for s in sources],
        "-o", str(out_asm),
        "-I", str(LIB_INCLUDE),
        *(extra_cflags or []),
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=REPO_ROOT, timeout=60,
        )
    except subprocess.TimeoutExpired:
        return "uc386 timed out after 60s"
    if proc.returncode != 0:
        return proc.stderr.strip() or proc.stdout.strip() or "uc386 failed"
    if not out_asm.exists():
        return "uc386 produced no output"
    return None


def build_and_run(manifest: Manifest, *, build_dir: Path) -> BuildResult:
    res = BuildResult(
        name=manifest.name,
        expected_stdout=manifest.expect_stdout,
        expected_exit=manifest.expect_exit,
    )
    build_dir.mkdir(parents=True, exist_ok=True)
    asm_path = build_dir / f"{manifest.name}.asm"
    err = compile_one(
        manifest.sources, asm_path, extra_cflags=manifest.extra_cflags,
    )
    if err:
        res.compile_error = err
        return res
    res.asm_path = asm_path
    res.asm_bytes = asm_path.stat().st_size

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from uc386.dos_emu import assemble_and_run

    try:
        emu = assemble_and_run(
            asm_path,
            timeout_seconds=manifest.timeout_seconds,
            instruction_limit=manifest.instruction_limit,
            stdin_bytes=manifest.stdin,
            argv=manifest.argv or None,
            vfiles_init=manifest.vfiles or None,
        )
    except Exception as e:
        res.run_error = f"{type(e).__name__}: {e}"
        return res

    res.stdout = emu.stdout
    res.exit_code = emu.exit_code
    res.timed_out = emu.timed_out
    if emu.error:
        res.run_error = emu.error
    return res


def find_manifests(category: str | None, name: str | None) -> list[Path]:
    """Locate manifests under addons/. Returns absolute paths.

    `name` of None or "all" expands to every manifest under the
    category roots.
    """
    out: list[Path] = []
    if category is None:
        roots = [ADDONS_ROOT / "gnu", ADDONS_ROOT / "games"]
    else:
        roots = [ADDONS_ROOT / category]
    want_all = name in (None, "all")
    for root in roots:
        if not root.exists():
            continue
        if not want_all:
            m = root / name / "manifest.toml"
            if m.exists():
                out.append(m)
            continue
        for sub in sorted(root.iterdir()):
            if sub.is_dir() and (sub / "manifest.toml").exists():
                out.append(sub / "manifest.toml")
    return out


def smoke() -> int:
    """Compile + run examples/hello.c through the harness path.

    Sanity check that the toolchain is installed and the harness can
    drive it end-to-end. hello.c prints one line via libc printf and
    returns 0, so a healthy toolchain gives exit_code=0 and
    "Hello, DOS!\\n" on stdout — which also proves the libc actually
    reached INT 21h rather than just linking.
    """
    print("== smoke: compile + run examples/hello.c ==")
    build_dir = REPO_ROOT / "build" / "addons-smoke"
    res = build_and_run(
        Manifest(
            name="hello",
            description="examples/hello.c smoke test",
            sources=[EXAMPLES_DIR / "hello.c"],
            expect_stdout="Hello, DOS!\n",
            expect_exit=0,
        ),
        build_dir=build_dir,
    )
    print(res.summary())
    return 0 if res.ok else 1


def run_manifests(paths: list[Path]) -> int:
    """Run each manifest. Returns 0 iff all pass."""
    if not paths:
        print("No manifests found.")
        return 1
    fails = 0
    results: list[BuildResult] = []
    for mp in paths:
        manifest = Manifest.from_path(mp)
        build_dir = REPO_ROOT / "build" / "addons" / manifest.name
        res = build_and_run(manifest, build_dir=build_dir)
        results.append(res)
        print(res.summary())
        if not res.ok:
            fails += 1
    print()
    print(f"== {len(results) - fails}/{len(results)} passed ==")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="addons.harness.build")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("smoke", help="compile+run examples/hello.c (sanity)")
    p_gnu = sub.add_parser("gnu", help="build a gnu/<name> addon")
    p_gnu.add_argument("name", nargs="?", default="all",
                       help="addon name (omit or 'all' for everything)")
    p_games = sub.add_parser("games", help="build a games/<name> addon")
    p_games.add_argument("name", nargs="?", default="all",
                         help="addon name (omit or 'all' for everything)")
    sub.add_parser("all", help="build everything under gnu/ and games/")
    args = ap.parse_args()

    if args.cmd == "smoke":
        return smoke()
    if args.cmd in ("gnu", "games"):
        return run_manifests(find_manifests(args.cmd, args.name))
    if args.cmd == "all":
        return run_manifests(find_manifests(None, None))
    ap.error(f"unknown command: {args.cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
