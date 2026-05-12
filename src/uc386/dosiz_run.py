"""dosiz-backed test harness — same shape as `dos_emu.run` but
dispatches through ``../dosiz`` (an in-process dosbox-staging fork
with full DPMI 0.9). Use when the MicroPython port's INT 21h / DPMI
paths trip a unicorn quirk in dos_emu; dosiz handles real DOS
semantics natively.

Toggle from the test harness by setting ``UC386_HARNESS=dosiz`` in
the environment (or by importing ``dosiz_run.run`` directly).

The wrapper accepts a flat ``.bin`` (the same artifact dos_emu eats)
*or* a ready-built ``.exe``. With a flat ``.bin`` it expects a
companion ``.exe`` at the same stem (e.g. ``micropython.bin`` →
``micropython.exe``); building that companion is the caller's job
because the wrapping (extender + stub) is configuration the port
owns, not the harness.

Layout vs. dos_emu:

- dos_emu loads a raw flat binary at linear 0 and runs it under
  unicorn with synthesized PSP / argv. No real DOS bookkeeping.
- dosiz runs an ``.exe`` (MZ or LE) through dosbox-staging's CPU +
  PC hardware. INT 21h / DPMI hit C++ host implementations the way
  a real DOS extender would. Means: bounce buffers, real-mode call
  structures, segment registers — all real, none of the unicorn
  quirks we patched around in pktdrv_int_invoke.

What carries over from dos_emu.run kwargs (unchanged semantics):

- ``stdin_bytes`` — piped to the program via stdin.
- ``timeout_seconds`` — subprocess timeout.
- ``argv`` — passed to the .exe as command-line args.
- ``env`` — exposed to the .exe via the PSP env block (host-side
  envvars set on the subprocess; dosiz copies them in).
- ``vfiles_init`` — written into a per-call temp dir that dosiz
  mounts as drive C:.

Ignored (dos_emu-specific):

- ``instruction_limit`` — dosiz doesn't bound instruction count.
- ``net`` — dosiz doesn't have a NetworkSimulator counterpart yet;
  packet-driver / lwIP tests need separate plumbing.
- ``program_path`` — implicit in the .exe argument to dosiz.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


# Re-export the same Result shape dos_emu uses so callers can swap
# the import line and nothing else changes.
@dataclass
class Result:
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    timed_out: bool = False
    error: str | None = None
    instructions_executed: int | None = None
    _bios_tick: int = 0


def _resolve_dosiz_binary() -> Path:
    """Locate the `dosiz` executable. Check (in order):

    1. ``UC386_DOSIZ`` env var (explicit path the user pinned).
    2. ``../dosiz/build/dosiz`` relative to this file (developer
       layout — uc386 and dosiz are sibling checkouts).
    3. ``dosiz`` on PATH.
    """
    env = os.environ.get("UC386_DOSIZ")
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            return p
        raise FileNotFoundError(
            f"UC386_DOSIZ={env} is set but not a usable file"
        )
    # Sibling checkout: <uc386>/src/uc386/dosiz_run.py →
    # ../../../dosiz/build/dosiz
    sibling = Path(__file__).resolve().parents[3] / "dosiz" / "build" / "dosiz"
    if sibling.is_file():
        return sibling
    on_path = shutil.which("dosiz")
    if on_path:
        return Path(on_path)
    raise FileNotFoundError(
        "dosiz not found. Build it (`make` in ~/src/dosiz) or set "
        "UC386_DOSIZ to the binary path."
    )


def _resolve_exe(binary: Path) -> Path:
    """A flat ``.bin`` isn't directly runnable under dosiz — it
    needs a DOS extender wrapper. If the caller hands us a ``.bin``,
    look for a sibling ``.exe`` at the same stem. If they hand us a
    ``.exe`` (or ``.com``) directly, use it as-is."""
    if binary.suffix.lower() in (".exe", ".com"):
        return binary
    if binary.suffix.lower() == ".bin":
        for ext in (".exe", ".EXE"):
            candidate = binary.with_suffix(ext)
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"dosiz wrapper needs a {binary.stem}.exe alongside "
            f"{binary.name}. Build one via uc386's "
            f"`addons/harness/exe.py` from the .asm produced by the "
            f"port build."
        )
    # Anything else — try it directly; dosiz will reject if it can't
    # load it.
    return binary


def run(
    binary: bytes | Path,
    *,
    timeout_seconds: float = 10.0,
    instruction_limit: int | None = None,   # accepted, ignored
    stdin_bytes: bytes = b"",
    argv: list[str] | None = None,
    env: dict[str, str] | None = None,
    program_path: str | None = None,        # accepted, ignored (implicit)
    vfiles_init: dict[bytes, bytes] | None = None,
    net=None,                                # accepted, ignored (not wired)
) -> Result:
    """Run a DOS program under dosiz, capturing stdout/stderr/exit.

    See module docstring for the kwarg compatibility notes vs.
    ``dos_emu.run``.
    """
    if isinstance(binary, (bytes, bytearray)):
        # dos_emu accepts an in-memory binary by writing it to a
        # tempfile internally. Match that.
        tmp = tempfile.NamedTemporaryFile(
            suffix=".bin", delete=False
        )
        tmp.write(bytes(binary))
        tmp.close()
        binary_path = Path(tmp.name)
        cleanup_binary = True
    else:
        binary_path = Path(binary)
        cleanup_binary = False

    res = Result()
    try:
        exe_path = _resolve_exe(binary_path)
        dosiz_bin = _resolve_dosiz_binary()
    except FileNotFoundError as e:
        res.error = str(e)
        return res

    # vfiles_init → temp directory mounted as drive C:.
    workdir = tempfile.mkdtemp(prefix="dosiz_run_")
    try:
        if vfiles_init:
            for name, data in vfiles_init.items():
                # Names are bytes (DOS 8.3-ish ASCII). Decode loosely;
                # invalid bytes are dropped (DOS doesn't accept them
                # in filenames anyway).
                fname = name.decode("ascii", errors="replace")
                (Path(workdir) / fname).write_bytes(data)

        # Build the .cfg that dosiz consumes. Inline as a sidecar so
        # we don't have to plumb a path through.
        cfg_path = Path(workdir) / "_dosiz.cfg"
        cfg_lines = [
            f"program        = {exe_path}",
            f"drive_C        = {workdir}",
            "default_mode   = binary",
        ]
        cfg_path.write_text("\n".join(cfg_lines) + "\n")

        cmd: list[str] = [str(dosiz_bin), str(cfg_path)]
        if argv:
            cmd.extend(argv)

        proc_env = os.environ.copy()
        if env:
            for k, v in env.items():
                proc_env[k] = v

        try:
            completed = subprocess.run(
                cmd,
                input=stdin_bytes,
                capture_output=True,
                timeout=timeout_seconds,
                env=proc_env,
            )
        except subprocess.TimeoutExpired as e:
            res.timed_out = True
            # Capture what we got before the timeout.
            res.stdout = (e.stdout or b"").decode("latin1",
                                                  errors="replace")
            res.stderr = (e.stderr or b"").decode("latin1",
                                                  errors="replace")
            return res
        except FileNotFoundError as e:
            res.error = f"dosiz invocation failed: {e}"
            return res

        # dos_emu normalizes CRLF → LF in stdout; do the same so
        # tests that grep for "\n42\n" etc. match unchanged.
        res.stdout = completed.stdout.decode(
            "latin1", errors="replace"
        ).replace("\r\n", "\n")
        res.stderr = completed.stderr.decode(
            "latin1", errors="replace"
        ).replace("\r\n", "\n")
        res.exit_code = completed.returncode
        return res
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
        if cleanup_binary:
            try:
                binary_path.unlink()
            except OSError:
                pass
