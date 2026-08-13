# dosiz as a uc386 test runner

[`../dosiz`](https://github.com/avwohl/dosiz) is a sibling MS-DOS
emulator that links dosbox-staging's CPU + PC-hardware emulator
in-process and traps DOS INT 21h to C++ host implementations.
Today it runs:

- 16-bit DOS .COM and .EXE (real-mode)
- 32-bit DOS/4GW + PMODE/W bound MZ executables (LE payload)
- Hand-crafted LE binaries (`tests/LE_MIN.EXE`)
- Watcom-cross-compiled `.exe` end-to-end (`tests/HELLO_W.EXE`)

We'd like to use it as a **second runner** for uc386 binaries,
alongside `uc386.dos_emu` (Unicorn-Python). Two runners catch each
other's bugs: dos_emu's INT 21h handler is ~600 lines of Python
covering ~20 syscalls; dosiz's is ~6 KLoC of C++ covering ~50.
Differential testing under both would shake out runner-specific
quirks — and dosiz exercises real DPMI, real video, real timing.

## The gap

uc386 emits flat 32-bit binaries via `nasm -f bin`. They start at
offset 0 with `_start`, expect EAX=argc and EBX=&argv on entry,
and exit via INT 21h AH=4Ch. dosiz's loader chain
(`load_program_at` in `dosiz/src/bridge.cc`) dispatches between:

	load_com_at      — 16-bit, loaded at PSP+0x100
	load_exe_at      — MZ EXE (real-mode 16-bit, or MZ+LE 32-bit
	                  with a recognised DOS-extender stub)
	load_le_at       — bare LE / LX (Linear Executable, 32-bit PM)

There is **no flat-binary loader** in dosiz today. Loading a
uc386 `.bin` would require either side to bridge the gap.

## Two paths forward

### Path A — uc386 wraps its output in MZ+LE

Make uc386 emit a Watcom-style bound executable: an MZ stub
that contains the DOS/4GW loader (or PMODE/W's), followed by an
LE payload carrying the flat 32-bit code. This is what `wlink`
does today.

- Pros: the resulting binary runs on real DOS + DOS/4GW too, not
  just dosiz. A single output works everywhere.
- Cons: substantial new work — page-directory tables, fixup
  records, descriptor setup, the LE header itself. Plus we'd have
  to either bundle DOS/4GW (proprietary) or use PMODE/W (BSD-ish,
  smaller).

### Path B — dosiz gains a flat-bin loader

Add `dosiz --flat-bin program.bin args...` to dosiz: read the
file into a PM-addressable region, install an LDT descriptor,
populate argc/argv at the location uc386 expects, and enter
32-bit PM at offset 0. dosiz already has all the LDT / PM-entry
plumbing from its LE loader; this is mostly a new entry point
that bypasses LE-record parsing.

- Pros: minimal change — likely a few hundred lines in
  `bridge.cc` reusing `load_le_at`'s descriptor setup.
- Cons: the format isn't general — only uc386's flat-bin output
  fits. Real DOS users still couldn't run the binary directly
  (they'd need DOS/4GW + an LE wrapper, which is path A).

**Resolved: Path A shipped, Path B was never needed.** The original
plan was Path B first (quick win), but the `.exe` pipeline landed
instead and turned out to subsume it — see
[`path-a-mz-le.md`](path-a-mz-le.md). uc386 now emits real MZ+LE
executables via `nasm -f obj` + the pure-Python `upyle` linker, and
dosiz already loads those with its existing `load_exe_at` /
`load_le_at` chain. No flat-binary loader was added to dosiz, and
none is planned: Path A's output runs on real DOS too, which was
always the larger payoff.

The sections below are kept for the design record.

## Calling convention dosiz needs to honour

The flat-bin loader needs to match `uc386.dos_emu`'s entry
contract (see `src/uc386/dos_emu.py`):

	EAX = argc
	EBX = address of argv[0] (an array of dword pointers; argv
	      strings live contiguously after the pointer array)
	ESP = top of a 64 KB+ stack
	CS / DS / ES / SS = flat 32-bit selectors with base 0
	                    and limit covering the program's address
	                    space (today dos_emu uses CODE_BASE +
	                    ARGV_BASE + STACK_BASE; dosiz can pick
	                    its own layout)

INT 21h syscalls dosiz needs (intersection with what uc386's
libc emits today):

	02h  putchar
	09h  print string
	3Ch  create handle
	3Dh  open handle
	3Eh  close handle
	3Fh  read handle
	40h  write handle
	41h  unlink
	42h  seek (lseek)
	4Ch  exit  ← required for clean exit + result.exit_code

dosiz already implements all of the above (and more) — see
`bridge.cc`'s INT 21h dispatch table.

## How it actually works now

`src/uc386/dosiz_run.py` is the dosiz-backed runner, with the same
call shape as `dos_emu.run`. Select it per-process:

	UC386_HARNESS=dosiz .venv/bin/pytest tests/

Tests should `from uc386.harness import run` — `harness.py` reads
`UC386_HARNESS` and dispatches to `dosiz_run.run` or `dos_emu.run`,
so the toggle needs no per-test edits.

It takes a ready-built `.exe`, or a flat `.bin` with a companion
`.exe` at the same stem (`micropython.bin` → `micropython.exe`).
Building that companion is the caller's job: the extender and stub
choice is configuration the port owns, not the harness.

`stdin_bytes`, `timeout_seconds`, `argv`, `env`, and `vfiles_init`
carry over unchanged. `instruction_limit` (dosiz doesn't bound
instruction count), `net` (no NetworkSimulator counterpart yet — the
packet-driver / lwIP tests still need dos_emu), and `program_path`
are ignored.

## Status

- 2026-05-01: ask filed in `docs/addons.txt`; this document captured
  the gap and proposed Path B first.
- **Superseded**: Path A (MZ+LE `.exe` output) landed instead, and
  dosiz runs those directly. `dosiz_run.py` + `harness.py` provide
  the second runner. The remaining gap is network simulation, which
  has no dosiz equivalent.
