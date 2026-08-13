# Installing uc386

uc386 runs on **Linux** and **macOS** with the same toolchain. The
core driver is pure Python and produces NASM-syntax `.asm` text.
NASM assembles it into a flat 32-bit binary that runs under
`uc386.dos_emu` (Unicorn-backed) for testing, or into a DOS `.exe`
via the `addons/harness/` pipeline.

## Just want the compiler?

	pip install uc386
	# plus nasm from your system package manager

That pulls the frontend (`uc_core`, `uplox`) and the asm-level
optimizer (`upeep386`) from PyPI automatically, and ships the DOS
libc (`i386_dos_libc.asm`) and the `lib/include/` headers. It gives
you `.c` → `.asm`. Add `pip install unicorn` to run the output under
`dos_emu`, and `pip install upyle` plus a source checkout to build
`.exe` files (see **Building `.exe` files** below).

## TL;DR — source checkout

	# Linux (Debian / Ubuntu)
	sudo apt-get install -y python3 python3-venv nasm git
	git clone https://github.com/avwohl/uc386 && cd uc386
	python3 -m venv .venv && . .venv/bin/activate
	pip install pytest unicorn upyle -e .
	pytest tests/

	# macOS (Homebrew)
	brew install python@3.12 nasm git
	git clone https://github.com/avwohl/uc386 && cd uc386
	python3.12 -m venv .venv && . .venv/bin/activate
	pip install pytest unicorn upyle -e .
	pytest tests/

	# Fedora / RHEL (dnf)
	sudo dnf install -y python3 python3-virtualenv nasm git
	git clone https://github.com/avwohl/uc386 && cd uc386
	python3 -m venv .venv && . .venv/bin/activate
	pip install pytest unicorn upyle -e .
	pytest tests/

A clean run prints `460 passed, 1 skipped`.

## Required tools

	tool	purpose	apt	brew	dnf
	python ≥ 3.11	driver + uc_core frontend	python3 python3-venv	python@3.12	python3 python3-virtualenv
	nasm	assembler for emitted .asm	nasm	nasm	nasm
	git	source checkout	git	git	git

Notes:

- **Python 3.12 is the recommended target; 3.11 is the floor.** 3.11
  through 3.13 are tested in CI. The floor is set by a transitive
  dependency rather than by uc386's own code: uc_core requires uplox,
  and uplox declares `requires-python >=3.11`, so on 3.10
  `pip install uc386` fails with `ResolutionImpossible` ("no matching
  distributions available ... uplox"). Apple's system Python 3.9 is
  far too old — install 3.12 from Homebrew on macOS.
- **The sibling packages are all on PyPI** and resolve automatically:
  `uc_core` (frontend), `uplox` (its parser generator), and
  `upeep386` (peephole optimizer, asm DCE, libc splitter) are
  declared dependencies of uc386.
- **`upyle` is the exception.** It is the pure-Python OMF→MZ+LE
  linker used by the `.exe` pipeline, which lives in `addons/` and
  isn't packaged — so uc386 does not declare it. Install it
  explicitly when you want `.exe` output.
- ⚠️ **The repo is `pyle`; the package is `upyle`.** `pip install
  upyle`, never `pip install pyle` — the bare name on PyPI belongs
  to an unrelated project.
- To **co-develop** a sibling, clone it next to uc386 and install
  that one editable (`pip install -e ../uc_core`); the rest can stay
  on PyPI. See `CLAUDE.md` for that layout.

## Building `.exe` files

The DOS-extender pipeline needs the source checkout (it lives in
`addons/harness/`, not in the wheel), `nasm`, and `upyle`:

	pip install upyle
	python -m addons.harness.exe addons/gnu/true/main.c -o true.exe

It defaults to the **DOS/32A** extender and needs no Open Watcom —
`upyle` does the linking and ships a pre-bound stub. Open Watcom's
`wlink` is required only for the `causeway` and `dos4g` extenders.
See [`path-a-mz-le.md`](path-a-mz-le.md).

## Optional: building the addons

The `addons/` tree (GNU utilities, BWK awk, period DOS games) needs
additional tools.

	tool	for	apt	brew	dnf
	bison	awk-bwk parser generator	bison	bison	bison
	flex	awk-bwk lexer generator	flex	flex	flex
	make	upstream Makefiles	build-essential	(part of Xcode CLT)	make
	cc	bootstrap host gcc/clang	build-essential	(part of Xcode CLT)	gcc
	curl	fetch.sh scripts	curl	curl	curl
	unzip	some upstream archives	unzip	unzip	unzip

	# Linux: one-shot
	sudo apt-get install -y bison flex build-essential curl unzip

	# macOS: install the Xcode Command Line Tools once
	xcode-select --install
	brew install bison flex curl

	# Fedora / RHEL
	sudo dnf install -y bison flex make gcc curl unzip

**Bison version note.** awk-bwk's `awkgram.y` exercises a typedef-
chain pattern that uc386 currently misparses when bison 3.x emits
the parser table; bison 2.3 (Homebrew on the dev workstation,
historical macOS default) avoids it. The release CI ships the
FOSS tarball without awk-bwk if the build trips this. Use bison
2.x if you want a guaranteed awk build; otherwise the in-tree
GNU utilities still build fine.

## Optional: size-comparison column (`addons/results.md`)

`addons/harness/compare.py` shows uc386 binary size next to the
same source built by competing DOS toolchains. Both compilers
are optional — `compare.py` skips any column whose toolchain is
missing.

### DJGPP (gcc → DOS, COFF / DPMI)

Linux:

	mkdir -p ~/.local/opt
	curl -sL -o /tmp/djgpp.tar.bz2 \
	  https://github.com/andrewwutw/build-djgpp/releases/download/v3.4/djgpp-linux64-gcc1220.tar.bz2
	tar xjf /tmp/djgpp.tar.bz2 -C ~/.local/opt
	export PATH="$HOME/.local/opt/djgpp/bin:$PATH"

macOS:

	# No Homebrew formula for DJGPP. Use the prebuilt tarball from
	# andrewwutw/build-djgpp (same release as Linux, different artifact):
	mkdir -p ~/.local/opt
	curl -sL -o /tmp/djgpp.tar.bz2 \
	  https://github.com/andrewwutw/build-djgpp/releases/download/v3.4/djgpp-osx-gcc1220.tar.bz2
	tar xjf /tmp/djgpp.tar.bz2 -C ~/.local/opt
	export PATH="$HOME/.local/opt/djgpp/bin:$PATH"

	# `addons/harness/compare.py` already searches ~/.local/opt/djgpp/bin
	# so the export is only needed if you want to call
	# `i586-pc-msdosdjgpp-gcc` directly from a shell.

### OpenWatcom V2 (the period reference compiler)

Linux — native build:

	mkdir -p ~/.local/opt/watcom
	curl -sL -o /tmp/watcom.bin \
	  https://github.com/open-watcom/open-watcom-v2/releases/download/Current-build/open-watcom-2_0-c-linux-x64
	unzip -q /tmp/watcom.bin -d ~/.local/opt/watcom
	export WATCOM=$HOME/.local/opt/watcom
	export INCLUDE=$WATCOM/h
	export PATH="$WATCOM/binl64:$PATH"

The Linux installer is an ELF wrapper around a regular ZIP; running it
under `-i=u` floating-point-faults on some hosts. Unzipping the
embedded archive directly (as above) avoids the installer entirely.

macOS — **no native build**, so run the *DOS-hosted* toolchain under
DOSBox-X. `addons/harness/watcom_dosbox.py` drives this automatically,
and `compare.py` falls back to it, so the Watcom column stays
*checked, not asserted* on a Mac. Roughly 99 MB extracted:

	brew install dosbox-x
	curl -sL -o /tmp/ow.exe \
	  https://github.com/open-watcom/open-watcom-v2/releases/download/Current-build/open-watcom-2_0-c-dos.exe
	mkdir -p ~/.local/opt/watcom-dos
	unzip -q /tmp/ow.exe 'binw/*' 'h/*' 'lib386/dos/*' 'lib386/*.lib' \
	  -d ~/.local/opt/watcom-dos

It searches `$WATCOM_DOS_DIR`, then `~/.local/opt/watcom-dos`, then
`/tmp/watcom`. (The Linux x64 build won't run on Darwin even under
Rosetta — Rosetta translates x86_64 → arm64 user space, it doesn't
bridge the Linux ABI.)

## Verifying the install

	. .venv/bin/activate
	pytest tests/ -q              # 460 passed, 1 skipped
	python -m uc386.main examples/hello.c -o /tmp/hello.asm
	nasm -f bin /tmp/hello.asm -o /tmp/hello.bin
	python -c "from pathlib import Path; from uc386.dos_emu import run; print(run(Path('/tmp/hello.bin')).stdout)"

That last line prints `Hello, DOS!`. If `pytest` reports
`460 passed, 1 skipped`, the core install is healthy. The optional
tools only matter if you plan to build addons or run the comparison
report.

The peephole / asm-DCE / libc-split tests now live in the sibling
package: `pytest ../upeep386/tests` prints `897 passed` against a
checkout of [upeep386](https://github.com/avwohl/upeep386).
