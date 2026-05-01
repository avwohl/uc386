# Installing uc386

uc386 runs on **Linux** and **macOS** with the same toolchain. The
core driver is pure Python and produces NASM-syntax `.asm` text.
NASM assembles it into a flat 32-bit binary that runs under
`uc386.dos_emu` (Unicorn-backed) for testing.

## TL;DR

	# Linux (Debian / Ubuntu)
	sudo apt-get install -y python3 python3-venv nasm
	python3 -m venv .venv && . .venv/bin/activate
	pip install pytest unicorn "uc_core @ git+https://github.com/avwohl/uc_core@main" -e .
	pytest tests/

	# macOS (Homebrew)
	brew install python@3.12 nasm
	python3.12 -m venv .venv && . .venv/bin/activate
	pip install pytest unicorn "uc_core @ git+https://github.com/avwohl/uc_core@main" -e .
	pytest tests/

	# Fedora / RHEL (dnf)
	sudo dnf install -y python3 python3-virtualenv nasm
	python3 -m venv .venv && . .venv/bin/activate
	pip install pytest unicorn "uc_core @ git+https://github.com/avwohl/uc_core@main" -e .
	pytest tests/

A clean run prints `1320 passed`.

## Required tools

	tool	purpose	apt	brew	dnf
	python ≥ 3.11	driver + uc_core frontend	python3 python3-venv	python@3.12	python3 python3-virtualenv
	nasm	assembler for emitted .asm	nasm	nasm	nasm
	git	cloning uc_core	git	git	git

Notes:

- **Python 3.12 is the recommended target.** uc_core uses
  `dataclass(kw_only=True)` (added in 3.10); 3.11/3.12/3.13 all
  work. Apple's system Python 3.9 is too old — install 3.12 from
  Homebrew on macOS.
- **`uc_core` is a sibling project**, not yet on PyPI. Either
  pip-install it from GitHub (as above), or — if you have a local
  checkout at `../uc_core` — use `pip install -e ../uc_core`
  instead.

## Optional: building the addons

The `addons/` tree (GNU utilities, BWK awk, MicroPython skeleton,
period DOS games) needs additional tools.

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
2.x if you want a guaranteed awk build; otherwise the 16 in-tree
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

	brew install --cask djgpp
	# or build from source via build-djgpp; see https://github.com/andrewwutw/build-djgpp

### OpenWatcom V2 (the period reference compiler)

Linux:

	mkdir -p ~/.local/opt/watcom
	curl -sL -o /tmp/watcom.bin \
	  https://github.com/open-watcom/open-watcom-v2/releases/download/Current-build/open-watcom-2_0-c-linux-x64
	unzip -q /tmp/watcom.bin -d ~/.local/opt/watcom
	export WATCOM=$HOME/.local/opt/watcom
	export INCLUDE=$WATCOM/h
	export PATH="$WATCOM/binl64:$PATH"

macOS: download `open-watcom-2_0-c-macos` from the same
release page and unzip it; set `WATCOM`, `INCLUDE`, and add
`$WATCOM/binmac` (or `binl64` on Apple Silicon w/ Rosetta) to
`PATH`.

The Linux installer is an ELF wrapper around a regular ZIP; running it
under `-i=u` floating-point-faults on some hosts. Unzipping the
embedded archive directly (as above) avoids the installer entirely.

## Verifying the install

	. .venv/bin/activate
	pytest tests/ -q              # 1320 unit tests
	python -m uc386.main examples/hello.c -o /tmp/hello.asm
	nasm -f bin /tmp/hello.asm -o /tmp/hello.bin
	python -c "from pathlib import Path; from uc386.dos_emu import run; print(run(Path('/tmp/hello.bin')))"

If `pytest` reports `1320 passed`, the core install is healthy.
The optional tools only matter if you plan to build addons or
run the comparison report.
