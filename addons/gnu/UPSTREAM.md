# Porting upstream GNU sources to uc386

The trivial in-tree utilities (`true`, `false`, `cat`, `echo`, `head`,
`tail`, `wc`, `yes`, `basename`, `dirname`) demonstrate that uc386 +
the dos_emu runtime can build POSIX-flavored userland from scratch.
Porting real upstream source unmodified is now partly done —
`sbase-cat` / `sbase-head` / `sbase-tee` and BWK awk build from
upstream trees — and a broader coreutils / gawk sweep remains
(addons.txt items 1-5).

## Libc gaps for upstream code

Most of the original gap list has since been filled. Verified
against `lib/i386_dos_libc.asm` on 2026-08-13, the functions GNU
code commonly assumes that uc386's libc **still does not provide**:

- `setenv` / `putenv` — `getenv` exists (reads the real PSP env
  block); writing it does not
- `getopt` / `getopt_long` (gnulib brings its own; needs porting)
- `time` / `gettimeofday` — `clock` exists
- `strtok`, `strspn`, `strcspn`, `strpbrk`, `strndup`
- `vsnprintf` — `snprintf` exists, the `va_list` form does not
- `qsort_r` — plain `qsort` and `bsearch` exist
- `regex.h` (BRE/ERE) — needed by `sed`, `grep`, gawk, `ed`.
  Still the single biggest item.

⚠️ **Still stubs — these link and then lie.** Nothing diagnoses
their use; the program just misbehaves:

| Symbol | Actual behaviour |
|---|---|
| `setbuf` / `setvbuf` | no-op |
| `strerror` | one fixed string, ignores `errno` |
| `popen` / `pclose` | always fail |

`setbuf` and `fflush` being no-ops is honest — output is
write-through, so there is nothing to buffer or flush. `strerror`
is the one that can genuinely mislead: every failure reports the
same message regardless of `errno`.

✅ **Fixed — the file-position and stream-state group is now real.**
`fseek` / `ftell` / `rewind` / `clearerr` / `feof` / `ferror` were
no-op stubs: `feof` was hardwired to 0 (so `while (!feof(f))` never
terminated), `fseek` returned success without seeking, and `ftell`
always returned 0. Seeking now goes through INT 21h AH=0x42 via the
real `_lseek`, and per-stream EOF/error state lives in
`_stdio_flags` — one byte per DOS handle, bit 0 EOF, bit 1 error —
because `FILE*` is the raw handle and there is no FILE struct to
hold it. `fgetc` / `fread` set those bits, and both now check the
carry flag so a DOS error is reported through `ferror` instead of
being returned as if it were a byte count. Pinned by
`tests/test_stdio_position.py`.

**Genuinely filled since this document was first written** (real
INT 21h implementations, not stubs): `getenv`, `strtol` /
`strtoul` / `strtoll`, `errno`, `ungetc`, `system` (AH=0x4B via
COMSPEC), `unlink`, `rename`, `mkdir`, `rmdir`, `stat`, `access`,
`clock`, `realloc`, `strdup`, `strcasecmp`, `strncasecmp`,
`bsearch`, `lseek`, `close`.

Also present: `puts`, `putchar`, `printf`, `snprintf`, `fprintf`,
`sprintf`, `fputs`, `fputc`, `fgetc`, `fgets`, `fread`, `fwrite`,
`fopen`, `fclose`, `getchar`, `getc`, `read`, `write`, `open`,
`creat`, `link`, `malloc`, `calloc`, `free`, `memcpy`, `memmove`,
`memset`, `memcmp`, `memchr`, `mempcpy`, `strlen`, `strcpy`,
`strncpy`, `strcat`, `strncat`, `strcmp`, `strncmp`, `strchr`,
`strrchr`, `strstr`, `isalpha`, `isdigit`, `isspace`, `tolower`,
`toupper`, `atoi`, `atol`, `atof`, `abs`, `labs`, `llabs`,
`signal`, `raise`, `setjmp`, `longjmp`, `qsort`, `exit`, `abort`,
`perror`, `tmpnam`, `remove`, math (`sin`, `cos`, `sqrt`, `pow`,
`floor`, `ceil`, `fabs`).

## Strategy

Two parallel tracks:

1. **Add missing libc** — incrementally extend `lib/i386_dos_libc.asm`
   with shims for the most-needed functions, keeping the uc386 suite
   (460 passed, 1 skipped) green, and no regression against the
   c-testsuite / gcc-c-torture corpora (see `../../README.md` for the
   current rates and `../../CLAUDE.md` for how to run them)
   green at every step. Items 1-6 of the original priority list are
   done; what's left, in order:
   1. `strtok` / `strspn` / `strcspn` / `strpbrk` — small, and a
      surprising amount of upstream string code wants them
   2. `getopt` (rolled in C, ~50 lines)
   3. `vsnprintf` (refactor `snprintf`'s core to take a `va_list`)
   4. `time` / `gettimeofday` (INT 21h AH=2Ah/2Ch)
   5. `setenv` / `putenv` — needs a writable copy of the env block
   6. `regex.h` — biggest item. Either port `regcomp`/`regexec`
      from glibc (~3000 lines) or use `re_search` / Spencer's regex.

2. **Pick small upstream targets** — single-file utilities with
   minimal gnulib dependencies. **Three have landed**: `sbase-cat`,
   `sbase-head`, and `sbase-tee` build verbatim from
   [sbase](https://git.suckless.org/sbase) upstream via the shared
   `_sbase_shim/util.c`, and BWK awk compiles ~6 KLoC of upstream C
   unmodified. The shim pattern is the template for the rest:
   - `nl` (~300 lines, line numbering)
   - `sum` / `cksum` (~200 lines each)
   - `tac` (reverse cat — uses `mmap` under coreutils, simpler
     port reads to memory)
   - `tr` (~600 lines)
   - `paste` (~300 lines)

   For each: copy the upstream `.c` into `addons/gnu/<name>/`,
   stub the `system.h` / `config.h` includes with minimal local
   shims, document the patches in `addons/gnu/<name>/UPSTREAM.md`.

3. **Bigger targets** once libc is fuller: `sort`, `uniq`, `cut`,
   `comm`, `od`, `expand`, `unexpand`, `fold`, `pr`. These mostly
   just need bigger I/O and string handling we already have.

4. **gawk** (addons.txt #4): the tallest port. Source is ~50K LoC
   organized as a multi-file build. Needs regex (above). Plan:
   download tarball, configure with a stub `config.h`, pick the
   subset of source files that don't depend on `gettext` /
   `mbtowc` / locale.h / `dlfcn.h`, build that subset, see what
   breaks. Likely a multi-iteration effort.

## Fetch convention

Each `addons/gnu/<name>/` MAY contain a `fetch.sh` that downloads
upstream source into `addons/gnu/<name>/upstream/`. The harness
will skip already-fetched directories. Fetched sources are
**not** checked into this repo — they're each upstream's
responsibility, and we don't want to duplicate their releases.

For the FOSS installer we ship binaries we built ourselves;
the source provenance is whatever upstream archive `fetch.sh`
pointed at. Each tool's `LICENSE` file (downloaded by `fetch.sh`)
travels with the binary.
