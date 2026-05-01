# Porting upstream GNU sources to uc386

The trivial in-tree utilities (`true`, `false`, `cat`, `echo`, `head`,
`tail`, `wc`, `yes`, `basename`, `dirname`) demonstrate that uc386 +
the dos_emu runtime can build POSIX-flavored userland from scratch.
Porting real GNU coreutils / gawk source unmodified is the next step
(addons.txt items 1-5).

## Libc gaps for upstream code

Functions GNU code commonly assumes that uc386's libc does **not**
yet provide:

- `getenv` / `setenv` / `putenv` — no environment under dos_emu
- `strtol` / `strtoul` / `strtoll` (only `atoi` exists)
- `strerror` + the `errno` global (libc has no errno mechanism yet)
- `fflush`, `setbuf`, `ungetc`, `feof`, `ferror`
- `fseek` / `ftell` / `rewind` / `clearerr`
- `system` (no shell under dos_emu)
- `getopt` / `getopt_long` (gnulib brings its own; needs porting)
- `unlink` / `rename` / `mkdir` / `rmdir` / `stat` / `access`
- `time` / `gettimeofday` / `clock`
- `realloc` (we have `malloc` / `free` / `calloc`)
- `strdup` / `strndup` / `strcasecmp` / `strncasecmp` / `strpbrk`
- `qsort_r`, `bsearch`
- `regex.h` (BRE/ERE) — needed by `sed`, `grep`, `awk`, `ed`

Functions we **do** have (per `lib/i386_dos_libc.asm`): `puts`,
`putchar`, `printf`, `snprintf`, `fprintf`, `sprintf`, `fputs`,
`fputc`, `fgetc`, `fgets`, `fread`, `fwrite`, `fopen`, `fclose`,
`getchar`, `getc`, `read`, `write`, `open`, `creat`, `link`,
`malloc`, `calloc`, `free`, `memcpy`, `memmove`, `memset`, `memcmp`,
`memchr`, `mempcpy`, `strlen`, `strcpy`, `strncpy`, `strcat`,
`strncat`, `strcmp`, `strncmp`, `strchr`, `strrchr`, `strstr`,
`isalpha`, `isdigit`, `isspace`, `tolower`, `toupper`, `atoi`,
`abs`, `labs`, `llabs`, `signal`, `raise`, `setjmp`, `longjmp`,
`qsort`, `exit`, `abort`, `perror`, `tmpnam`, `remove`, math
(`sin`, `cos`, `sqrt`, `pow`, `floor`, `ceil`, `fabs`).

## Strategy

Two parallel tracks:

1. **Add missing libc** — incrementally extend `lib/i386_dos_libc.asm`
   with shims for the most-needed functions, keeping all 1320 unit
   tests + 220 c-testsuite + 1514 gcc-c-torture green at every step.
   Priority:
   1. `getenv` (returns NULL — empty environment) — unlocks any code
      that reads env vars optionally
   2. `errno` global + `strerror` (returns static "Unknown error")
   3. `strtol` / `strtoul` (full C99 with base, endptr, optional sign)
   4. `fflush` (no-op — we write immediately)
   5. `strdup` (calls `malloc` + `strcpy`)
   6. `unlink` / `rename` (route to dos_emu vfile system)
   7. `getopt` (rolled in C, ~50 lines)
   8. `regex.h` — biggest item. Either port `regcomp`/`regexec`
      from glibc (~3000 lines) or use `re_search` / Spencer's regex.

2. **Pick small upstream targets** — start with single-file utilities
   that have minimal gnulib dependencies:
   - `tee` (~150 lines, only stdio)
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
