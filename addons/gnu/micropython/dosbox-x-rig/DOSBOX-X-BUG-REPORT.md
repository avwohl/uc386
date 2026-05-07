# Draft DOSBox-X bug report

Draft for upstream DOSBox-X
(https://github.com/joncampbell123/dosbox-x). Root-caused
2026-05-07; candidate fix below tested under instrumented build
in CI.

---

## Title

`-silent` + `[autoexec]`-driven `program > out.txt / exit`:
`tinyfd_messageBox` in headless mode loops on stdin, blocks
DOSBox-X exit until SIGKILL → buffered `fwrite` output for the
last writes is lost.

## Affected versions

- DOSBox-X `2026.05.02` SDL2 (Homebrew bottle, macOS Darwin
  25.4.0)
- `apt install dosbox-x` on Ubuntu 22.04 GitHub Actions runner
  (same `2026.05.02` family)

Both reproduce identically.

## Symptom

A DOS program that writes to a redirected file and then
crashes (or never returns) loses its trailing output. Concretely,
a 484 KB MicroPython binary built with PMODE/W writes 657 bytes
of expected output to `MPOUT.TXT` over 46 `INT 21h AH=0x40`
calls. Under DOSBox-X with the trigger config (below), only the
first 481 bytes — covering the first 37 calls — appear in the
final `MPOUT.TXT`. The remaining 176 bytes (the next 9 calls
including the MicroPython REPL banner) are silently lost.

## Reliable reproducer

```ini
# dosbox-x.conf
[cpu]
cputype = pentium
core    = normal
cycles  = max
[dosbox]
memsize = 16

[autoexec]
mount C /path/with/mp.exe
C:
MP.EXE > MPOUT.TXT
exit
```

```sh
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
    timeout -k 10 60 dosbox-x -silent -conf dosbox-x.conf
```

`MP.EXE` is the uc386-MicroPython binary at
https://github.com/avwohl/uc386 (artifact `mp-rig-artifacts/MP.EXE`
from any successful `mp-rig.yml` run). Direct CI demo at
[`dosbox-x-bug.yml`](https://github.com/avwohl/uc386/actions/workflows/dosbox-x-bug.yml).

`MPOUT.TXT` ends up exactly 481 bytes ending at
`[bridge-pre-call-main]\n`. Same MP.EXE under classic DOSBox
0.74-3 or QEMU+FreeDOS produces 657 bytes ending at the
MicroPython REPL prompt `>>> `.

## Root cause (CONFIRMED)

Two layered behaviors compound:

### 1. `LocalFile::Write` is buffered

[`src/dos/drive_local.cpp:2860`](https://github.com/joncampbell123/dosbox-x/blob/master/src/dos/drive_local.cpp#L2860)
calls
`*size = (uint16_t)fwrite(data, 1, *size, fhandle); return true;`

`fwrite` writes into the host C library's stdio buffer.
Bytes only hit the underlying file when:

- the buffer fills (default 4 KB / 8 KB on Linux glibc), OR
- `fflush` is called (DOSBox-X invokes this only via
  `LocalFile::Flush`, called from
  [`DOS_FlushFile`](https://github.com/joncampbell123/dosbox-x/blob/master/src/dos/dos_files.cpp)
  which is the `INT 21h AH=0x68` "commit file" handler), OR
- the file is `fclose`d.

So every `INT 21h AH=0x40 BX=fd EDX=buf ECX=count` call by a
DOS program WRITES TO BUFFER but NOT TO DISK unless followed by
`AH=0x68`, the buffer fills, or the file closes. This is
strictly different behavior from real DOS (which has no
host-stdio layer) and from classic DOSBox 0.74-3 (which uses
unbuffered `write(2)` at this layer).

In our reproducer, MP.EXE's bridge stub does AH=40 + AH=68
after every diagnostic marker print, so those writes flush
correctly. But MP.EXE's `_main → _write` (a normal libc-style
`write(2)` wrapper) calls only AH=40, so post-marker bytes stay
in the stdio buffer.

### 2. `-silent` + running-program quit hangs forever, then SIGKILL

When the autoexec runs `exit` after MP.EXE, DOSBox-X tries to
shut down. `RunningProgram` is still set to MP.EXE's name (the
program crashed via invalid-opcode and DoCommand returned, but
`RunningProgram` is reset by COMMAND only on certain exit paths).
[`CheckQuit`](https://github.com/joncampbell123/dosbox-x/blob/master/src/gui/sdlmain.cpp#L1377)
calls
`systemmessagebox("Quit DOSBox-X warning", QUIT_PROGRAM_CONFIRM, "yesno", "question", 1)`.

Under `-silent` with `SDL_VIDEODRIVER=dummy`, there is no GUI
target, so `systemmessagebox` falls through to
`tinyfd_messageBox`'s console path
([`tinyfiledialogs.c:2950`](https://github.com/joncampbell123/dosbox-x/blob/master/src/libs/tinyfiledialogs/tinyfiledialogs.c#L2950)):

```c
do {
    ...
    printf("y/n: ");
    lChar = (char)tolower(_getch());
    printf("\n\n");
} while (lChar != 'y' && lChar != 'n');
```

In headless mode `_getch()` returns immediately on EOF or
returns the same char repeatedly, never 'y' or 'n', so the loop
spins forever. We observed thousands of repetitions of
`y/n: \n\n` in stderr before `timeout -k 10 60` fired SIGKILL
on DOSBox-X.

SIGKILL skips all atexit handlers and host-process shutdown
flushing, so the in-buffer 176 bytes from the unflushed `_write`
calls are discarded by the kernel.

## Suggested fix

Two-part patch (verified end-to-end via instrumented CI build at
[`dosbox-x-instrument.yml`](https://github.com/avwohl/uc386/actions/workflows/dosbox-x-instrument.yml);
the "silent + fflush" run produces the full 657 bytes).

**Patch 1 — skip the quit confirmation in `-silent` mode**
(`src/gui/sdlmain.cpp` near the start of `CheckQuit()`):

```c
bool CheckQuit(void) {
#if !defined(HX_DOS)
    /* In -silent mode there is no UI to answer the y/n
     * confirmation. tinyfd_messageBox falls back to
     * printf+_getch which loops forever waiting for stdin.
     * Auto-confirm so DOSBox-X exits cleanly (closing all open
     * files and flushing their stdio buffers in the process). */
    if (control && control->opt_silent) return true;
    /* ... rest of existing implementation ... */
```

This alone fixes the symptom for our reproducer, because the
clean shutdown path closes all `Files[handle]` instances, which
calls `fclose(fhandle)`, which flushes the stdio buffer.

**Patch 2 — `fflush` after every `LocalFile::Write`**
(`src/dos/drive_local.cpp` non-Windows branch at the end of the
function):

```c
*size = file_access_tries>0
    ? (uint16_t)write(fileno(fhandle),data,*size)
    : (uint16_t)fwrite(data,1,*size,fhandle);
fflush(fhandle); /* So stdio buffers can't be lost on SIGKILL. */
return true;
```

Patch 2 adds a per-write fflush, matching what classic DOSBox
0.74-3 effectively does by using unbuffered `write(2)`. This
guarantees that even programs killed by SIGKILL (e.g., DOSBox-X
crashes, host OOM, user-initiated kill) have their writes
visible up to the last call. Performance impact is negligible
for redirected-output workloads (one extra syscall per AH=40,
vs. zero extra under buffer-fill amortization — but
buffer-fill is uncommon for typical DOS-program output bursts).

Either patch alone fixes the symptom; both together make the
fix robust to other future SIGKILL paths.

## Diagnostic infrastructure

CI workflows in
https://github.com/avwohl/uc386 reproduce and characterize the
bug end-to-end:

- [`dosbox-x-bug.yml`](https://github.com/avwohl/uc386/blob/main/.github/workflows/dosbox-x-bug.yml):
  runs MP.EXE under apt-installed DOSBox-X / classic DOSBox
  0.74-3 / QEMU+FreeDOS, captures output sizes + xxd dumps,
  shows the 481-vs-657 byte split.
- [`dosbox-x-instrument.yml`](https://github.com/avwohl/uc386/blob/main/.github/workflows/dosbox-x-instrument.yml):
  patches `src/dos/dos.cpp` to log every `INT 21h AH=0x40` call's
  full register state, source linear address, first 16 bytes of
  `dos_copybuf`, the resolved `Files[handle]->name`, and
  pre/post `*amount` from `DOS_WriteFile`. Builds DOSBox-X from
  source and runs MP.EXE. The captured log shows ALL 46 AH=40
  calls succeed (`fWritten=1`, `post_towrite==pre_towrite`),
  with identical buffer contents in failing and working configs
  — proving the bug is downstream of `DOS_WriteFile` (in the
  stdio buffering and the SIGKILL exit path).

## Why this matters

This bug blocks a multi-MB DOS application (the uc386
MicroPython port) from completing its CI pipeline under
DOSBox-X — even though the program runs cleanly under FreeDOS
(the actual production target — FreeDOS in VMware on Windows),
classic DOSBox 0.74-3, and on real DOS hardware.

More broadly: the bug affects any DOS program that:

1. Writes to a redirected file via `INT 21h AH=0x40`
2. Doesn't follow with `INT 21h AH=0x68` (most C-libc-style
   `write` wrappers don't)
3. Is invoked via DOSBox-X's `[autoexec]` section
4. The autoexec's next command is `exit`
5. DOSBox-X is run with `-silent` (typical for CI / scripted
   testing)

Under those conditions, the program's last hundred bytes of
output are silently truncated, with no error indication — a
significant correctness hazard for any test rig that expects
DOS-program stdout to be a faithful capture.
