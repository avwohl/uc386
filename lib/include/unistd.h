/* unistd.h — minimal POSIX I/O declarations for uc386.
 * The companion implementations live in lib/i386_dos_libc.asm
 * (_read, _write, _close, _open, _unlink) and route through
 * dos_emu's INT 21h handlers (and AH=0xA0 POSIX-open handler).
 */
#ifndef _UNISTD_H
#define _UNISTD_H

#include <sys/types.h>

#define STDIN_FILENO   0
#define STDOUT_FILENO  1
#define STDERR_FILENO  2

ssize_t read(int fd, void *buf, size_t count);
ssize_t write(int fd, const void *buf, size_t count);
int close(int fd);
int unlink(const char *path);

/* No process model under dos_emu — these are stubs. */
int isatty(int fd);
char *getcwd(char *buf, size_t size);
int chdir(const char *path);

#endif /* _UNISTD_H */
