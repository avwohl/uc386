/* io.h — Watcom/Borland low-level I/O. Most period code that
 * `#include <io.h>` uses open / read / write / close, plus filelength
 * and lseek. Forward to the POSIX-flavored declarations in
 * <unistd.h> + <fcntl.h>; add the few Watcom-specific bits here.
 */
#ifndef _IO_H
#define _IO_H

#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>  /* S_IREAD / S_IWRITE / S_IEXEC */

/* Watcom-specific helpers: file size by fd / by name. */
long filelength(int fd);
int  setmode(int fd, int mode);
int  eof(int fd);

/* DOS file modes (passed to open with O_BINARY|O_RDONLY etc). */
#define O_BINARY 0x8000
#define O_TEXT   0x4000

/* Open mode bits — Watcom uses S_I* from sys/stat.h, but old code
 * sometimes references these. */
#define O_NDELAY 0
#define O_NOCTTY 0

#endif /* _IO_H */
