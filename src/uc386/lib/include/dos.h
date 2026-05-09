/* dos.h — minimal stub for DOS-era period code (ROTT, Watcom-era games).
 *
 * dos_emu doesn't simulate real-mode interrupts; the structs and
 * function declarations here let programs parse and link, but the
 * runtime behavior is best-effort no-op or fall-through to libc.
 */
#ifndef _DOS_H
#define _DOS_H

#include <stddef.h>

/* Watcom-era typedefs that period code uses without including
 * a typedef header — the compiler had them as keywords / built-ins.
 * Provide as plain typedefs so source that does `byte b;` compiles. */
typedef unsigned char  uchar;
typedef unsigned short ushort;
typedef unsigned int   uint;

/* Watcom-flavored register pack — three views of the same storage:
 *   .x  full 32-bit (eax/ebx/...)
 *   .w  16-bit halves (ax/bx/...)
 *   .h  8-bit halves (al/ah/bl/bh/...)
 * Period code uses inregs.x.eax / inregs.w.ax / inregs.h.al. */
struct DWORDREGS { unsigned int eax, ebx, ecx, edx, esi, edi, cflag; };
struct WORDREGS  { unsigned short ax, _ax_high, bx, _bx_high, cx, _cx_high,
                                  dx, _dx_high, si, _si_high, di, _di_high,
                                  cflag, flags; };
struct BYTEREGS  { unsigned char al, ah, bl, bh, cl, ch, dl, dh; };
union REGS {
    struct DWORDREGS x;
    struct WORDREGS  w;
    struct BYTEREGS  h;
};

struct SREGS { unsigned short es, cs, ss, ds; };

/* DOS file-attribute bits — used by find_first/find_next. */
#define _A_NORMAL   0x00
#define _A_RDONLY   0x01
#define _A_HIDDEN   0x02
#define _A_SYSTEM   0x04
#define _A_VOLID    0x08
#define _A_SUBDIR   0x10
#define _A_ARCH     0x20

/* find_first / find_next file-info struct. */
struct find_t {
    char     reserved[21];
    char     attrib;
    unsigned short wr_time;
    unsigned short wr_date;
    unsigned long  size;
    char     name[260];
};

int int86(int intno, union REGS *inregs, union REGS *outregs);
int int86x(int intno, union REGS *inregs, union REGS *outregs, struct SREGS *sregs);
int intdos(union REGS *inregs, union REGS *outregs);
int intdosx(union REGS *inregs, union REGS *outregs, struct SREGS *sregs);
void segread(struct SREGS *sregs);

unsigned _dos_findfirst(const char *path, unsigned attrib, struct find_t *buffer);
unsigned _dos_findnext(struct find_t *buffer);

/* Watcom DOS time/date/diskinfo structs — period code samples them
 * via _dos_gettime / _dos_getdate / _dos_getdiskfree. dos_emu doesn't
 * track real time; the runtime impls return zero-filled values. */
struct dostime_t {
    unsigned char hour, minute, second, hsecond;
};
struct dosdate_t {
    unsigned char day, month, dayofweek;
    unsigned short year;
};
struct diskfree_t {
    unsigned short total_clusters;
    unsigned short avail_clusters;
    unsigned short sectors_per_cluster;
    unsigned short bytes_per_sector;
};

unsigned _dos_gettime(struct dostime_t *time);
unsigned _dos_settime(struct dostime_t *time);
unsigned _dos_getdate(struct dosdate_t *date);
unsigned _dos_setdate(struct dosdate_t *date);
unsigned _dos_getdiskfree(unsigned drive, struct diskfree_t *space);

/* Watcom hardware-error handler interface — period code uses
 * _harderr() to install a custom INT 24h handler. dos_emu has no
 * such handler; these are stubs. */
#define _HARDERR_IGNORE  0
#define _HARDERR_RETRY   1
#define _HARDERR_ABORT   2
#define _HARDERR_FAIL    3
/* `far` is a Watcom 16-bit segment qualifier; flat-32 ignores it. */
typedef int (* _HARDERR_HANDLER)(unsigned, unsigned, unsigned *);
void _harderr(_HARDERR_HANDLER handler);
void _hardresume(int action);
void _hardretn(int errcode);

void _enable(void);
void _disable(void);
void delay(unsigned msec);

#endif /* _DOS_H */
