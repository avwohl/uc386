/* i86.h — Watcom-specific x86 intrinsics. Programs use these for
 * port I/O (inp/outp), interrupt control (_enable/_disable), and
 * direct INT calls (int86). dos_emu doesn't simulate real port I/O
 * or hardware interrupts; the runtime impls return 0 / no-op.
 *
 * The struct types are defined in <dos.h> too; we pull them in.
 */
#ifndef _I86_H
#define _I86_H

#include <dos.h>

/* Port I/O. */
unsigned char inp(unsigned port);
unsigned short inpw(unsigned port);
unsigned long inpd(unsigned port);

unsigned char outp(unsigned port, unsigned char value);
unsigned short outpw(unsigned port, unsigned short value);
unsigned long outpd(unsigned port, unsigned long value);

#endif /* _I86_H */
