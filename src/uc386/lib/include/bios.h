/* bios.h — BIOS-call stubs for period code. dos_emu doesn't expose
 * real BIOS functions (no INT 16h / INT 10h / INT 1Ah handling beyond
 * what the libc shims need); these declarations are here so programs
 * including <bios.h> link cleanly. The runtime impls return 0.
 */
#ifndef _BIOS_H
#define _BIOS_H

/* INT 16h keyboard subfunctions. */
#define _KEYBRD_READ    0x00
#define _KEYBRD_READY   0x01
#define _NKEYBRD_READ   0x10
#define _NKEYBRD_READY  0x11

/* INT 1Ah / system timer subfunctions. */
unsigned _bios_keybrd(unsigned cmd);
unsigned long _bios_timeofday(unsigned cmd, long *timep);

/* INT 10h video. */
unsigned _bios_setvideomode(unsigned mode);

#endif /* _BIOS_H */
