/* setjmp.h - Non-local jumps for uc386.
 *
 * jmp_buf must be wide enough for the platform's _setjmp implementation.
 * For i386 the runtime in lib/i386_dos_libc.asm saves 6 dwords (ebx, esi,
 * edi, saved ebp, caller esp, return eip), so jmp_buf is 6 * 4 = 24 bytes.
 *
 * (The earlier `unsigned char[6]` declaration was a leftover from the
 * Z80-era uc80 frontend and would buffer-overflow when uc386's _setjmp
 * wrote 24 bytes into a 6-byte slot — broke MicroPython's setjmp-based
 * NLR.)
 */
#ifndef _SETJMP_H
#define _SETJMP_H

typedef unsigned long jmp_buf[6];

extern int setjmp(jmp_buf env);
extern void longjmp(jmp_buf env, int val);

#endif /* _SETJMP_H */
