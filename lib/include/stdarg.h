/* stdarg.h - Variable argument list handling for uc386 (i386 / cdecl) */
#ifndef _STDARG_H
#define _STDARG_H

/*
 * cdecl on i386: arguments pushed right-to-left, callee accesses them
 * via [ebp + 8 + offset]. va_list is just a pointer to the current
 * argument position; uc386 has builtin codegen for va_start, va_arg,
 * and va_end that consult this pointer directly. The user must declare
 * `va_list` themselves via this typedef; the actual lowering happens
 * in codegen.py.
 */

typedef char *va_list;

#define va_copy(dest, src) ((dest) = (src))

#endif /* _STDARG_H */
