/* argv_probe — dump argc + each argv[i] using only putchar/fputs.
 *
 * Diagnostic for Phase 7 (the .exe argv bridge): under PMODE/W,
 * `echo hello dos` produces "exe hello dos", meaning argv[1..] are
 * "exe", "hello", "dos" and argc=4. We don't yet know what argv[0]
 * holds — could be the program path, the basename, "" or NULL. This
 * probe prints argc and argv[0..argc-1] explicitly.
 *
 * Why no printf: our libc's `_printf` routes through INT 21h AH=0x5F
 * which is a dos_emu harness intercept, not a real DOS call. Under
 * PMODE/W → real DOS that's a no-op, so printf output disappears.
 * `_putchar` / `_fputs` use AH=02h (display char) which IS real DOS,
 * so they work under both runners.
 */
#include <stdio.h>

/* Print v as an unsigned decimal. Uses recursion instead of an
 * `if-zero-return; while-positive` shape — that pattern triggers
 * a uc386 codegen bug where the while emits jle with no cmp,
 * relying on stale flags from the prior if. */
static void putdec_u(unsigned int v) {
    if (v >= 10) {
        putdec_u(v / 10);
    }
    putchar('0' + (v % 10));
}

static void putdec(int v) {
    if (v < 0) {
        putchar('-');
        putdec_u((unsigned int)(-v));
    } else {
        putdec_u((unsigned int)v);
    }
}

static void puthex8(unsigned int v) {
    static const char hex[] = "0123456789abcdef";
    int i;
    for (i = 28; i >= 0; i -= 4) {
        putchar(hex[(v >> i) & 0xF]);
    }
}

int main(int argc, char **argv) {
    int i;
    fputs("argc=", stdout);
    putdec(argc);
    putchar('\n');
    for (i = 0; i < argc; i++) {
        fputs("argv[", stdout);
        putdec(i);
        fputs("]@", stdout);
        puthex8((unsigned int)(unsigned long)argv[i]);
        if (argv[i] == 0) {
            fputs(":NULL\n", stdout);
        } else {
            fputs(":'", stdout);
            fputs(argv[i], stdout);
            fputs("'\n", stdout);
        }
    }
    return 0;
}
