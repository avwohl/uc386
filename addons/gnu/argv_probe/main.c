/* argv_probe — dump argc + each argv[i] using only putchar/fputs.
 *
 * Diagnostic for Phase 7 (the .exe argv bridge): under PMODE/W,
 * argv_probe.exe shows argc=768 + every argv[i] empty — the dos_emu
 * register-passing convention (EAX=argc, EBX=&argv) doesn't match
 * what PMODE/W puts in EAX/EBX at entry. The companion register
 * probe `addons/harness/exe_regs_probe.c` (CI-only) reveals the
 * actual entry contract.
 *
 * Under dos_emu (.bin) the answer is normal: argc + argv strings
 * match what the manifest passes.
 */
#include <stdio.h>

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
    for (i = 0; i < argc && i < 10; i++) {
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
