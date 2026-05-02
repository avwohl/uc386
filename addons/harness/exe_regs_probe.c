/* exe_regs_probe — print the registers PMODE/W passed at entry.
 *
 * Built ONLY by the .exe pipeline (not as a manifest-driven addon)
 * because it depends on `_pmodew_*_at_entry` globals that the bridge
 * stub provides — those don't exist under the dos_emu .bin path.
 *
 * Reads the saved-state globals captured at PMODE/W entry by
 * `addons/harness/exe.py`'s `_pmodew_start` bridge and prints them
 * via libc → INT 21h. With this we can see EAX/EBX/etc. PMODE/W
 * actually passed and design the real argv bridge against it
 * instead of guessing.
 */
#include <stdio.h>

extern unsigned int _pmodew_eax_at_entry;
extern unsigned int _pmodew_ebx_at_entry;
extern unsigned int _pmodew_ecx_at_entry;
extern unsigned int _pmodew_edx_at_entry;
extern unsigned int _pmodew_esi_at_entry;
extern unsigned int _pmodew_edi_at_entry;
extern unsigned int _pmodew_ebp_at_entry;
extern unsigned int _pmodew_esp_at_entry;

static void puthex8(unsigned int v) {
    static const char hex[] = "0123456789abcdef";
    int i;
    for (i = 28; i >= 0; i -= 4) {
        putchar(hex[(v >> i) & 0xF]);
    }
}

static void put_named_reg(const char *name, unsigned int v) {
    fputs(name, stdout);
    fputs("=0x", stdout);
    puthex8(v);
    putchar('\n');
}

int main(void) {
    fputs("--regs at PMODE/W entry--\n", stdout);
    put_named_reg("eax", _pmodew_eax_at_entry);
    put_named_reg("ebx", _pmodew_ebx_at_entry);
    put_named_reg("ecx", _pmodew_ecx_at_entry);
    put_named_reg("edx", _pmodew_edx_at_entry);
    put_named_reg("esi", _pmodew_esi_at_entry);
    put_named_reg("edi", _pmodew_edi_at_entry);
    put_named_reg("ebp", _pmodew_ebp_at_entry);
    put_named_reg("esp", _pmodew_esp_at_entry);
    return 0;
}
