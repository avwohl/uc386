/* exe_regs_probe — print the registers PMODE/W passed at entry.
 *
 * Built ONLY by the .exe pipeline (not as a manifest-driven addon)
 * because it depends on `pmodew_*_at_entry` globals that the bridge
 * stub provides — those don't exist under the dos_emu .bin path.
 *
 * Reads the saved-state globals captured at PMODE/W entry by
 * `addons/harness/exe.py`'s `pmodew_start` bridge and prints them
 * via libc → INT 21h. With this we can see EAX/EBX/etc. PMODE/W
 * actually passed and design the real argv bridge against it
 * instead of guessing.
 */
#include <stdio.h>

extern unsigned int pmodew_eax_at_entry;
extern unsigned int pmodew_ebx_at_entry;
extern unsigned int pmodew_ecx_at_entry;
extern unsigned int pmodew_edx_at_entry;
extern unsigned int pmodew_esi_at_entry;
extern unsigned int pmodew_edi_at_entry;
extern unsigned int pmodew_ebp_at_entry;
extern unsigned int pmodew_esp_at_entry;
extern unsigned int pmodew_psp_selector;
extern unsigned int pmodew_psp_linear;
extern unsigned int pmodew_cmdline_len;
extern unsigned int pmodew_argc;
extern unsigned char pmodew_psp_dump[32];
extern unsigned int pmodew_dpmi_alloc_sel;
extern unsigned int pmodew_dpmi_alloc_cy;
extern unsigned int pmodew_int21h_psp;

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
    put_named_reg("eax", pmodew_eax_at_entry);
    put_named_reg("ebx", pmodew_ebx_at_entry);
    put_named_reg("ecx", pmodew_ecx_at_entry);
    put_named_reg("edx", pmodew_edx_at_entry);
    put_named_reg("esi", pmodew_esi_at_entry);
    put_named_reg("edi", pmodew_edi_at_entry);
    put_named_reg("ebp", pmodew_ebp_at_entry);
    put_named_reg("esp", pmodew_esp_at_entry);
    fputs("--DPMI argv parser--\n", stdout);
    put_named_reg("psp_selector", pmodew_psp_selector);
    put_named_reg("psp_linear", pmodew_psp_linear);
    put_named_reg("cmdline_len", pmodew_cmdline_len);
    put_named_reg("parsed_argc", pmodew_argc);
    put_named_reg("dpmi_alloc_sel", pmodew_dpmi_alloc_sel);
    put_named_reg("dpmi_alloc_cy", pmodew_dpmi_alloc_cy);
    put_named_reg("int21h_psp", pmodew_int21h_psp);
    fputs("psp_dump[0x80..0xA0]:", stdout);
    {
        int i;
        for (i = 0; i < 32; i++) {
            putchar(' ');
            puthex8((unsigned int)pmodew_psp_dump[i]);
        }
        putchar('\n');
    }
    return 0;
}
