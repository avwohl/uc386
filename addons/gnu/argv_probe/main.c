/* argv_probe — dump argc + each argv[i] address & contents.
 *
 * Diagnostic for Phase 7 (the .exe argv bridge): under PMODE/W,
 * `echo hello dos` produces "exe hello dos", meaning argv[1..] are
 * "exe", "hello", "dos" and argc=4. We don't yet know what argv[0]
 * holds — could be the program path, the basename, "" or NULL. This
 * probe prints argc and argv[0..argc-1] explicitly so we can read it
 * out of CI's RESULT.TXT.
 *
 * Under dos_emu (the .bin runner) the answer should look totally
 * normal: argc and each argv string match what we passed in. The
 * interesting run is .exe under DOSBox.
 */
#include <stdio.h>

int main(int argc, char **argv) {
    int i;
    printf("argc=%d\n", argc);
    for (i = 0; i < argc; i++) {
        if (argv[i] == 0) {
            printf("argv[%d]=NULL\n", i);
        } else {
            printf("argv[%d]@%p='%s'\n", i, (void *)argv[i], argv[i]);
        }
    }
    return 0;
}
