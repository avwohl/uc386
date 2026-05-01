/* `echo` — print argv[1..] separated by spaces, terminated by '\n'.
 *
 * No flag parsing yet; matches the simplest POSIX echo behavior.
 * Demonstrates argv plumbing through dos_emu → _start → main.
 */
#include <stdio.h>

int main(int argc, char **argv) {
    int i;
    for (i = 1; i < argc; i++) {
        if (i > 1) {
            putchar(' ');
        }
        fputs(argv[i], stdout);
    }
    putchar('\n');
    return 0;
}
