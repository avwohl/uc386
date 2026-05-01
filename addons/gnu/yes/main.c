/* `yes` capped at a fixed count.
 *
 * On Unix, `yes` outputs forever and dies on SIGPIPE. dos_emu has
 * no pipe and no SIGPIPE; the instruction-limit guard would just
 * call this a timeout. So we cap at a finite N and exit cleanly —
 * still a useful "loop emits the same byte sequence repeatedly"
 * exercise.
 *
 * argv-driven message + argv-driven count both land in phase 3.
 */
#include <stdio.h>

#define COUNT 1000

int main(void) {
    int i;
    for (i = 0; i < COUNT; i++) {
        putchar('y');
        putchar('\n');
    }
    return 0;
}
