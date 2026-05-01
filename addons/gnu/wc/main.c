/* `wc` over stdin: count lines, words, characters.
 *
 * Matches POSIX wc default output for stdin (no leading filename).
 * Word boundary uses isspace(); a "word" is a maximal non-space run.
 */
#include <stdio.h>
#include <ctype.h>

int main(void) {
    long lines = 0;
    long words = 0;
    long chars = 0;
    int in_word = 0;
    int c;
    while ((c = getchar()) != EOF) {
        chars++;
        if (c == '\n') {
            lines++;
        }
        if (isspace(c)) {
            in_word = 0;
        } else if (!in_word) {
            in_word = 1;
            words++;
        }
    }
    printf("%ld %ld %ld\n", lines, words, chars);
    return 0;
}
