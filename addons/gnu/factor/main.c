/* `factor` — print the prime factorization of each argument.
 *
 * Behaves like POSIX/GNU factor for small numbers (≤ 32-bit). Trial
 * division up to sqrt(n) — fine for command-line use, slow on big
 * primes. Reads numbers from argv only; reading from stdin (`factor`
 * with no args) lands later.
 *
 * Demonstrates strtoul + 32-bit arithmetic in argv-driven code.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void factor_one(unsigned long n) {
    printf("%lu:", n);
    if (n < 2) {
        putchar('\n');
        return;
    }
    while ((n & 1UL) == 0UL) {
        printf(" 2");
        n >>= 1;
    }
    unsigned long d = 3;
    while (d <= n / d) {
        while (n % d == 0UL) {
            printf(" %lu", d);
            n /= d;
        }
        d += 2;
    }
    if (n > 1UL) {
        printf(" %lu", n);
    }
    putchar('\n');
}

int main(int argc, char **argv) {
    int i;
    if (argc < 2) {
        fputs("factor: missing operand\n", stderr);
        return 1;
    }
    for (i = 1; i < argc; i++) {
        unsigned long n = strtoul(argv[i], NULL, 10);
        factor_one(n);
    }
    return 0;
}
