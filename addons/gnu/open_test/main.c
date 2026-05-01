/* Smoke test for the new _open + AH=0xA0 path in dos_emu.
 *
 * Opens a vfile for write (O_WRONLY|O_CREAT|O_TRUNC), writes data,
 * closes, reopens for read, reads back, prints. Then re-opens with
 * O_APPEND and adds more bytes; final read shows both writes.
 */
#include <stdio.h>
#include <fcntl.h>
#include <string.h>

extern int read(int fd, void *buf, unsigned int n);
extern int write(int fd, const void *buf, unsigned int n);
extern int close(int fd);

int main(void) {
    char buf[64];
    int fd;
    int n;

    /* Truncating write */
    fd = open("test.dat", O_WRONLY | O_CREAT | O_TRUNC, 0666);
    if (fd < 0) {
        puts("FAIL: open trunc");
        return 1;
    }
    write(fd, "first ", 6);
    close(fd);

    /* Appending write */
    fd = open("test.dat", O_WRONLY | O_CREAT | O_APPEND, 0666);
    if (fd < 0) {
        puts("FAIL: open append");
        return 1;
    }
    write(fd, "second\n", 7);
    close(fd);

    /* Read back */
    fd = open("test.dat", O_RDONLY, 0);
    if (fd < 0) {
        puts("FAIL: open read");
        return 1;
    }
    n = read(fd, buf, 63);
    if (n < 0) {
        puts("FAIL: read");
        return 1;
    }
    buf[n] = 0;
    fputs("got: ", stdout);
    fputs(buf, stdout);
    close(fd);
    return 0;
}
