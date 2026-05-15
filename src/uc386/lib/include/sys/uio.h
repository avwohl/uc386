/* sys/uio.h — struct iovec + readv/writev shim for the uc386 libc.
 *
 * libssh2 references struct iovec in transport.c for scatter/gather
 * I/O (writev). Our send/recv path is single-buffer through the
 * MP-side socket, but libssh2's code references the struct shape
 * regardless. Provide the typedef so compilation succeeds; readv/
 * writev are declared but not implemented (link error if called).
 */
#ifndef _SYS_UIO_H
#define _SYS_UIO_H

#include <sys/types.h>

struct iovec {
    void   *iov_base;
    size_t  iov_len;
};

long readv(int fd, const struct iovec *iov, int iovcnt);
long writev(int fd, const struct iovec *iov, int iovcnt);

#endif
