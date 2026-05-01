/* values.h — legacy SVR4/Linux header for MAXINT/MININT/etc.
 *
 * Long since superseded by <limits.h> in standard C, but old C codebases
 * (including original DOOM linuxdoom-1.10) still include it. We just
 * forward to the modern names.
 */
#ifndef _VALUES_H
#define _VALUES_H

#include <limits.h>
#include <float.h>

#define BITSPERBYTE  CHAR_BIT
#define BITS(type)   (CHAR_BIT * (int)sizeof(type))

#define MAXSHORT     SHRT_MAX
#define MAXINT       INT_MAX
#define MAXLONG      LONG_MAX

#define MINSHORT     SHRT_MIN
#define MININT       INT_MIN
#define MINLONG      LONG_MIN

#define MAXFLOAT     FLT_MAX
#define MAXDOUBLE    DBL_MAX
#define MINFLOAT     FLT_MIN
#define MINDOUBLE    DBL_MIN

#endif /* _VALUES_H */
