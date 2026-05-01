/* Hand-written stand-in for the autotools-generated config.h that
 * chocolate-doom expects. Pass -I addons/games/heretic/uc386_config/
 * before -I upstream/src so doomtype.h's `#include "config.h"`
 * resolves here.
 *
 * We claim the small set of POSIX features uc386's libc actually
 * provides; everything optional (fluidsynth, libsamplerate, midi,
 * ALSA, SDL_mixer's newer APIs) is left undefined.
 */
#ifndef _UC386_CHOCO_CONFIG_H
#define _UC386_CHOCO_CONFIG_H

/* Project metadata. PACKAGE_TARNAME drives the data-file lookup
 * directory name; we leave it as "heretic" so any DOS data files
 * under that name still resolve. */
#define PACKAGE_NAME    "heretic"
#define PACKAGE_TARNAME "heretic"
#define PACKAGE_VERSION "1.3-uc386"
#define PACKAGE_STRING  "heretic 1.3-uc386"
#define PACKAGE_BUGREPORT "uc386@local"

/* Standard libc decls we have. */
#define HAVE_DECL_STRCASECMP  1
#define HAVE_DECL_STRNCASECMP 1

/* Headers we have stubs / shims for. */
#define HAVE_DIRENT_H   0    /* not yet — would need dos_emu support */
#define HAVE_LIBM       1

/* Optional subsystems we never have. */
#undef HAVE_FLUIDSYNTH
#undef HAVE_LIBSAMPLERATE
#undef HAVE_LIBPNG
#undef HAVE_DEV_ISA_SPKRIO_H

#endif /* _UC386_CHOCO_CONFIG_H */
