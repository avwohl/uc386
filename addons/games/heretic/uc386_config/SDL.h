/* SDL.h — minimal shim for chocolate-doom under uc386.
 *
 * Real SDL2 is way out of scope; we only need enough for the headers
 * and source-as-data references to parse cleanly. Functions / structs
 * that are actually CALLED would need stubs in a doom_stubs.c
 * companion file; for now this gets us through the parse stage.
 */
#ifndef _UC386_SDL_H
#define _UC386_SDL_H

#include <stdint.h>
#include <stddef.h>

/* SDL types — opaque or shaped enough that pointer parameters parse.
 * doomtype.h includes this; downstream files use SDL_Event * etc. as
 * function-parameter types. */

typedef int32_t SDL_Keycode;
typedef int32_t SDL_Scancode;

typedef union SDL_Event {
    uint32_t type;
    /* Real SDL_Event is a tagged union of >50 specific event structs.
     * Anything that touches event.{key,motion,...} fields will need
     * a fuller layout — pull it in incrementally as port work
     * demands. */
    uint8_t  padding[56];
} SDL_Event;

/* Common keycodes period code references (subset). */
#define SDLK_UNKNOWN     0
#define SDLK_RETURN      '\r'
#define SDLK_ESCAPE      '\x1b'
#define SDLK_BACKSPACE   '\b'
#define SDLK_TAB         '\t'
#define SDLK_SPACE       ' '

/* Boolean / status. */
typedef int SDL_bool;
#define SDL_FALSE 0
#define SDL_TRUE  1

/* Init flags + bare API surface. */
#define SDL_INIT_VIDEO     0x00000020
#define SDL_INIT_AUDIO     0x00000010
#define SDL_INIT_JOYSTICK  0x00000200

int SDL_Init(uint32_t flags);
void SDL_Quit(void);
const char *SDL_GetError(void);
uint32_t SDL_GetTicks(void);
void SDL_Delay(uint32_t ms);

/* Polling — programs that depend on the actual event stream get zero
 * events from this stub; that's fine for headless boot. */
int SDL_PollEvent(SDL_Event *event);
int SDL_WaitEvent(SDL_Event *event);

/* Opaque pointer types — chocolate-doom code holds these as
 * (typically extern) pointers and never dereferences them when
 * we're running headless. */
typedef struct SDL_Window SDL_Window;
typedef struct SDL_Renderer SDL_Renderer;
typedef struct SDL_Texture SDL_Texture;
typedef struct SDL_Surface SDL_Surface;
typedef struct SDL_RWops SDL_RWops;
typedef struct SDL_Thread SDL_Thread;
typedef struct SDL_mutex SDL_mutex;
typedef struct SDL_cond SDL_cond;

/* Color / Rect / Point — period code keeps them as values. */
typedef struct SDL_Color { uint8_t r, g, b, a; } SDL_Color;
typedef struct SDL_Rect  { int x, y, w, h; } SDL_Rect;
typedef struct SDL_Point { int x, y; } SDL_Point;

#endif /* _UC386_SDL_H */

