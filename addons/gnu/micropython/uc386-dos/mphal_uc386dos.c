// uc386-dos MicroPython HAL — out-of-line implementations.
//
// The minimal port's mphalport.h normally inlines mp_hal_ticks_ms()
// to a constant 0 because the minimal port has no realtime source.
// We override that with a real BIOS-tick-backed version so the
// `time` module can return non-trivial ticks_ms / sleep behavior.
//
// All timing is derived from INT 1Ah AH=00h (BIOS tick counter,
// ~18.2 Hz, ~55 ms per tick). That's the only realtime source DOS
// programs can portably read without programming the PIT directly,
// and it matches what most DOS C runtimes use under the hood.

#include "py/mpconfig.h"
#include "py/mphal.h"

extern unsigned long bios_ticks(void);

// Each BIOS tick is 1000 / 18.20649 ≈ 54.9254 ms. Use 55 — the ~0.13%
// drift is well below what user code that reaches for ticks_ms can
// distinguish, and matches the rounding most DJGPP/Watcom DOS time
// helpers do.
#define BIOS_TICK_MS 55U

mp_uint_t mp_hal_ticks_ms(void) {
    return bios_ticks() * BIOS_TICK_MS;
}

mp_uint_t mp_hal_ticks_us(void) {
    // Single-tick resolution. Multiply by 1000 (giving us 55000 µs
    // step) so callers comparing two ticks_us readings see a
    // monotonically-non-decreasing sequence with predictable steps.
    return bios_ticks() * (BIOS_TICK_MS * 1000U);
}

mp_uint_t mp_hal_ticks_cpu(void) {
    // No RDTSC or PIT-direct reads here; reuse the µs scale so the
    // value is at least monotonically non-decreasing.
    return mp_hal_ticks_us();
}

void mp_hal_delay_ms(mp_uint_t ms) {
    // Convert ms → ticks (round up), then busy-wait until the BIOS
    // counter has advanced that many ticks. Polls in a tight loop;
    // there's no HLT/yield equivalent we can portably issue under
    // PMODE/W since the extender owns interrupt dispatch.
    if (ms == 0) {
        return;
    }
    unsigned long ticks_to_wait = (ms + BIOS_TICK_MS - 1) / BIOS_TICK_MS;
    if (ticks_to_wait == 0) {
        ticks_to_wait = 1;
    }
    unsigned long start = bios_ticks();
    while ((bios_ticks() - start) < ticks_to_wait) {
        // poll
    }
}

void mp_hal_delay_us(mp_uint_t us) {
    // Sub-tick precision isn't available, so we round up to the
    // nearest ms and reuse the ms path. Anything < 55000 µs ends up
    // waiting one BIOS tick, anything >= 55000 µs waits the
    // appropriate number.
    if (us == 0) {
        return;
    }
    unsigned long ms = (us + 999U) / 1000U;
    mp_hal_delay_ms(ms);
}
