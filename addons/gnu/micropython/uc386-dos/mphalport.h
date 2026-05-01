// uc386-dos MicroPython HAL header.
//
// `mp_hal_stdin_rx_chr` / `mp_hal_stdout_tx_strn` are out-of-line in
// `upstream/ports/minimal/uart_core.c`; this header just provides the
// trivial inlines. ticks_ms returns 0 (no realtime needed for a REPL
// port; a real implementation would call INT 1Ah BIOS time).

static inline mp_uint_t mp_hal_ticks_ms(void) {
    return 0;
}

static inline void mp_hal_set_interrupt_char(char c) {
    (void)c;
}
