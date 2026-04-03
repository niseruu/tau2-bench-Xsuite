"""Audio format constants and tick-size helpers for Amazon Nova Sonic.

Nova Sonic uses different audio formats than the telephony standard:
- Input:  16kHz LPCM mono (vs 8kHz μ-law for telephony)
- Output: 24kHz LPCM mono (vs 8kHz μ-law for telephony)

Note: Output is 24kHz (different from input 16kHz).

Streaming conversion uses ``StreamingTelephonyConverter`` in ``audio_converter.py``.
"""

from tau2.config import (
    DEFAULT_NOVA_INPUT_SAMPLE_RATE,
    DEFAULT_NOVA_OUTPUT_SAMPLE_RATE,
    DEFAULT_TELEPHONY_RATE,
)

# Telephony format: 8kHz μ-law, 1 byte per sample
TELEPHONY_SAMPLE_RATE = DEFAULT_TELEPHONY_RATE
TELEPHONY_BYTES_PER_SECOND = DEFAULT_TELEPHONY_RATE  # 1 byte/sample for μ-law

# Nova Sonic audio formats (from config, PCM16 mono, 2 bytes per sample)
NOVA_INPUT_SAMPLE_RATE = DEFAULT_NOVA_INPUT_SAMPLE_RATE
NOVA_OUTPUT_SAMPLE_RATE = DEFAULT_NOVA_OUTPUT_SAMPLE_RATE


def calculate_telephony_bytes_per_tick(tick_duration_ms: int) -> int:
    """Calculate the expected audio bytes per tick for telephony format.

    Args:
        tick_duration_ms: Duration of each tick in milliseconds.

    Returns:
        Expected number of bytes per tick (8kHz μ-law = 8000 bytes/sec).
    """
    return int(TELEPHONY_BYTES_PER_SECOND * tick_duration_ms / 1000)
