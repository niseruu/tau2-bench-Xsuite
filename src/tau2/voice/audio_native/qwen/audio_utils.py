"""Audio format constants and tick-size helpers for Qwen Omni Flash Realtime API.

Qwen Realtime uses different audio formats than the telephony standard:
- Input:  16kHz PCM16 mono (vs 8kHz μ-law for telephony)
- Output: 24kHz PCM16 mono (vs 8kHz μ-law for telephony)

Note: "pcm24" in Qwen's API refers to 24kHz sample rate, not 24-bit depth.
The actual format is 16-bit signed PCM at 24kHz.

Streaming conversion uses ``StreamingTelephonyConverter`` in ``audio_converter.py``.
"""

from tau2.config import (
    DEFAULT_QWEN_INPUT_SAMPLE_RATE,
    DEFAULT_QWEN_OUTPUT_SAMPLE_RATE,
    DEFAULT_TELEPHONY_RATE,
)

# Telephony format: 8kHz μ-law, 1 byte per sample
TELEPHONY_SAMPLE_RATE = DEFAULT_TELEPHONY_RATE
TELEPHONY_BYTES_PER_SECOND = DEFAULT_TELEPHONY_RATE  # 1 byte/sample for μ-law

# Qwen audio formats (from config, PCM16 mono, 2 bytes per sample)
QWEN_INPUT_SAMPLE_RATE = DEFAULT_QWEN_INPUT_SAMPLE_RATE
QWEN_OUTPUT_SAMPLE_RATE = DEFAULT_QWEN_OUTPUT_SAMPLE_RATE
QWEN_INPUT_BYTES_PER_SECOND = QWEN_INPUT_SAMPLE_RATE * 2
QWEN_OUTPUT_BYTES_PER_SECOND = QWEN_OUTPUT_SAMPLE_RATE * 2


def calculate_qwen_bytes_per_tick(
    tick_duration_ms: int,
    direction: str = "input",
) -> int:
    """Calculate the expected audio bytes per tick for Qwen format.

    Args:
        tick_duration_ms: Duration of each tick in milliseconds.
        direction: "input" for 16kHz or "output" for 24kHz.

    Returns:
        Expected number of bytes per tick.
    """
    if direction == "input":
        bytes_per_second = QWEN_INPUT_BYTES_PER_SECOND
    else:
        bytes_per_second = QWEN_OUTPUT_BYTES_PER_SECOND

    return int(bytes_per_second * tick_duration_ms / 1000)


def calculate_telephony_bytes_per_tick(tick_duration_ms: int) -> int:
    """Calculate the expected audio bytes per tick for telephony format.

    Args:
        tick_duration_ms: Duration of each tick in milliseconds.

    Returns:
        Expected number of bytes per tick.
    """
    return int(TELEPHONY_BYTES_PER_SECOND * tick_duration_ms / 1000)
