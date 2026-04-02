"""Audio format constants and tick-size helpers for Gemini Live API.

Gemini Live uses different audio formats than the telephony standard:
- Input:  16kHz PCM16 mono (vs 8kHz μ-law for telephony)
- Output: 24kHz PCM16 mono (vs 8kHz μ-law for telephony)

Streaming conversion uses ``StreamingTelephonyConverter`` in ``audio_converter.py``.
"""

from tau2.config import (
    DEFAULT_GEMINI_INPUT_SAMPLE_RATE,
    DEFAULT_GEMINI_OUTPUT_SAMPLE_RATE,
    DEFAULT_TELEPHONY_RATE,
)
from tau2.data_model.audio import AudioEncoding, AudioFormat

# Telephony format: 8kHz μ-law, 1 byte per sample
TELEPHONY_SAMPLE_RATE = DEFAULT_TELEPHONY_RATE
TELEPHONY_BYTES_PER_SECOND = DEFAULT_TELEPHONY_RATE  # 1 byte/sample for μ-law

# Gemini audio format constants (from config)
GEMINI_INPUT_SAMPLE_RATE = DEFAULT_GEMINI_INPUT_SAMPLE_RATE
GEMINI_OUTPUT_SAMPLE_RATE = DEFAULT_GEMINI_OUTPUT_SAMPLE_RATE
GEMINI_INPUT_BYTES_PER_SECOND = GEMINI_INPUT_SAMPLE_RATE * 2  # 16-bit = 2 bytes
GEMINI_OUTPUT_BYTES_PER_SECOND = GEMINI_OUTPUT_SAMPLE_RATE * 2

# Gemini audio formats
GEMINI_INPUT_FORMAT = AudioFormat(
    encoding=AudioEncoding.PCM_S16LE,
    sample_rate=GEMINI_INPUT_SAMPLE_RATE,
    channels=1,
)

GEMINI_OUTPUT_FORMAT = AudioFormat(
    encoding=AudioEncoding.PCM_S16LE,
    sample_rate=GEMINI_OUTPUT_SAMPLE_RATE,
    channels=1,
)

TELEPHONY_FORMAT = AudioFormat(
    encoding=AudioEncoding.ULAW,
    sample_rate=TELEPHONY_SAMPLE_RATE,
    channels=1,
)


def calculate_gemini_bytes_per_tick(
    tick_duration_ms: int,
    direction: str = "input",
) -> int:
    """Calculate the expected audio bytes per tick for Gemini format.

    Args:
        tick_duration_ms: Duration of each tick in milliseconds.
        direction: "input" for 16kHz or "output" for 24kHz.

    Returns:
        Expected number of bytes per tick.
    """
    if direction == "input":
        bytes_per_second = GEMINI_INPUT_BYTES_PER_SECOND
    else:
        bytes_per_second = GEMINI_OUTPUT_BYTES_PER_SECOND

    return int(bytes_per_second * tick_duration_ms / 1000)


def calculate_telephony_bytes_per_tick(tick_duration_ms: int) -> int:
    """Calculate the expected audio bytes per tick for telephony format.

    Args:
        tick_duration_ms: Duration of each tick in milliseconds.

    Returns:
        Expected number of bytes per tick.
    """
    return int(TELEPHONY_BYTES_PER_SECOND * tick_duration_ms / 1000)
