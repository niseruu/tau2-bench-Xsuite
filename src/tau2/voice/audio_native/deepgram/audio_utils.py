"""Audio format constants and tick-size helpers for Deepgram Voice Agent API.

Deepgram Voice Agent uses different audio formats than the telephony standard:
- Input:  16kHz PCM16 mono (vs 8kHz μ-law for telephony)
- Output: 16kHz PCM16 mono (vs 8kHz μ-law for telephony)

Note: Unlike Gemini (which uses 24kHz output), Deepgram uses 16kHz for both
input and output, simplifying the conversion logic.

Streaming conversion uses ``StreamingTelephonyConverter`` in ``audio_converter.py``.
"""

from tau2.config import (
    DEFAULT_DEEPGRAM_INPUT_SAMPLE_RATE,
    DEFAULT_DEEPGRAM_OUTPUT_SAMPLE_RATE,
    DEFAULT_TELEPHONY_RATE,
)
from tau2.data_model.audio import AudioEncoding, AudioFormat

# Telephony format: 8kHz μ-law, 1 byte per sample
TELEPHONY_SAMPLE_RATE = DEFAULT_TELEPHONY_RATE
TELEPHONY_BYTES_PER_SECOND = DEFAULT_TELEPHONY_RATE  # 1 byte/sample for μ-law

# Deepgram audio format constants (from config)
DEEPGRAM_INPUT_SAMPLE_RATE = DEFAULT_DEEPGRAM_INPUT_SAMPLE_RATE
DEEPGRAM_OUTPUT_SAMPLE_RATE = DEFAULT_DEEPGRAM_OUTPUT_SAMPLE_RATE
DEEPGRAM_INPUT_BYTES_PER_SECOND = DEEPGRAM_INPUT_SAMPLE_RATE * 2
DEEPGRAM_OUTPUT_BYTES_PER_SECOND = DEEPGRAM_OUTPUT_SAMPLE_RATE * 2

# Deepgram audio formats (both input and output are 16kHz PCM16)
DEEPGRAM_INPUT_FORMAT = AudioFormat(
    encoding=AudioEncoding.PCM_S16LE,
    sample_rate=DEEPGRAM_INPUT_SAMPLE_RATE,
    channels=1,
)

DEEPGRAM_OUTPUT_FORMAT = AudioFormat(
    encoding=AudioEncoding.PCM_S16LE,
    sample_rate=DEEPGRAM_OUTPUT_SAMPLE_RATE,
    channels=1,
)

TELEPHONY_FORMAT = AudioFormat(
    encoding=AudioEncoding.ULAW,
    sample_rate=TELEPHONY_SAMPLE_RATE,
    channels=1,
)


def calculate_deepgram_bytes_per_tick(
    tick_duration_ms: int,
    direction: str = "input",
) -> int:
    """Calculate the expected audio bytes per tick for Deepgram format.

    Args:
        tick_duration_ms: Duration of each tick in milliseconds.
        direction: "input" or "output" (both are 16kHz for Deepgram).

    Returns:
        Expected number of bytes per tick.
    """
    if direction == "input":
        bytes_per_second = DEEPGRAM_INPUT_BYTES_PER_SECOND
    else:
        bytes_per_second = DEEPGRAM_OUTPUT_BYTES_PER_SECOND

    return int(bytes_per_second * tick_duration_ms / 1000)


def calculate_telephony_bytes_per_tick(tick_duration_ms: int) -> int:
    """Calculate the expected audio bytes per tick for telephony format.

    Args:
        tick_duration_ms: Duration of each tick in milliseconds.

    Returns:
        Expected number of bytes per tick.
    """
    return int(TELEPHONY_BYTES_PER_SECOND * tick_duration_ms / 1000)
