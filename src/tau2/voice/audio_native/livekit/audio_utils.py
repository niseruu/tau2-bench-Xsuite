"""Audio format constants for LiveKit cascaded voice pipeline.

The LiveKit cascaded pipeline uses:
- STT Input:  16kHz PCM16 mono (Deepgram expects this)
- TTS Output: Variable rate PCM16 mono (default 24kHz for Deepgram TTS)

The simulation framework uses telephony format:
- 8kHz μ-law mono

Streaming conversion uses ``StreamingTelephonyConverter`` in ``audio_converter.py``.
"""

from tau2.config import DEFAULT_PCM_SAMPLE_RATE, DEFAULT_TELEPHONY_RATE

# Telephony format: 8kHz μ-law, 1 byte per sample
TELEPHONY_SAMPLE_RATE = DEFAULT_TELEPHONY_RATE
TELEPHONY_BYTES_PER_SECOND = DEFAULT_TELEPHONY_RATE  # 1 byte/sample for μ-law

# STT input format (Deepgram): 16kHz PCM16
STT_SAMPLE_RATE = DEFAULT_PCM_SAMPLE_RATE
STT_BYTES_PER_SECOND = STT_SAMPLE_RATE * 2  # 2 bytes/sample for PCM16

# TTS output format: Variable (default 24kHz for Deepgram Aura)
DEFAULT_TTS_SAMPLE_RATE = 24000
