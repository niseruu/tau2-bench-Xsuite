"""Discrete-time audio native adapter for Deepgram Voice Agent API.

This adapter provides a tick-based interface for Deepgram Voice Agent API, designed
for discrete-time simulation where audio time is the primary clock.

Note: Deepgram Voice Agent is a CASCADED system (STT → LLM → TTS), unlike native
audio models (OpenAI Realtime, Gemini Live, Nova Sonic) that process audio directly.

Key features:
- Tick-based interface via run_tick()
- Audio format conversion (telephony ↔ Deepgram formats)
- Audio capping: max bytes_per_tick of agent audio per tick
- Audio buffering: excess agent audio carries to next tick
- Proportional transcript: text distributed based on audio played
- Interruption handling via UserStartedSpeaking events

Usage:
    adapter = DiscreteTimeDeepgramAdapter(
        tick_duration_ms=1000,
        send_audio_instant=True,
    )
    adapter.connect(system_prompt, tools, vad_config, modality="audio")

    for tick in range(max_ticks):
        result = adapter.run_tick(user_audio_bytes, tick_number=tick)
        # result.get_played_agent_audio() - capped agent audio (telephony format)
        # result.proportional_transcript - text for this tick
        # result.tool_calls - function calls

    adapter.disconnect()
"""

import asyncio
import base64
import json
import uuid
from typing import Any, Dict, List, Optional

from loguru import logger

from tau2.config import (
    DEFAULT_AUDIO_NATIVE_CONNECT_TIMEOUT,
    DEFAULT_AUDIO_NATIVE_DISCONNECT_TIMEOUT,
    DEFAULT_AUDIO_NATIVE_TICK_TIMEOUT_BUFFER,
)
from tau2.data_model.message import ToolCall
from tau2.environment.tool import Tool
from tau2.voice.audio_native.adapter import DiscreteTimeAdapter
from tau2.voice.audio_native.async_loop import BackgroundAsyncLoop
from tau2.voice.audio_native.audio_converter import StreamingTelephonyConverter
from tau2.voice.audio_native.deepgram.audio_utils import (
    DEEPGRAM_OUTPUT_BYTES_PER_SECOND,
    calculate_deepgram_bytes_per_tick,
)
from tau2.voice.audio_native.deepgram.events import (
    DeepgramAgentAudioDoneEvent,
    DeepgramAgentStartedSpeakingEvent,
    DeepgramAudioEvent,
    DeepgramConversationTextEvent,
    DeepgramErrorEvent,
    DeepgramFunctionCallRequestEvent,
    DeepgramTimeoutEvent,
    DeepgramUserStartedSpeakingEvent,
)
from tau2.voice.audio_native.deepgram.provider import (
    DEEPGRAM_INPUT_BYTES_PER_SECOND,
    DeepgramVADConfig,
    DeepgramVoiceAgentProvider,
)
from tau2.voice.audio_native.tick_result import (
    TickResult,
    UtteranceTranscript,
)


class DiscreteTimeDeepgramAdapter(DiscreteTimeAdapter):
    """Adapter for discrete-time full-duplex simulation with Deepgram Voice Agent API.

    Implements DiscreteTimeAdapter for Deepgram Voice Agent.

    This adapter runs an async event loop in a background thread to communicate
    with the Deepgram Voice Agent API, while exposing a synchronous interface for
    the agent and orchestrator.

    Audio format handling:
    - Input: Receives telephony audio (8kHz μ-law), converts to 16kHz PCM16
    - Output: Receives 16kHz PCM16 from Deepgram, converts to 8kHz μ-law

    Attributes:
        tick_duration_ms: Duration of each tick in milliseconds.
        bytes_per_tick: Audio bytes per tick in telephony format (8kHz μ-law).
        send_audio_instant: If True, send audio in one call per tick.
            If False, send in 20ms chunks with sleeps (VoIP-style streaming).
        provider: Optional provider instance. Created lazily if not provided.
    """

    def __init__(
        self,
        tick_duration_ms: int,
        send_audio_instant: bool = True,
        provider: Optional[DeepgramVoiceAgentProvider] = None,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        tts_model: Optional[str] = None,
    ):
        """Initialize the discrete-time Deepgram adapter.

        Args:
            tick_duration_ms: Duration of each tick in milliseconds. Must be > 0.
            send_audio_instant: If True, send audio in one call (discrete-time mode).
            provider: Optional provider instance. Created lazily if not provided.
            llm_provider: LLM provider (e.g., "open_ai", "anthropic").
            llm_model: LLM model (e.g., "gpt-4o-mini").
            tts_model: TTS model including voice (e.g., "aura-2-thalia-en").
        """
        super().__init__(tick_duration_ms, send_audio_instant=send_audio_instant)

        self._chunk_size = int(
            DEEPGRAM_INPUT_BYTES_PER_SECOND * self._voip_interval_ms / 1000
        )
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.tts_model = tts_model

        # Deepgram output format (16kHz PCM16) - for internal processing
        self._deepgram_output_bytes_per_tick = calculate_deepgram_bytes_per_tick(
            tick_duration_ms, direction="output"
        )

        # Provider - created lazily if not provided
        self._provider = provider
        self._owns_provider = provider is None

        # Audio format converter (preserves state for streaming)
        self._audio_converter = StreamingTelephonyConverter(
            input_sample_rate=16000,
            output_sample_rate=16000,
        )

        # Async event loop management
        self._bg_loop = BackgroundAsyncLoop()
        self._connected = False

        # Deepgram-specific: maps call_id -> function_name
        self._tool_call_info: Dict[str, str] = {}

    @property
    def provider(self) -> DeepgramVoiceAgentProvider:
        """Get the provider, creating it if needed."""
        if self._provider is None:
            self._provider = DeepgramVoiceAgentProvider(
                llm_provider=self.llm_provider,
                llm_model=self.llm_model,
                tts_model=self.tts_model,
            )
        return self._provider

    @property
    def is_connected(self) -> bool:
        """Check if connected to the API."""
        return self._connected and self._bg_loop.is_running

    def connect(
        self,
        system_prompt: str,
        tools: List[Tool],
        vad_config: Any = None,
        modality: str = "audio",
    ) -> None:
        """Connect to the Deepgram Voice Agent API and configure the session.

        Args:
            system_prompt: System prompt for the agent.
            tools: List of tools the agent can use.
            vad_config: VAD configuration. Defaults to automatic VAD.
            modality: "audio" or "text" (Deepgram always uses audio).
        """
        if self._connected:
            logger.warning("Already connected, disconnecting first")
            self.disconnect()

        # Default VAD config
        if vad_config is None:
            vad_config = DeepgramVADConfig()

        self._bg_loop.start()

        try:
            self._bg_loop.run_coroutine(
                self._async_connect(system_prompt, tools, vad_config, modality),
                timeout=DEFAULT_AUDIO_NATIVE_CONNECT_TIMEOUT,
            )
            self._connected = True
            logger.info(
                f"DiscreteTimeDeepgramAdapter connected to Deepgram Voice Agent API "
                f"(tick={self.tick_duration_ms}ms, bytes_per_tick={self.bytes_per_tick})"
            )
        except Exception as e:
            logger.error(
                f"DiscreteTimeDeepgramAdapter failed to connect to Deepgram Voice Agent API: "
                f"{type(e).__name__}: {e}"
            )
            self._bg_loop.stop()
            raise RuntimeError(
                f"Failed to connect to Deepgram Voice Agent API: {e}"
            ) from e

    async def _async_connect(
        self,
        system_prompt: str,
        tools: List[Tool],
        vad_config: DeepgramVADConfig,
        modality: str,
    ) -> None:
        """Async connection and configuration."""
        await self.provider.connect()
        await self.provider.configure_session(
            system_prompt=system_prompt,
            tools=tools,
            vad_config=vad_config,
        )

    def disconnect(self) -> None:
        """Disconnect from the API and clean up resources."""
        if not self._connected:
            return

        if self._bg_loop.is_running:
            try:
                self._bg_loop.run_coroutine(
                    self._async_disconnect(),
                    timeout=DEFAULT_AUDIO_NATIVE_DISCONNECT_TIMEOUT,
                )
            except Exception as e:
                logger.warning(f"Error during disconnect: {e}")

        self._bg_loop.stop()
        self._connected = False
        self._tick_count = 0
        self._cumulative_user_audio_ms = 0
        self.clear_buffers()
        self._tool_call_info.clear()
        self._audio_converter.reset()
        logger.info("DiscreteTimeDeepgramAdapter disconnected")

    async def _async_disconnect(self) -> None:
        """Async disconnection."""
        if self._owns_provider and self._provider is not None:
            await self.provider.disconnect()

    def run_tick(
        self, user_audio: bytes, tick_number: Optional[int] = None
    ) -> TickResult:
        """Run one tick of the simulation.

        Args:
            user_audio: User audio bytes in telephony format (8kHz μ-law).
            tick_number: Optional tick number for logging.

        Returns:
            TickResult with audio in telephony format (8kHz μ-law).
        """
        if not self.is_connected:
            raise RuntimeError(
                "Not connected to Deepgram Voice Agent API. Call connect() first."
            )

        if tick_number is None:
            tick_number = self._tick_count
        self._tick_count = tick_number + 1

        try:
            return self._bg_loop.run_coroutine(
                self._async_run_tick(user_audio, tick_number),
                timeout=self.tick_duration_ms / 1000
                + DEFAULT_AUDIO_NATIVE_TICK_TIMEOUT_BUFFER,
            )
        except Exception as e:
            logger.error(f"Error in run_tick (tick={tick_number}): {e}")
            raise

    def send_tool_result(
        self,
        call_id: str,
        result: str,
        request_response: bool = True,
        is_error: bool = False,
    ) -> None:
        """Queue a tool result, resolving the Deepgram function name first."""
        name = self._tool_call_info.pop(call_id, "unknown")
        super().send_tool_result(call_id, result, request_response, is_error)
        # Re-patch the last entry to include the function name for flush
        # Base class stores (call_id, result, request_response, is_error)
        # We need (call_id, name, result, request_response) for Deepgram
        self._pending_tool_results[-1] = (call_id, name, result, request_response)

    async def _flush_pending_tool_results(self) -> None:
        """Send pending tool results to Deepgram."""
        for call_id, name, result_str, _request_response in self._pending_tool_results:
            await self.provider.send_tool_result(
                call_id=call_id,
                function_name=name,
                result=result_str,
            )
        self._pending_tool_results.clear()

    async def _execute_tick(
        self,
        user_audio: bytes,
        tick_number: int,
        result: TickResult,
        tick_start: float,
    ) -> None:
        """Deepgram-specific: convert audio, send, receive events, process."""
        # Convert user audio from telephony to Deepgram format
        deepgram_audio = self._audio_converter.convert_input(user_audio)

        deepgram_audio_received: list[tuple[bytes, Optional[str]]] = []

        async def receive_events():
            elapsed_so_far = asyncio.get_running_loop().time() - tick_start
            remaining = max(0.01, (self.tick_duration_ms / 1000) - elapsed_so_far)
            return await self.provider.receive_events_for_duration(remaining)

        _, events = await asyncio.gather(
            self._send_audio_chunked(
                deepgram_audio, self.provider.send_audio, self._chunk_size
            ),
            receive_events(),
        )

        for event in events:
            self._process_event(result, event, deepgram_audio_received)

        # Convert Deepgram audio to telephony format and add to result
        for deepgram_bytes, item_id in deepgram_audio_received:
            telephony_bytes = self._audio_converter.convert_output(deepgram_bytes)
            result.agent_audio_chunks.append((telephony_bytes, item_id))

        if deepgram_audio_received:
            total_deepgram = sum(len(d) for d, _ in deepgram_audio_received)
            total_telephony = sum(len(d) for d, _ in result.agent_audio_chunks)
            logger.debug(
                f"Audio conversion: {len(deepgram_audio_received)} chunks, "
                f"{total_deepgram} deepgram bytes -> {total_telephony} telephony bytes"
            )

    def _process_event(
        self,
        result: TickResult,
        event: Any,
        deepgram_audio_received: list[tuple[bytes, Optional[str]]],
    ) -> None:
        """Process a Deepgram event."""
        result.events.append(event)

        if isinstance(event, DeepgramAudioEvent):
            audio_data = base64.b64decode(event.audio) if event.audio else b""
            logger.debug(
                f"DeepgramAudioEvent: {len(audio_data)} bytes, "
                f"item_id={self._current_item_id}"
            )
            if audio_data:
                item_id = self._current_item_id or str(uuid.uuid4())[:8]

                # Skip audio from truncated item
                if result.skip_item_id is not None and item_id == result.skip_item_id:
                    estimated_telephony_bytes = int(
                        len(audio_data)
                        * self.audio_format.bytes_per_second
                        / DEEPGRAM_OUTPUT_BYTES_PER_SECOND
                    )
                    result.truncated_audio_bytes += estimated_telephony_bytes
                else:
                    deepgram_audio_received.append((audio_data, item_id))

                    # Track for transcript distribution
                    if item_id not in self._utterance_transcripts:
                        self._utterance_transcripts[item_id] = UtteranceTranscript(
                            item_id=item_id
                        )
                    estimated_telephony_bytes = int(
                        len(audio_data)
                        * self.audio_format.bytes_per_second
                        / DEEPGRAM_OUTPUT_BYTES_PER_SECOND
                    )
                    self._utterance_transcripts[item_id].add_audio(
                        estimated_telephony_bytes
                    )

        elif isinstance(event, DeepgramConversationTextEvent):
            if event.role == "assistant" and event.content:
                item_id = self._current_item_id or str(uuid.uuid4())[:8]
                if item_id not in self._utterance_transcripts:
                    self._utterance_transcripts[item_id] = UtteranceTranscript(
                        item_id=item_id
                    )
                self._utterance_transcripts[item_id].add_transcript(event.content)
                self._current_item_id = item_id

        elif isinstance(event, DeepgramUserStartedSpeakingEvent):
            logger.debug("User started speaking - interruption detected")
            result.vad_events.append("user_started_speaking")
            # Clear buffered audio
            if self._buffered_agent_audio:
                buffered_bytes = sum(len(c[0]) for c in self._buffered_agent_audio)
                result.truncated_audio_bytes += buffered_bytes
                self._buffered_agent_audio.clear()

            # Mark truncation
            result.was_truncated = True
            result.skip_item_id = self._current_item_id

            # Clear current item_id so next response gets a new one
            self._current_item_id = None

            # Reset audio converter state after interruption
            self._audio_converter.reset()

        elif isinstance(event, DeepgramAgentStartedSpeakingEvent):
            self._current_item_id = str(uuid.uuid4())[:8]
            logger.debug(f"Agent started speaking (item_id={self._current_item_id})")

        elif isinstance(event, DeepgramAgentAudioDoneEvent):
            logger.debug("Agent audio done")
            self._skip_item_id = None
            result.skip_item_id = None

        elif isinstance(event, DeepgramFunctionCallRequestEvent):
            for func in event.functions:
                self._tool_call_info[func.id] = func.name

                try:
                    arguments = json.loads(func.arguments) if func.arguments else {}
                except json.JSONDecodeError:
                    arguments = {}

                tool_call = ToolCall(
                    id=func.id,
                    name=func.name,
                    arguments=arguments,
                )
                result.tool_calls.append(tool_call)
                logger.debug(f"Tool call detected: {func.name}({func.id})")

        elif isinstance(event, DeepgramErrorEvent):
            logger.error(
                f"Deepgram error: {event.error_message} (code={event.error_code})"
            )

        elif isinstance(event, DeepgramTimeoutEvent):
            pass

        else:
            logger.debug(f"Event {type(event).__name__} received")
