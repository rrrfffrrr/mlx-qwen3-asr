"""Experimental rolling streaming ASR with prefix rollback.

This module currently re-transcribes accumulated audio context to improve
stability of partial output. It is not a true incremental decoder with KV
cache reuse across chunks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import DEFAULT_MODEL_ID
from .model import Qwen3ASRModel

# Streaming constants (from official repo)
UNFIXED_CHUNK_NUM = 2     # Trailing chunks considered unfixed
UNFIXED_TOKEN_NUM = 5     # Trailing tokens considered unfixed


@dataclass
class StreamingState:
    """State for streaming ASR session.

    Attributes:
        buffer: Pending audio samples not yet processed
        audio_accum: All accumulated audio so far
        text: Current best transcription
        language: Detected language
        chunk_id: Number of chunks processed
        unfixed_chunk_num: Number of initial chunks to keep fully unstable
        unfixed_token_num: Number of trailing units to keep unstable
        chunk_size_samples: Samples per chunk
        max_context_samples: Max samples used for rolling decode window
        stable_text: Text considered stable (won't change)
    """
    buffer: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    audio_accum: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    text: str = ""
    language: str = "unknown"
    chunk_id: int = 0
    unfixed_chunk_num: int = UNFIXED_CHUNK_NUM
    unfixed_token_num: int = UNFIXED_TOKEN_NUM
    chunk_size_samples: int = 32000  # 2 seconds at 16kHz
    max_context_samples: int = 480000  # 30 seconds at 16kHz
    stable_text: str = ""
    _model_path: str = DEFAULT_MODEL_ID


def init_streaming(
    model: str = DEFAULT_MODEL_ID,
    unfixed_chunk_num: int = UNFIXED_CHUNK_NUM,
    unfixed_token_num: int = UNFIXED_TOKEN_NUM,
    chunk_size_sec: float = 2.0,
    max_context_sec: float = 30.0,
    sample_rate: int = 16000,
) -> StreamingState:
    """Initialize a streaming ASR session.

    Args:
        model: Model name or path
        unfixed_chunk_num: Number of initial chunks to keep fully unstable
        unfixed_token_num: Number of trailing units kept unstable afterwards
        chunk_size_sec: Audio chunk size in seconds
        max_context_sec: Max rolling decode context in seconds
        sample_rate: Audio sample rate

    Returns:
        Initial streaming state
    """
    if chunk_size_sec <= 0:
        raise ValueError(f"chunk_size_sec must be > 0, got: {chunk_size_sec}")
    if max_context_sec <= 0:
        raise ValueError(f"max_context_sec must be > 0, got: {max_context_sec}")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be > 0, got: {sample_rate}")
    if max_context_sec < chunk_size_sec:
        raise ValueError(
            f"max_context_sec must be >= chunk_size_sec, got: "
            f"{max_context_sec} < {chunk_size_sec}"
        )

    return StreamingState(
        unfixed_chunk_num=int(unfixed_chunk_num),
        unfixed_token_num=int(unfixed_token_num),
        chunk_size_samples=int(chunk_size_sec * sample_rate),
        max_context_samples=int(max_context_sec * sample_rate),
        _model_path=model,
    )


def feed_audio(
    pcm: np.ndarray,
    state: StreamingState,
    model: Optional[Qwen3ASRModel] = None,
) -> StreamingState:
    """Feed audio chunk to rolling streaming ASR.

    Each chunk of audio:
    1. Accumulate in buffer
    2. When buffer >= chunk_size, process
    3. Re-transcribe accumulated audio context
    4. Compare with previous transcription
    5. Apply prefix rollback for stability

    Args:
        pcm: Audio samples as float32 numpy array
        state: Current streaming state
        model: Pre-loaded model (if None, loads from state._model_path)

    Returns:
        Updated streaming state with new transcription
    """
    from .transcribe import transcribe

    if pcm is None:
        raise ValueError("pcm must not be None")

    x = _sanitize_stream_pcm(pcm)
    if x.size == 0:
        return state

    # Accumulate audio
    state.buffer = np.concatenate([state.buffer, x])
    state.audio_accum = np.concatenate([state.audio_accum, x])

    # Check if we have enough audio for a new chunk
    if len(state.buffer) < state.chunk_size_samples:
        return state

    # Retain leftover samples beyond chunk_size for next iteration
    leftover = state.buffer[state.chunk_size_samples:]

    # Process: transcribe all accumulated audio
    decode_audio = state.audio_accum
    if len(decode_audio) > state.max_context_samples:
        decode_audio = decode_audio[-state.max_context_samples:]

    result = transcribe(
        audio=decode_audio,
        model=state._model_path if model is None else model,
        verbose=False,
    )

    new_text = result.text
    new_language = result.language

    # Apply prefix rollback
    if state.chunk_id < state.unfixed_chunk_num:
        stable, unstable = state.stable_text, new_text
    else:
        stable, unstable = _split_stable_unstable(
            state.stable_text,
            new_text,
            unfixed_tokens=state.unfixed_token_num,
        )

    state.buffer = leftover
    state.text = new_text
    state.language = new_language
    state.chunk_id += 1
    state.stable_text = stable
    _ = unstable  # Reserved for future APIs exposing unstable tails.
    return state


def finish_streaming(
    state: StreamingState,
    model: Optional[Qwen3ASRModel] = None,
) -> StreamingState:
    """Finalize streaming session, processing any remaining audio.

    Args:
        state: Current streaming state
        model: Pre-loaded model

    Returns:
        Final streaming state
    """
    if len(state.audio_accum) == 0:
        return state

    from .transcribe import transcribe

    # Final transcription of all audio
    result = transcribe(
        audio=state.audio_accum,
        model=state._model_path if model is None else model,
        verbose=False,
    )

    state.buffer = np.array([], dtype=np.float32)
    state.text = result.text
    state.language = result.language
    state.stable_text = result.text
    return state


def _sanitize_stream_pcm(pcm: np.ndarray) -> np.ndarray:
    """Normalize streaming PCM to mono float32 waveform."""
    x = np.asarray(pcm)
    if x.ndim == 0:
        raise ValueError("pcm must be 1-D or 2-D audio array")
    if x.ndim > 2:
        raise ValueError(f"pcm must be 1-D or 2-D, got shape {x.shape}")

    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        x = x.astype(np.float32)
        if info.min >= 0:
            midpoint = (info.max + 1) / 2.0
            x = (x - midpoint) / midpoint
        else:
            x = x / float(max(abs(info.min), info.max))
    else:
        x = x.astype(np.float32, copy=False)

    if x.ndim == 2:
        n0, n1 = int(x.shape[0]), int(x.shape[1])
        if n0 <= 8 and n1 <= 8:
            if n0 == n1:
                channel_axis = 1
            else:
                channel_axis = 0 if n0 < n1 else 1
        elif n0 <= 8:
            channel_axis = 0
        elif n1 <= 8:
            channel_axis = 1
        else:
            channel_axis = 1
        x = x.mean(axis=channel_axis)

    return np.asarray(x, dtype=np.float32)


def _split_text_units(text: str) -> tuple[list[str], str]:
    """Split text into rollback units and return the join delimiter.

    For whitespace-delimited languages, units are words and the delimiter is
    a single space. For languages without spaces (CJK), units are Unicode
    codepoints and the delimiter is empty.
    """
    if any(ch.isspace() for ch in text):
        return text.split(), " "
    return list(text), ""


def _split_stable_unstable(
    prev_stable: str,
    new_text: str,
    unfixed_tokens: int = UNFIXED_TOKEN_NUM,
) -> tuple[str, str]:
    """Split transcription into stable and unstable parts.

    The last `unfixed_tokens` words are considered unstable and may change
    with future audio input.

    Args:
        prev_stable: Previously stable text
        new_text: New full transcription
        unfixed_tokens: Number of trailing tokens considered unstable

    Returns:
        Tuple of (stable_text, unstable_text)
    """
    units, joiner = _split_text_units(new_text)

    if len(units) <= unfixed_tokens:
        return prev_stable, new_text

    stable_units = units[:-unfixed_tokens]
    unstable_units = units[-unfixed_tokens:]

    stable = joiner.join(stable_units)
    unstable = joiner.join(unstable_units)

    # Ensure new stable text is at least as long as previous
    if len(stable) < len(prev_stable):
        stable = prev_stable

    return stable, unstable
