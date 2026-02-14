"""High-level transcription pipeline for Qwen3-ASR."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import mlx.core as mx
import numpy as np

from .audio import SAMPLE_RATE, compute_features, load_audio
from .chunking import split_audio_into_chunks
from .generate import GenerationConfig, generate
from .load_models import _ModelHolder
from .model import Qwen3ASRModel
from .tokenizer import Tokenizer, parse_asr_output


@dataclass(frozen=True)
class TranscriptionResult:
    """Result of audio transcription.

    Attributes:
        text: The transcribed text
        language: Detected or forced language
        segments: Optional list of segments with timestamps [{text, start, end}, ...]
    """
    text: str
    language: str
    segments: Optional[list[dict]] = None


def transcribe(
    audio: Union[str, Path, np.ndarray, mx.array, tuple[np.ndarray, int]],
    *,
    model: Union[str, Qwen3ASRModel] = "Qwen/Qwen3-ASR-1.7B",
    language: Optional[str] = None,
    return_timestamps: bool = False,
    forced_aligner: Optional[str] = None,
    dtype: mx.Dtype = mx.float16,
    max_new_tokens: int = 1024,
    verbose: bool = False,
) -> TranscriptionResult:
    """Transcribe audio to text using Qwen3-ASR.

    Pipeline:
    1. Load audio -> mono 16kHz
    2. Split into chunks if > 1200s
    3. Per chunk: mel spectrogram -> encode -> build prompt -> decode -> parse
    4. Merge chunks
    5. Optional: forced aligner for word-level timestamps

    Args:
        audio: Audio source - file path, numpy array, mx.array, or (array, sr) tuple
        model: Model name/path or pre-loaded model instance
        language: Force language detection (e.g., "English", "Chinese")
        return_timestamps: Whether to return word-level timestamps.
            Not yet implemented; raises NotImplementedError when True.
        forced_aligner: Path/name of forced aligner model (reserved for future use)
        dtype: Model dtype
        max_new_tokens: Maximum tokens to generate per chunk
        verbose: Print progress information

    Returns:
        TranscriptionResult with text, language, and optional segments
    """
    if return_timestamps:
        raise NotImplementedError(
            "return_timestamps is not available yet: forced alignment is still WIP."
        )

    # Load model
    if isinstance(model, str):
        model_obj, _ = _ModelHolder.get(model, dtype=dtype)
        model_path = model
    else:
        model_obj = model
        model_path = "Qwen/Qwen3-ASR-1.7B"

    # Load tokenizer
    tokenizer = Tokenizer(model_path)

    # Load audio
    if isinstance(audio, mx.array):
        audio_np = np.array(audio)
    elif isinstance(audio, np.ndarray):
        audio_np = audio.astype(np.float32)
    elif isinstance(audio, tuple):
        audio_np = np.array(load_audio(audio))
    else:
        audio_np = np.array(load_audio(audio))

    if verbose:
        duration = len(audio_np) / SAMPLE_RATE
        print(f"Audio duration: {duration:.1f}s ({len(audio_np)} samples)")

    # Split into chunks if needed
    chunks = split_audio_into_chunks(audio_np, sr=SAMPLE_RATE)

    if verbose and len(chunks) > 1:
        print(f"Split into {len(chunks)} chunks")

    # Transcribe each chunk
    all_texts = []
    detected_language = language or "unknown"

    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=0.0,
    )

    for chunk_idx, (chunk_audio, offset) in enumerate(chunks):
        if verbose:
            print(f"Processing chunk {chunk_idx + 1}/{len(chunks)} "
                  f"(offset={offset:.1f}s, duration={len(chunk_audio)/SAMPLE_RATE:.1f}s)")

        # Compute mel spectrogram using HF WhisperFeatureExtractor
        mel, feature_lens = compute_features(chunk_audio)  # (1, 128, 3000), (1,)

        # Encode audio (encoder handles padding mask and trims to valid tokens)
        audio_features, output_lens = model_obj.audio_tower(
            mel.astype(dtype), feature_lens
        )  # (1, n_valid_tokens, dim)
        n_audio_tokens = audio_features.shape[1]

        if verbose:
            print(f"  Audio features: {n_audio_tokens} tokens")

        # Build prompt
        prompt_tokens = tokenizer.build_prompt_tokens(
            n_audio_tokens=n_audio_tokens,
            language=language,
        )
        input_ids = mx.array([prompt_tokens])

        # Build position_ids for MRoPE (3 spatial dims, all same for text)
        seq_len = input_ids.shape[1]
        positions = mx.arange(seq_len)[None, :]  # (1, seq_len)
        position_ids = mx.stack([positions, positions, positions], axis=1)  # (1, 3, seq_len)

        # Generate
        output_tokens = generate(
            model=model_obj,
            input_ids=input_ids,
            audio_features=audio_features,
            position_ids=position_ids,
            config=gen_config,
        )

        # Decode tokens to text
        raw_text = tokenizer.decode(output_tokens)
        lang, text = parse_asr_output(raw_text)

        if detected_language == "unknown":
            detected_language = lang

        all_texts.append(text)

        if verbose:
            print(f"  [{lang}] {text[:100]}{'...' if len(text) > 100 else ''}")

    # Merge chunks
    full_text = " ".join(all_texts)

    return TranscriptionResult(
        text=full_text,
        language=detected_language,
        segments=None,
    )
