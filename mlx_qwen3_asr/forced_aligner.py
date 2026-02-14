"""Forced alignment wrapper for word-level timestamps.

This module intentionally uses the official `qwen-asr` forced aligner as a
backend. It provides a thin, stable interface for this project while keeping
the dependency optional.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
import numpy as np

DEFAULT_FORCED_ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"


@dataclass(frozen=True)
class AlignedWord:
    """Word-level alignment item."""

    text: str
    start_time: float
    end_time: float


class ForcedAligner:
    """Word-level forced aligner using the official Qwen backend.

    Args:
        model_path: HF repo ID or local path for the forced aligner model.
        dtype: Reserved for future native-MLX backend compatibility.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_FORCED_ALIGNER_MODEL,
        dtype: mx.Dtype = mx.float16,
    ):
        self.model_path = model_path
        self.dtype = dtype
        self._backend: Optional[Any] = None

    def _ensure_loaded(self) -> None:
        if self._backend is not None:
            return
        try:
            from qwen_asr import Qwen3ForcedAligner  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "Timestamps require optional dependency `qwen-asr`. "
                "Install with: pip install qwen-asr"
            ) from e

        # CPU keeps this portable and avoids silently requiring CUDA.
        self._backend = Qwen3ForcedAligner.from_pretrained(
            self.model_path,
            device_map="cpu",
        )

    def align(
        self,
        audio: np.ndarray,
        text: str,
        language: str,
    ) -> list[AlignedWord]:
        """Align a transcript against audio and return word-level timestamps."""
        self._ensure_loaded()

        if text.strip() == "":
            return []

        # Backend returns a list with one result for single-input calls.
        results = self._backend.align(  # type: ignore[union-attr]
            audio=[(audio.astype(np.float32), 16000)],
            text=[text],
            language=[language],
        )
        if not results:
            return []

        first = results[0]
        items = getattr(first, "items", None)
        if items is None:
            items = first

        out: list[AlignedWord] = []
        for item in items:
            word = getattr(item, "text", "")
            start = float(getattr(item, "start_time", 0.0))
            end = float(getattr(item, "end_time", 0.0))
            out.append(AlignedWord(text=str(word), start_time=start, end_time=end))
        return out
