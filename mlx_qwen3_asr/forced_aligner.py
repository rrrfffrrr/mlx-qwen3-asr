"""Forced alignment wrapper for word-level timestamps.

This module intentionally uses the official `qwen-asr` forced aligner as a
backend. It provides a thin, stable interface for this project while keeping
the dependency optional.
"""

from __future__ import annotations

import unicodedata
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


class ForcedAlignTextProcessor:
    """Text processor utilities for timestamp alignment input/output."""

    @staticmethod
    def is_kept_char(ch: str) -> bool:
        if ch == "'":
            return True
        cat = unicodedata.category(ch)
        return cat.startswith("L") or cat.startswith("N")

    @classmethod
    def clean_token(cls, token: str) -> str:
        return "".join(ch for ch in token if cls.is_kept_char(ch))

    @staticmethod
    def is_cjk_char(ch: str) -> bool:
        code = ord(ch)
        return (
            0x4E00 <= code <= 0x9FFF
            or 0x3400 <= code <= 0x4DBF
            or 0x20000 <= code <= 0x2A6DF
            or 0x2A700 <= code <= 0x2B73F
            or 0x2B740 <= code <= 0x2B81F
            or 0x2B820 <= code <= 0x2CEAF
            or 0xF900 <= code <= 0xFAFF
        )

    @classmethod
    def split_segment_with_cjk(cls, seg: str) -> list[str]:
        tokens: list[str] = []
        buf: list[str] = []

        def flush() -> None:
            nonlocal buf
            if buf:
                tokens.append("".join(buf))
                buf = []

        for ch in seg:
            if cls.is_cjk_char(ch):
                flush()
                tokens.append(ch)
            else:
                buf.append(ch)

        flush()
        return tokens

    @classmethod
    def tokenize_space_lang(cls, text: str) -> list[str]:
        tokens: list[str] = []
        for seg in text.split():
            cleaned = cls.clean_token(seg)
            if cleaned:
                tokens.extend(cls.split_segment_with_cjk(cleaned))
        return tokens

    @classmethod
    def tokenize_text(cls, text: str, language: str) -> list[str]:
        """Tokenize alignment text into word/character units.

        Current default path matches official behavior for space-delimited
        languages and mixed CJK text. Japanese/Korean specialized tokenizers
        are intentionally deferred to keep core dependencies minimal.
        """
        _ = language
        return cls.tokenize_space_lang(text)

    @classmethod
    def encode_timestamp_prompt(cls, text: str, language: str) -> tuple[list[str], str]:
        words = cls.tokenize_text(text, language)
        input_text = "<timestamp><timestamp>".join(words) + "<timestamp><timestamp>"
        input_text = "<|audio_start|><|audio_pad|><|audio_end|>" + input_text
        return words, input_text

    @staticmethod
    def fix_timestamp(data: np.ndarray) -> list[int]:
        """Repair non-monotonic timestamp sequence via LIS-based correction."""
        arr = data.tolist()
        n = len(arr)
        if n == 0:
            return []

        dp = [1] * n
        parent = [-1] * n

        for i in range(1, n):
            for j in range(i):
                if arr[j] <= arr[i] and dp[j] + 1 > dp[i]:
                    dp[i] = dp[j] + 1
                    parent[i] = j

        max_len = max(dp)
        idx = dp.index(max_len)

        lis_indices: list[int] = []
        while idx != -1:
            lis_indices.append(idx)
            idx = parent[idx]
        lis_indices.reverse()

        is_normal = [False] * n
        for i in lis_indices:
            is_normal[i] = True

        out = arr.copy()
        i = 0
        while i < n:
            if is_normal[i]:
                i += 1
                continue

            j = i
            while j < n and not is_normal[j]:
                j += 1

            anomaly_count = j - i
            left_val = next((out[k] for k in range(i - 1, -1, -1) if is_normal[k]), None)
            right_val = next((out[k] for k in range(j, n) if is_normal[k]), None)

            if anomaly_count <= 2:
                for k in range(i, j):
                    if left_val is None:
                        out[k] = right_val
                    elif right_val is None:
                        out[k] = left_val
                    else:
                        out[k] = left_val if (k - (i - 1)) <= (j - k) else right_val
            else:
                if left_val is not None and right_val is not None:
                    step = (right_val - left_val) / (anomaly_count + 1)
                    for k in range(i, j):
                        out[k] = left_val + step * (k - i + 1)
                elif left_val is not None:
                    for k in range(i, j):
                        out[k] = left_val
                elif right_val is not None:
                    for k in range(i, j):
                        out[k] = right_val

            i = j

        return [int(v) for v in out]

    @classmethod
    def parse_timestamp_ms(cls, words: list[str], timestamp_ms: np.ndarray) -> list[AlignedWord]:
        fixed = cls.fix_timestamp(np.asarray(timestamp_ms))
        out: list[AlignedWord] = []
        for i, word in enumerate(words):
            start_i = i * 2
            end_i = start_i + 1
            if end_i >= len(fixed):
                break
            out.append(
                AlignedWord(
                    text=word,
                    start_time=round(float(fixed[start_i]) / 1000.0, 3),
                    end_time=round(float(fixed[end_i]) / 1000.0, 3),
                )
            )
        return out


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
