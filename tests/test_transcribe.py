"""Tests for mlx_qwen3_asr/transcribe.py."""

from __future__ import annotations

import importlib

import mlx.core as mx
import numpy as np

from mlx_qwen3_asr.forced_aligner import AlignedWord
from mlx_qwen3_asr.transcribe import transcribe


class _DummyModel:
    audio_token_id = 151676

    def audio_tower(self, mel: mx.array, feature_lens: mx.array):
        n = 2
        feats = mx.zeros((1, n, 2048), dtype=mx.float16)
        lens = mx.array([n], dtype=mx.int32)
        return feats, lens


class _DummyTokenizer:
    def __init__(self, model_path: str):
        self.model_path = model_path

    def build_prompt_tokens(self, n_audio_tokens: int, language: str | None = None) -> list[int]:
        return [1, 2, 3]

    def decode(self, ids: list[int]) -> str:
        return "language English<asr_text>hello world"


class _DummyTokenizerHolder:
    @staticmethod
    def get(model_path: str):
        return _DummyTokenizer(model_path)


class _DummyAligner:
    def align(self, audio: np.ndarray, text: str, language: str):
        return [AlignedWord(text="hello", start_time=0.1, end_time=0.4)]


def test_transcribe_basic(monkeypatch):
    tmod = importlib.import_module("mlx_qwen3_asr.transcribe")

    monkeypatch.setattr(tmod, "_TokenizerHolder", _DummyTokenizerHolder)
    monkeypatch.setattr(tmod._ModelHolder, "get", lambda *a, **k: (_DummyModel(), None))
    monkeypatch.setattr(
        tmod,
        "compute_features",
        lambda audio: (mx.zeros((1, 128, 100), dtype=mx.float32), mx.array([100], dtype=mx.int32)),
    )
    monkeypatch.setattr(tmod, "generate", lambda **kwargs: [10, 11, 12])

    result = transcribe(np.zeros(3200, dtype=np.float32))
    assert result.language == "English"
    assert result.text == "hello world"
    assert result.segments is None


def test_transcribe_with_timestamps(monkeypatch):
    tmod = importlib.import_module("mlx_qwen3_asr.transcribe")

    monkeypatch.setattr(tmod, "_TokenizerHolder", _DummyTokenizerHolder)
    monkeypatch.setattr(tmod._ModelHolder, "get", lambda *a, **k: (_DummyModel(), None))
    monkeypatch.setattr(
        tmod,
        "compute_features",
        lambda audio: (mx.zeros((1, 128, 100), dtype=mx.float32), mx.array([100], dtype=mx.int32)),
    )
    monkeypatch.setattr(tmod, "generate", lambda **kwargs: [10, 11, 12])
    monkeypatch.setattr(
        tmod,
        "split_audio_into_chunks",
        lambda audio, sr: [
            (np.zeros(16000, dtype=np.float32), 0.0),
            (np.zeros(16000, dtype=np.float32), 1.5),
        ],
    )
    monkeypatch.setattr(tmod, "parse_asr_output", lambda raw: ("English", "hello world"))

    result = transcribe(
        np.zeros(32000, dtype=np.float32),
        return_timestamps=True,
        forced_aligner=_DummyAligner(),
    )

    assert result.language == "English"
    assert result.text == "hello world hello world"
    assert result.segments == [
        {"text": "hello", "start": 0.1, "end": 0.4},
        {"text": "hello", "start": 1.6, "end": 1.9},
    ]
