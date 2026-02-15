"""Tests for mlx_qwen3_asr/transcribe.py."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from mlx_qwen3_asr.config import DEFAULT_MODEL_ID
from mlx_qwen3_asr.forced_aligner import AlignedWord
from mlx_qwen3_asr.transcribe import _to_audio_np, transcribe


class _DummyModel:
    audio_token_id = 151676
    config = SimpleNamespace(text_config=SimpleNamespace(vocab_size=151936))

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
    monkeypatch.setattr(
        tmod,
        "parse_asr_output",
        lambda raw, user_language=None: ("English", "hello world"),
    )

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


def test_transcribe_uses_default_model_id(monkeypatch):
    tmod = importlib.import_module("mlx_qwen3_asr.transcribe")
    called_paths = []

    def _fake_get(path, **kwargs):  # noqa: ANN001
        called_paths.append(path)
        return _DummyModel(), None

    monkeypatch.setattr(tmod, "_TokenizerHolder", _DummyTokenizerHolder)
    monkeypatch.setattr(tmod._ModelHolder, "get", _fake_get)
    monkeypatch.setattr(
        tmod._ModelHolder,
        "get_resolved_path",
        lambda path, dtype=mx.float16: path,
    )
    monkeypatch.setattr(
        tmod,
        "compute_features",
        lambda audio: (mx.zeros((1, 128, 100), dtype=mx.float32), mx.array([100], dtype=mx.int32)),
    )
    monkeypatch.setattr(tmod, "generate", lambda **kwargs: [10, 11, 12])

    _ = transcribe(np.zeros(3200, dtype=np.float32))
    assert called_paths == [DEFAULT_MODEL_ID]


def test_transcribe_uses_resolved_model_path_for_tokenizer(monkeypatch):
    tmod = importlib.import_module("mlx_qwen3_asr.transcribe")
    token_paths = []

    class _RecordingTokenizerHolder:
        @staticmethod
        def get(model_path: str):
            token_paths.append(model_path)
            return _DummyTokenizer(model_path)

    monkeypatch.setattr(tmod, "_TokenizerHolder", _RecordingTokenizerHolder)
    monkeypatch.setattr(tmod._ModelHolder, "get", lambda *a, **k: (_DummyModel(), None))
    monkeypatch.setattr(
        tmod._ModelHolder,
        "get_resolved_path",
        lambda path, dtype=mx.float16: "/tmp/qwen3-resolved-model",
    )
    monkeypatch.setattr(
        tmod,
        "compute_features",
        lambda audio: (mx.zeros((1, 128, 100), dtype=mx.float32), mx.array([100], dtype=mx.int32)),
    )
    monkeypatch.setattr(tmod, "generate", lambda **kwargs: [10, 11, 12])

    _ = transcribe(np.zeros(3200, dtype=np.float32))
    assert token_paths == ["/tmp/qwen3-resolved-model"]


def test_transcribe_uses_speculative_path_when_draft_model_is_set(monkeypatch):
    tmod = importlib.import_module("mlx_qwen3_asr.transcribe")

    get_calls = []

    def _fake_get(path, **kwargs):  # noqa: ANN001
        get_calls.append(path)
        return _DummyModel(), None

    monkeypatch.setattr(tmod, "_TokenizerHolder", _DummyTokenizerHolder)
    monkeypatch.setattr(tmod._ModelHolder, "get", _fake_get)
    monkeypatch.setattr(
        tmod._ModelHolder,
        "get_resolved_path",
        lambda path, dtype=mx.float16: path,
    )
    monkeypatch.setattr(
        tmod,
        "compute_features",
        lambda audio: (mx.zeros((1, 128, 100), dtype=mx.float32), mx.array([100], dtype=mx.int32)),
    )
    monkeypatch.setattr(tmod, "generate", lambda **kwargs: [10, 11, 12])

    calls = {"spec": 0}

    def _fake_speculative(**kwargs):  # noqa: ANN003
        calls["spec"] += 1
        return [10, 11, 12]

    monkeypatch.setattr(tmod, "generate_speculative", _fake_speculative)

    result = transcribe(
        np.zeros(3200, dtype=np.float32),
        model="Qwen/Qwen3-ASR-1.7B",
        draft_model="Qwen/Qwen3-ASR-0.6B",
    )

    assert result.language == "English"
    assert result.text == "hello world"
    assert calls["spec"] == 1
    assert get_calls == ["Qwen/Qwen3-ASR-1.7B", "Qwen/Qwen3-ASR-0.6B"]


def test_to_audio_np_uses_canonical_downmix_for_numpy():
    stereo = np.array([[1.0, 3.0], [5.0, 7.0]], dtype=np.float32)
    out = _to_audio_np(stereo)
    np.testing.assert_allclose(out, np.array([2.0, 6.0], dtype=np.float32), atol=1e-6)


def test_to_audio_np_normalizes_integer_pcm_numpy():
    pcm = np.array([16384, -16384], dtype=np.int16)
    out = _to_audio_np(pcm)
    np.testing.assert_allclose(out, np.array([0.5, -0.5], dtype=np.float32), atol=1e-6)
