"""Tests for explicit Session API."""

import mlx.core as mx
import numpy as np

import mlx_qwen3_asr.session as sessmod
from mlx_qwen3_asr.transcribe import TranscriptionResult


def test_session_loads_model_and_tokenizer_from_resolved_path(monkeypatch):
    created = {}

    class _DummyTokenizer:
        def __init__(self, path):  # noqa: ANN001
            created["tokenizer_path"] = path

    class _DummyModel:
        pass

    monkeypatch.setattr(sessmod, "Tokenizer", _DummyTokenizer)
    monkeypatch.setattr(sessmod, "load_model", lambda model, dtype: (_DummyModel(), object()))
    monkeypatch.setattr(sessmod, "_resolve_path", lambda model: "/tmp/resolved-model")

    session = sessmod.Session("repo/a", dtype=mx.float16)
    assert session.model_id == "repo/a"
    assert session.dtype == mx.float16
    assert created["tokenizer_path"] == "/tmp/resolved-model"


def test_session_transcribe_passes_explicit_components(monkeypatch):
    class _DummyTokenizer:
        def __init__(self, path):  # noqa: ANN001
            self.path = path

    class _DummyModel:
        pass

    calls = {}

    monkeypatch.setattr(sessmod, "Tokenizer", _DummyTokenizer)
    monkeypatch.setattr(sessmod, "load_model", lambda model, dtype: (_DummyModel(), object()))
    monkeypatch.setattr(sessmod, "_resolve_path", lambda model: "/tmp/resolved-model")
    monkeypatch.setattr(sessmod, "_to_audio_np", lambda audio: np.zeros(160, dtype=np.float32))
    monkeypatch.setattr(sessmod, "_resolve_aligner", lambda rt, fa: "ALIGNER")
    monkeypatch.setattr(sessmod, "_resolve_draft_model", lambda **kwargs: None)

    def fake_transcribe_loaded_components(**kwargs):  # noqa: ANN003
        calls.update(kwargs)
        return TranscriptionResult(text="ok", language="English")

    monkeypatch.setattr(sessmod, "_transcribe_loaded_components", fake_transcribe_loaded_components)

    session = sessmod.Session("repo/a", dtype=mx.float32)
    out = session.transcribe(
        np.zeros(160, dtype=np.float32),
        language="English",
        return_timestamps=True,
        max_new_tokens=77,
        verbose=True,
    )

    assert out.text == "ok"
    assert calls["dtype"] == mx.float32
    assert calls["language"] == "English"
    assert calls["aligner"] == "ALIGNER"
    assert calls["return_timestamps"] is True
    assert calls["max_new_tokens"] == 77
    assert calls["num_draft_tokens"] == 4
    assert calls["verbose"] is True


def test_session_transcribe_supports_draft_model(monkeypatch):
    class _DummyTokenizer:
        def __init__(self, path):  # noqa: ANN001
            self.path = path

    class _DummyModel:
        pass

    calls = {}

    monkeypatch.setattr(sessmod, "Tokenizer", _DummyTokenizer)
    monkeypatch.setattr(sessmod, "load_model", lambda model, dtype: (_DummyModel(), object()))
    monkeypatch.setattr(sessmod, "_resolve_path", lambda model: "/tmp/resolved-model")
    monkeypatch.setattr(sessmod, "_to_audio_np", lambda audio: np.zeros(160, dtype=np.float32))
    monkeypatch.setattr(sessmod, "_resolve_aligner", lambda rt, fa: None)
    monkeypatch.setattr(sessmod, "_resolve_draft_model", lambda **kwargs: "DRAFT")

    def fake_transcribe_loaded_components(**kwargs):  # noqa: ANN003
        calls.update(kwargs)
        return TranscriptionResult(text="ok", language="English")

    monkeypatch.setattr(sessmod, "_transcribe_loaded_components", fake_transcribe_loaded_components)

    session = sessmod.Session("repo/a", dtype=mx.float16)
    out = session.transcribe(
        np.zeros(160, dtype=np.float32),
        draft_model="repo/draft",
        num_draft_tokens=7,
    )

    assert out.text == "ok"
    assert calls["draft_model_obj"] == "DRAFT"
    assert calls["num_draft_tokens"] == 7


def test_session_with_preloaded_model_requires_tokenizer_metadata():
    class _DummyModel:
        pass

    with np.testing.assert_raises(ValueError):
        sessmod.Session(_DummyModel())


def test_session_with_preloaded_model_uses_embedded_tokenizer_metadata(monkeypatch):
    created = {}

    class _DummyTokenizer:
        def __init__(self, path):  # noqa: ANN001
            created["tokenizer_path"] = path

    class _DummyModel:
        _resolved_model_path = "/tmp/preloaded-tokenizer-model"
        _source_model_id = "repo/preloaded"

    monkeypatch.setattr(sessmod, "Tokenizer", _DummyTokenizer)
    session = sessmod.Session(_DummyModel())
    assert session.model_id == "repo/preloaded"
    assert created["tokenizer_path"] == "/tmp/preloaded-tokenizer-model"
