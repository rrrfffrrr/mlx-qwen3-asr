"""Tests for mlx_qwen3_asr/tokenizer.py.

Only tests parse_asr_output() -- the Tokenizer class requires HF download.
"""

import sys

import mlx_qwen3_asr.tokenizer as tokmod
from mlx_qwen3_asr.tokenizer import _TokenizerHolder, parse_asr_output


class TestParseASROutputStandard:
    """Test parse_asr_output() with standard format."""

    def test_english(self):
        text = "language English<asr_text>hello world"
        lang, transcript = parse_asr_output(text)
        assert lang == "English"
        assert transcript == "hello world"

    def test_chinese(self):
        text = "language Chinese<asr_text>你好世界"
        lang, transcript = parse_asr_output(text)
        assert lang == "Chinese"
        assert transcript == "你好世界"

    def test_japanese(self):
        text = "language Japanese<asr_text>こんにちは"
        lang, transcript = parse_asr_output(text)
        assert lang == "Japanese"
        assert transcript == "こんにちは"


class TestParseASROutputSpecialTokens:
    """Test parse_asr_output() cleans trailing special tokens."""

    def test_cleans_im_end(self):
        text = "language English<asr_text>hello world<|im_end|>"
        lang, transcript = parse_asr_output(text)
        assert transcript == "hello world"

    def test_cleans_endoftext(self):
        text = "language English<asr_text>hello world<|endoftext|>"
        lang, transcript = parse_asr_output(text)
        assert transcript == "hello world"

    def test_cleans_multiple_special_tokens(self):
        text = "language English<asr_text>hello world<|im_end|><|endoftext|>"
        lang, transcript = parse_asr_output(text)
        assert transcript == "hello world"


class TestParseASROutputFallback:
    """Test parse_asr_output() fallback with no <asr_text> marker."""

    def test_no_marker(self):
        text = "just some text without marker"
        lang, transcript = parse_asr_output(text)
        assert lang == "unknown"
        assert transcript == "just some text without marker"

    def test_empty_string(self):
        lang, transcript = parse_asr_output("")
        assert lang == "unknown"
        assert transcript == ""

    def test_whitespace_only(self):
        lang, transcript = parse_asr_output("   ")
        assert lang == "unknown"
        assert transcript == ""


class TestParseASROutputEdgeCases:
    """Test parse_asr_output() edge cases."""

    def test_empty_transcription(self):
        text = "language English<asr_text>"
        lang, transcript = parse_asr_output(text)
        assert lang == "English"
        assert transcript == ""

    def test_no_language_prefix(self):
        """If the text before <asr_text> doesn't start with 'language '."""
        text = "English<asr_text>hello"
        lang, transcript = parse_asr_output(text)
        assert lang == "English"
        assert transcript == "hello"

    def test_whitespace_in_transcript(self):
        text = "language English<asr_text>  hello world  "
        lang, transcript = parse_asr_output(text)
        assert transcript == "hello world"


def test_tokenizer_holder_caches_by_model_path(monkeypatch):
    created = []

    class _DummyTokenizer:
        def __init__(self, model_path: str):
            created.append(model_path)
            self.model_path = model_path

    monkeypatch.setattr(tokmod, "Tokenizer", _DummyTokenizer)
    _TokenizerHolder.clear()

    t1 = _TokenizerHolder.get("repo/a")
    t2 = _TokenizerHolder.get("repo/a")
    t3 = _TokenizerHolder.get("repo/b")

    assert t1 is t2
    assert t1 is not t3
    assert created == ["repo/a", "repo/b"]


def test_load_hf_tokenizer_uses_fix_mistral_regex(monkeypatch):
    calls = []

    class _DummyAutoTokenizer:
        @staticmethod
        def from_pretrained(model_path, **kwargs):  # noqa: ANN001
            calls.append((model_path, kwargs))
            return object()

    class _DummyTransformersModule:
        AutoTokenizer = _DummyAutoTokenizer

    monkeypatch.setitem(sys.modules, "transformers", _DummyTransformersModule)

    tok = tokmod._load_hf_tokenizer("repo/a")
    assert tok is not None
    assert len(calls) == 1
    assert calls[0][0] == "repo/a"
    assert calls[0][1]["trust_remote_code"] is True
    assert calls[0][1]["fix_mistral_regex"] is True


def test_load_hf_tokenizer_falls_back_when_fix_flag_unsupported(monkeypatch):
    calls = []

    class _DummyAutoTokenizer:
        @staticmethod
        def from_pretrained(model_path, **kwargs):  # noqa: ANN001
            calls.append((model_path, kwargs))
            if kwargs.get("fix_mistral_regex"):
                raise TypeError("unexpected keyword argument 'fix_mistral_regex'")
            return object()

    class _DummyTransformersModule:
        AutoTokenizer = _DummyAutoTokenizer

    monkeypatch.setitem(sys.modules, "transformers", _DummyTransformersModule)

    tok = tokmod._load_hf_tokenizer("repo/b")
    assert tok is not None
    assert len(calls) == 2
    assert calls[0][1]["fix_mistral_regex"] is True
    assert "fix_mistral_regex" not in calls[1][1]
