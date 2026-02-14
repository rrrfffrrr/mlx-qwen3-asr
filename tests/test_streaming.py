"""Tests for mlx_qwen3_asr/streaming.py."""

import importlib
from types import SimpleNamespace

import numpy as np

from mlx_qwen3_asr.config import DEFAULT_MODEL_ID
from mlx_qwen3_asr.streaming import (
    UNFIXED_TOKEN_NUM,
    StreamingState,
    _split_stable_unstable,
    feed_audio,
    init_streaming,
)

# ---------------------------------------------------------------------------
# init_streaming
# ---------------------------------------------------------------------------


class TestInitStreaming:
    """Test init_streaming() returns correct initial state."""

    def test_default_state(self):
        state = init_streaming()
        assert isinstance(state, StreamingState)
        assert state.unfixed_chunk_num == 2
        assert state.unfixed_token_num == 5
        assert state.chunk_size_samples == 32000  # 2.0 * 16000
        assert state.max_context_samples == 480000  # 30.0 * 16000
        assert state._model_path == DEFAULT_MODEL_ID
        assert state.text == ""
        assert state.language == "unknown"
        assert state.chunk_id == 0
        assert len(state.buffer) == 0
        assert len(state.audio_accum) == 0
        assert state.stable_text == ""

    def test_custom_chunk_size(self):
        state = init_streaming(chunk_size_sec=5.0)
        assert state.chunk_size_samples == 80000  # 5.0 * 16000

    def test_custom_sample_rate(self):
        state = init_streaming(chunk_size_sec=2.0, sample_rate=8000)
        assert state.chunk_size_samples == 16000  # 2.0 * 8000

    def test_custom_context_window(self):
        state = init_streaming(max_context_sec=5.0, sample_rate=8000)
        assert state.max_context_samples == 40000

    def test_custom_unfixed_controls(self):
        state = init_streaming(unfixed_chunk_num=3, unfixed_token_num=7)
        assert state.unfixed_chunk_num == 3
        assert state.unfixed_token_num == 7

    def test_custom_model(self):
        state = init_streaming(model="my/custom-model")
        assert state._model_path == "my/custom-model"

    def test_invalid_chunk_size_raises(self):
        with np.testing.assert_raises(ValueError):
            init_streaming(chunk_size_sec=0.0)

    def test_invalid_context_window_raises(self):
        with np.testing.assert_raises(ValueError):
            init_streaming(max_context_sec=0.0)

    def test_invalid_sample_rate_raises(self):
        with np.testing.assert_raises(ValueError):
            init_streaming(sample_rate=0)


# ---------------------------------------------------------------------------
# StreamingState defaults
# ---------------------------------------------------------------------------


class TestStreamingStateDefaults:
    """Test StreamingState default values."""

    def test_default_buffer(self):
        state = StreamingState()
        assert isinstance(state.buffer, np.ndarray)
        assert len(state.buffer) == 0
        assert state.buffer.dtype == np.float32

    def test_default_audio_accum(self):
        state = StreamingState()
        assert isinstance(state.audio_accum, np.ndarray)
        assert len(state.audio_accum) == 0

    def test_default_text(self):
        state = StreamingState()
        assert state.text == ""

    def test_default_language(self):
        state = StreamingState()
        assert state.language == "unknown"

    def test_default_chunk_id(self):
        state = StreamingState()
        assert state.chunk_id == 0

    def test_default_unfixed_controls(self):
        state = StreamingState()
        assert state.unfixed_chunk_num == 2
        assert state.unfixed_token_num == 5

    def test_default_chunk_size_samples(self):
        state = StreamingState()
        assert state.chunk_size_samples == 32000

    def test_default_max_context_samples(self):
        state = StreamingState()
        assert state.max_context_samples == 480000

    def test_default_stable_text(self):
        state = StreamingState()
        assert state.stable_text == ""


# ---------------------------------------------------------------------------
# _split_stable_unstable
# ---------------------------------------------------------------------------


class TestSplitStableUnstable:
    """Test _split_stable_unstable() with various inputs."""

    def test_short_text_all_unstable(self):
        """Text with fewer words than unfixed_tokens is all unstable."""
        stable, unstable = _split_stable_unstable("", "hello world")
        # "hello world" = 2 words, unfixed_tokens=5 by default
        assert stable == ""
        assert unstable == "hello world"

    def test_long_text_splits_correctly(self):
        """Text with more words than unfixed_tokens should split."""
        text = "one two three four five six seven eight"
        stable, unstable = _split_stable_unstable("", text)
        # 8 words, unfixed_tokens=5 -> stable = first 3, unstable = last 5
        assert stable == "one two three"
        assert unstable == "four five six seven eight"

    def test_preserves_previous_stable_text(self):
        """New stable text should be at least as long as previous stable."""
        prev_stable = "this is already stable text that is long"
        text = "hello world"  # Too short, all unstable
        stable, unstable = _split_stable_unstable(prev_stable, text)
        assert stable == prev_stable

    def test_empty_text(self):
        stable, unstable = _split_stable_unstable("", "")
        assert stable == ""
        assert unstable == ""

    def test_exactly_unfixed_tokens(self):
        """Text with exactly unfixed_tokens words should be all unstable."""
        words = ["word"] * UNFIXED_TOKEN_NUM
        text = " ".join(words)
        stable, unstable = _split_stable_unstable("", text)
        assert stable == ""
        assert unstable == text

    def test_one_more_than_unfixed(self):
        """Text with unfixed_tokens + 1 words: 1 stable, rest unstable."""
        words = [f"word{i}" for i in range(UNFIXED_TOKEN_NUM + 1)]
        text = " ".join(words)
        stable, unstable = _split_stable_unstable("", text)
        assert stable == words[0]
        assert unstable == " ".join(words[1:])

    def test_custom_unfixed_tokens(self):
        text = "a b c d e f g h"
        stable, unstable = _split_stable_unstable("", text, unfixed_tokens=3)
        assert stable == "a b c d e"
        assert unstable == "f g h"

    def test_cjk_without_spaces_splits_by_character(self):
        text = "こんにちは世界ありがとう"
        stable, unstable = _split_stable_unstable("", text, unfixed_tokens=3)
        assert stable == text[:-3]
        assert unstable == text[-3:]

    def test_stable_grows_monotonically(self):
        """Stable text should only grow, never shrink."""
        prev_stable = "one two three"
        # New transcription has more words
        text = "one two three four five six seven eight nine ten"
        stable, unstable = _split_stable_unstable(prev_stable, text)
        assert len(stable) >= len(prev_stable)


class TestFeedAudio:
    """Test rolling streaming decode behavior."""

    def test_feed_audio_reuses_same_state_object(self, monkeypatch):
        calls = []

        def fake_transcribe(audio, model, verbose):  # noqa: ANN001
            calls.append((len(audio), model, verbose))
            return SimpleNamespace(text="hello world", language="English")

        transcribe_module = importlib.import_module("mlx_qwen3_asr.transcribe")
        monkeypatch.setattr(transcribe_module, "transcribe", fake_transcribe)

        state = init_streaming(chunk_size_sec=1.0, sample_rate=10)
        out = feed_audio(np.ones(10, dtype=np.float32), state)
        assert out is state
        assert state.chunk_id == 1
        assert calls[0][0] == 10

    def test_feed_audio_caps_decode_context_window(self, monkeypatch):
        call_lengths = []

        def fake_transcribe(audio, model, verbose):  # noqa: ANN001
            call_lengths.append(len(audio))
            return SimpleNamespace(text="a b c d e f g", language="English")

        transcribe_module = importlib.import_module("mlx_qwen3_asr.transcribe")
        monkeypatch.setattr(transcribe_module, "transcribe", fake_transcribe)

        state = init_streaming(chunk_size_sec=1.0, max_context_sec=2.0, sample_rate=10)
        feed_audio(np.ones(10, dtype=np.float32), state)
        feed_audio(np.ones(10, dtype=np.float32), state)
        feed_audio(np.ones(10, dtype=np.float32), state)

        assert call_lengths == [10, 20, 20]

    def test_feed_audio_honors_unfixed_chunk_warmup(self, monkeypatch):
        def fake_transcribe(audio, model, verbose):  # noqa: ANN001
            return SimpleNamespace(text="one two three four five six", language="English")

        transcribe_module = importlib.import_module("mlx_qwen3_asr.transcribe")
        monkeypatch.setattr(transcribe_module, "transcribe", fake_transcribe)

        state = init_streaming(
            chunk_size_sec=1.0,
            sample_rate=10,
            unfixed_chunk_num=1,
            unfixed_token_num=2,
        )

        feed_audio(np.ones(10, dtype=np.float32), state)
        # First chunk is warmup: stable text should not advance.
        assert state.stable_text == ""

        feed_audio(np.ones(10, dtype=np.float32), state)
        # After warmup, keep trailing 2 words unstable.
        assert state.stable_text == "one two three four"

    def test_feed_audio_accepts_int16_pcm(self, monkeypatch):
        captured = {}

        def fake_transcribe(audio, model, verbose):  # noqa: ANN001
            captured["dtype"] = np.asarray(audio).dtype
            captured["max_abs"] = float(np.max(np.abs(np.asarray(audio))))
            return SimpleNamespace(text="hello", language="English")

        transcribe_module = importlib.import_module("mlx_qwen3_asr.transcribe")
        monkeypatch.setattr(transcribe_module, "transcribe", fake_transcribe)

        state = init_streaming(chunk_size_sec=1.0, sample_rate=10)
        feed_audio(np.full((10,), 16384, dtype=np.int16), state)

        assert captured["dtype"] == np.float32
        assert np.isclose(captured["max_abs"], 0.5)

    def test_feed_audio_flattens_non_1d_input(self, monkeypatch):
        call_lengths = []

        def fake_transcribe(audio, model, verbose):  # noqa: ANN001
            call_lengths.append(len(audio))
            return SimpleNamespace(text="hello", language="English")

        transcribe_module = importlib.import_module("mlx_qwen3_asr.transcribe")
        monkeypatch.setattr(transcribe_module, "transcribe", fake_transcribe)

        state = init_streaming(chunk_size_sec=1.0, sample_rate=10)
        feed_audio(np.ones((2, 5), dtype=np.float32), state)

        assert call_lengths == [10]

    def test_feed_audio_none_raises(self):
        state = init_streaming(chunk_size_sec=1.0, sample_rate=10)
        with np.testing.assert_raises(ValueError):
            feed_audio(None, state)  # type: ignore[arg-type]
