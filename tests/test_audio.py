"""Tests for mlx_qwen3_asr/audio.py."""

import mlx.core as mx
import numpy as np
import pytest

from mlx_qwen3_asr.audio import (
    N_FFT,
    SAMPLE_RATE,
    _reflect_pad,
    compute_features,
    load_audio,
    log_mel_spectrogram,
    mel_filters,
    stft,
)

# ---------------------------------------------------------------------------
# load_audio
# ---------------------------------------------------------------------------


class TestLoadAudioNumpy:
    """Test load_audio() with numpy array inputs."""

    def test_mono_array(self):
        arr = np.random.randn(16000).astype(np.float32)
        result = load_audio(arr)
        assert isinstance(result, mx.array)
        assert result.shape == (16000,)

    def test_stereo_to_mono(self):
        """Stereo (2-channel) should be averaged to mono."""
        stereo = np.random.randn(16000, 2).astype(np.float32)
        result = load_audio(stereo)
        assert result.ndim == 1
        assert result.shape == (16000,)
        # Check that it's the mean of the two channels
        expected = stereo.mean(axis=-1)
        np.testing.assert_allclose(np.array(result), expected, atol=1e-5)

    def test_empty_array(self):
        arr = np.array([], dtype=np.float32)
        result = load_audio(arr)
        assert result.size == 0

    def test_integer_array_cast_to_float(self):
        """Integer arrays should be cast to float32."""
        arr = np.array([100, 200, 300], dtype=np.int16)
        result = load_audio(arr)
        assert result.dtype == mx.float32


class TestLoadAudioTuple:
    """Test load_audio() with (array, sample_rate) tuple inputs."""

    def test_same_sample_rate(self):
        arr = np.random.randn(16000).astype(np.float32)
        result = load_audio((arr, SAMPLE_RATE))
        assert result.shape == (16000,)

    def test_empty_tuple(self):
        arr = np.array([], dtype=np.float32)
        result = load_audio((arr, SAMPLE_RATE))
        assert result.size == 0

    def test_stereo_tuple(self):
        stereo = np.random.randn(16000, 2).astype(np.float32)
        result = load_audio((stereo, SAMPLE_RATE))
        assert result.ndim == 1


class TestLoadAudioErrors:
    """Test load_audio() error handling."""

    def test_unsupported_type_raises_value_error(self):
        with pytest.raises(ValueError, match="Unsupported source type"):
            load_audio(12345)

    def test_unsupported_type_list_raises_value_error(self):
        with pytest.raises(ValueError, match="Unsupported source type"):
            load_audio([1, 2, 3])


# ---------------------------------------------------------------------------
# _reflect_pad
# ---------------------------------------------------------------------------


class TestReflectPad:
    """Test _reflect_pad() with known inputs."""

    def test_known_values(self):
        x = mx.array([1.0, 2.0, 3.0, 4.0])
        result = _reflect_pad(x, pad_len=2)
        # Reflect: [3, 2, 1, 2, 3, 4, 3, 2]
        expected = [3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]
        np.testing.assert_array_equal(np.array(result), expected)

    def test_zero_pad(self):
        x = mx.array([1.0, 2.0, 3.0])
        result = _reflect_pad(x, pad_len=0)
        np.testing.assert_array_equal(np.array(result), [1.0, 2.0, 3.0])

    def test_pad_len_one(self):
        x = mx.array([10.0, 20.0, 30.0])
        result = _reflect_pad(x, pad_len=1)
        # Left pad: x[1:2][::-1] = [20]
        # Right pad: x[-2:-1][::-1] = [20]
        expected = [20.0, 10.0, 20.0, 30.0, 20.0]
        np.testing.assert_array_equal(np.array(result), expected)

    def test_output_length(self):
        x = mx.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pad_len = 3
        result = _reflect_pad(x, pad_len)
        assert result.shape[0] == len(np.array(x)) + 2 * pad_len


# ---------------------------------------------------------------------------
# mel_filters
# ---------------------------------------------------------------------------


class TestMelFilters:
    """Test mel_filters() loads and returns correct shape."""

    def test_default_shape(self):
        filters = mel_filters()
        assert filters.shape == (128, 201)

    def test_explicit_128(self):
        filters = mel_filters(n_mels=128)
        assert filters.shape == (128, 201)

    def test_dtype_is_float(self):
        filters = mel_filters()
        assert filters.dtype == mx.float32


# ---------------------------------------------------------------------------
# stft
# ---------------------------------------------------------------------------


class TestSTFT:
    """Test stft() output shape for known input length."""

    def test_output_shape(self):
        # n_samples = 1600 (0.1s at 16kHz)
        audio = mx.zeros((1600,))
        window = mx.array(np.hanning(N_FFT + 1)[:-1].astype(np.float32))
        result = stft(audio, window, nperseg=N_FFT)
        # After reflect padding: 1600 + 2*(400//2) = 1600 + 400 = 2000
        # num_frames = 1 + (2000 - 400) // 160 = 1 + 1600//160 = 1 + 10 = 11
        assert result.shape[0] == 11
        # rfft output: N_FFT // 2 + 1 = 201
        assert result.shape[1] == N_FFT // 2 + 1

    def test_output_shape_longer_audio(self):
        audio = mx.zeros((16000,))  # 1 second
        window = mx.array(np.hanning(N_FFT + 1)[:-1].astype(np.float32))
        result = stft(audio, window, nperseg=N_FFT)
        # After padding: 16000 + 400 = 16400
        # num_frames = 1 + (16400 - 400) // 160 = 1 + 16000//160 = 1 + 100 = 101
        assert result.shape[0] == 101
        assert result.shape[1] == 201


# ---------------------------------------------------------------------------
# log_mel_spectrogram
# ---------------------------------------------------------------------------


class TestLogMelSpectrogram:
    """Test log_mel_spectrogram() output shape and error handling."""

    def test_output_shape_for_known_length(self):
        # 1 second of audio
        audio = mx.array(np.random.randn(16000).astype(np.float32))
        result = log_mel_spectrogram(audio)
        assert result.shape[0] == 128  # n_mels
        # n_frames = 101 for 16000 samples (from STFT test)
        assert result.shape[1] == 101

    def test_output_shape_short_audio(self):
        # 0.1 second of audio
        audio = mx.array(np.random.randn(1600).astype(np.float32))
        result = log_mel_spectrogram(audio)
        assert result.shape[0] == 128
        assert result.shape[1] == 11

    def test_raises_for_empty_audio(self):
        audio = mx.array(np.array([], dtype=np.float32))
        with pytest.raises(ValueError, match="empty audio"):
            log_mel_spectrogram(audio)

    def test_output_dtype(self):
        audio = mx.array(np.random.randn(1600).astype(np.float32))
        result = log_mel_spectrogram(audio)
        assert result.dtype == mx.float32


# ---------------------------------------------------------------------------
# compute_features
# ---------------------------------------------------------------------------


class TestComputeFeatures:
    """Test compute_features() shape/length behavior."""

    def test_default_no_padding(self):
        audio = np.random.randn(2 * SAMPLE_RATE).astype(np.float32)
        mel, feature_lens = compute_features(audio)
        assert mel.shape == (1, 128, 200)
        assert int(feature_lens.item()) == 200

    def test_long_audio_not_truncated(self):
        # 31s should produce 3100 frames with 10ms hop, not 3000.
        audio = np.random.randn(31 * SAMPLE_RATE).astype(np.float32)
        mel, feature_lens = compute_features(audio)
        assert mel.shape == (1, 128, 3100)
        assert int(feature_lens.item()) == 3100

    def test_max_length_padding_still_keeps_long_audio(self):
        # With truncation disabled, padding="max_length" should not clamp >30s.
        audio = np.random.randn(31 * SAMPLE_RATE).astype(np.float32)
        mel, feature_lens = compute_features(audio, padding="max_length")
        assert mel.shape == (1, 128, 3100)
        assert int(feature_lens.item()) == 3100
