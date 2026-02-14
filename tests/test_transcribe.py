"""Tests for mlx_qwen3_asr/transcribe.py."""

import numpy as np
import pytest

from mlx_qwen3_asr.transcribe import transcribe


def test_transcribe_timestamps_not_implemented():
    audio = np.zeros(1600, dtype=np.float32)

    with pytest.raises(NotImplementedError, match="return_timestamps is not available yet"):
        transcribe(audio, return_timestamps=True)
