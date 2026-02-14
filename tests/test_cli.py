"""Tests for mlx_qwen3_asr/cli.py."""

from __future__ import annotations

import subprocess
import sys


def test_cli_timestamps_fails_fast():
    result = subprocess.run(
        [sys.executable, "-m", "mlx_qwen3_asr.cli", "--timestamps", "dummy.wav"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "--timestamps is not available yet" in result.stderr
