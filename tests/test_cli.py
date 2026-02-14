"""Tests for mlx_qwen3_asr/cli.py."""

from __future__ import annotations

import subprocess
import sys


def test_cli_timestamps_flag_is_accepted():
    result = subprocess.run(
        [sys.executable, "-m", "mlx_qwen3_asr.cli", "--timestamps", "dummy.wav"],
        capture_output=True,
        text=True,
        check=False,
    )

    # The command should parse and run; missing file is handled at runtime.
    assert result.returncode == 1
    assert "File not found: dummy.wav" in result.stderr
