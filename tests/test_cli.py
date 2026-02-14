"""Tests for mlx_qwen3_asr/cli.py."""

from __future__ import annotations

import subprocess
import sys

import pytest


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


def test_cli_timestamps_preflight_error_is_clear(monkeypatch, capsys, tmp_path):
    cli = __import__("mlx_qwen3_asr.cli", fromlist=["main"])
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"RIFF")

    monkeypatch.setattr(sys, "argv", ["mlx-qwen3-asr", "--timestamps", str(audio_path)])
    monkeypatch.setattr(
        "importlib.util.find_spec",
        lambda name: None if name == "qwen_asr" else object(),
    )

    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1
    assert "--timestamps requires optional dependency `qwen-asr`" in capsys.readouterr().err
