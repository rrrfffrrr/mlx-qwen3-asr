"""Unit tests for scripts/eval_manifest_quality.py helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "eval_manifest_quality.py"
    module_name = "eval_manifest_quality_script"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_normalize_quality_text_is_unicode_safe():
    mod = _load_module()
    assert mod._normalize_quality_text("你好，世界！  テスト。") == "你好 世界 テスト"  # noqa: SLF001
    assert mod._normalize_quality_text("Café — it's me") == "café it's me"  # noqa: SLF001


def test_wer_tokens_falls_back_to_characters_without_spaces():
    mod = _load_module()
    assert mod._wer_tokens("hello world") == ["hello", "world"]  # noqa: SLF001
    assert mod._wer_tokens("你好世界") == ["你", "好", "世", "界"]  # noqa: SLF001


def test_char_primary_language_detection():
    mod = _load_module()
    assert mod._is_char_primary_language("Chinese") is True  # noqa: SLF001
    assert mod._is_char_primary_language("ja_jp") is True  # noqa: SLF001
    assert mod._is_char_primary_language("ko-kr") is True  # noqa: SLF001
    assert mod._is_char_primary_language("English") is False  # noqa: SLF001


def test_threshold_failures_reports_expected_items():
    mod = _load_module()
    failures = mod._threshold_failures(  # noqa: SLF001
        wer=0.11,
        cer=0.07,
        primary_error_rate=0.09,
        fail_wer_above=0.10,
        fail_cer_above=0.06,
        fail_primary_above=0.08,
    )
    assert len(failures) == 3
    assert "WER regression gate failed" in failures[0]
    assert "CER regression gate failed" in failures[1]
    assert "Primary regression gate failed" in failures[2]


def test_parse_manifest_requires_reference_text(tmp_path: Path):
    mod = _load_module()
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"stub")
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(
        '{"sample_id":"s1","audio_path":"'
        + str(audio)
        + '","reference_text":""}\n',
        encoding="utf-8",
    )
    try:
        mod._parse_manifest(manifest)  # noqa: SLF001
    except ValueError as exc:
        assert "reference_text" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError for missing reference_text")
