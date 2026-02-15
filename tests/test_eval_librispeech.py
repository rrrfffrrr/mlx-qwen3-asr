"""Tests for scripts/eval_librispeech.py sampling behavior."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "eval_librispeech.py"
    module_name = "eval_librispeech_script"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _write_transcript_chunk(base: Path, speaker: str, chapter: str, utt_texts: list[str]) -> None:
    chapter_dir = base / speaker / chapter
    chapter_dir.mkdir(parents=True, exist_ok=True)
    trans = chapter_dir / f"{speaker}-{chapter}.trans.txt"
    lines = []
    for idx, text in enumerate(utt_texts, start=1):
        sample_id = f"{speaker}-{chapter}-{idx:04d}"
        lines.append(f"{sample_id} {text}\n")
        (chapter_dir / f"{sample_id}.flac").write_bytes(b"fake")
    trans.write_text("".join(lines), encoding="utf-8")


def test_collect_samples_round_robin_balances_speakers(tmp_path: Path):
    mod = _load_module()
    split_root = tmp_path / "LibriSpeech" / "test-clean"

    _write_transcript_chunk(split_root, "100", "200", ["a", "b", "c"])
    _write_transcript_chunk(split_root, "101", "201", ["d", "e", "f"])
    _write_transcript_chunk(split_root, "102", "202", ["g", "h", "i"])

    samples = mod._collect_samples(
        split_root=split_root,
        max_samples=6,
        sampling="speaker_round_robin",
    )
    speaker_ids = [s.speaker_id for s in samples]

    # Deterministic round-robin should interleave speakers.
    assert speaker_ids == ["100", "101", "102", "100", "101", "102"]


def test_collect_samples_sequential_keeps_sorted_order(tmp_path: Path):
    mod = _load_module()
    split_root = tmp_path / "LibriSpeech" / "test-clean"

    _write_transcript_chunk(split_root, "100", "200", ["a", "b", "c"])
    _write_transcript_chunk(split_root, "101", "201", ["d", "e", "f"])

    samples = mod._collect_samples(
        split_root=split_root,
        max_samples=4,
        sampling="sequential",
    )
    sample_ids = [s.sample_id for s in samples]
    assert sample_ids == [
        "100-200-0001",
        "100-200-0002",
        "100-200-0003",
        "101-201-0001",
    ]


def test_threshold_failures_none_when_under_limits():
    mod = _load_module()
    failures = mod._threshold_failures(  # noqa: SLF001
        wer=0.02,
        cer=0.01,
        fail_wer_above=0.03,
        fail_cer_above=0.02,
    )
    assert failures == []


def test_threshold_failures_reports_both_metrics():
    mod = _load_module()
    failures = mod._threshold_failures(  # noqa: SLF001
        wer=0.05,
        cer=0.03,
        fail_wer_above=0.03,
        fail_cer_above=0.02,
    )
    assert len(failures) == 2
    assert "WER regression gate failed" in failures[0]
    assert "CER regression gate failed" in failures[1]
