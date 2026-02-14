"""Tests for ASR evaluation metrics."""

from __future__ import annotations

from mlx_qwen3_asr.eval_metrics import compute_cer, compute_wer, edit_distance, normalize_text


def test_normalize_text():
    assert normalize_text("Hello,  WORLD!!") == "hello world"
    assert normalize_text("Don't-stop") == "don't stop"


def test_edit_distance():
    assert edit_distance(["a", "b", "c"], ["a", "x", "c"]) == 1
    assert edit_distance([], ["x"]) == 1
    assert edit_distance(["x"], []) == 1


def test_compute_wer_and_cer():
    refs = ["hello world", "fast speech"]
    hyps = ["hello wurld", "fast"]

    wer = compute_wer(refs, hyps)
    cer = compute_cer(refs, hyps)

    # WER: 2 substitutions/deletions over 4 words.
    assert wer == 0.5
    assert 0.0 < cer < 1.0
