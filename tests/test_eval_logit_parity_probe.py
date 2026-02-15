"""Unit tests for scripts/eval_logit_parity_probe.py helpers."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import mlx.core as mx


def _load_module():
    path = Path("scripts/eval_logit_parity_probe.py")
    spec = importlib.util.spec_from_file_location("eval_logit_parity_probe", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_first_mismatch_handles_equal_content_and_length_delta():
    mod = _load_module()
    assert mod._first_mismatch([1, 2, 3], [1, 2, 3]) == -1
    assert mod._first_mismatch([1, 2, 3], [1, 9, 3]) == 1
    assert mod._first_mismatch([1, 2], [1, 2, 3]) == 2


def test_trim_trailing_eos_trims_step_infos_in_lockstep():
    mod = _load_module()
    eos = sorted(mod.EOS_IDS)[0]
    tokens = [7, 9, eos, eos]
    steps = [{"s": 1}, {"s": 2}, {"s": 3}, {"s": 4}]
    mod._trim_trailing_eos(tokens, steps)
    assert tokens == [7, 9]
    assert steps == [{"s": 1}, {"s": 2}]


def test_topk_info_returns_descending_ids_values_and_margin():
    mod = _load_module()
    logits = mx.array([[0.5, 3.0, 1.25, 2.0]], dtype=mx.float32)
    out = mod._topk_info(logits, k=3)
    assert out["topk_token_ids"] == [1, 3, 2]
    assert out["top_token_id"] == 1
    assert out["topk_logits"][0] >= out["topk_logits"][1] >= out["topk_logits"][2]
    assert out["margin_top1_top2"] == out["topk_logits"][0] - out["topk_logits"][1]


def test_collect_sample_ids_prefers_explicit_ids_and_limits():
    mod = _load_module()
    out = mod._collect_sample_ids(
        parity_json=None,
        requested_sample_ids=["a", "b", "c"],
        num_samples=2,
    )
    assert out == ["a", "b"]


def test_collect_sample_ids_reads_mismatch_rows_from_parity_json(tmp_path: Path):
    mod = _load_module()
    parity = tmp_path / "parity.json"
    parity.write_text(
        json.dumps(
            {
                "rows": [
                    {"sample_id": "s0", "token_match": True},
                    {"sample_id": "s1", "token_match": False},
                    {"sample_id": "s2", "token_match": False},
                ]
            }
        ),
        encoding="utf-8",
    )
    out = mod._collect_sample_ids(
        parity_json=parity,
        requested_sample_ids=[],
        num_samples=1,
    )
    assert out == ["s1"]
