"""Unit tests for scripts/eval_reference_parity_suite.py helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_module():
    path = Path("scripts/eval_reference_parity_suite.py")
    spec = importlib.util.spec_from_file_location("eval_reference_parity_suite", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_trim_eos_removes_only_trailing():
    mod = _load_module()
    eos = sorted(mod.EOS_IDS)[0]
    assert mod._trim_eos([1, eos, 2, eos, eos]) == [1, eos, 2]


def test_first_mismatch_handles_len_and_content():
    mod = _load_module()
    assert mod._first_mismatch([1, 2, 3], [1, 2, 3]) == -1
    assert mod._first_mismatch([1, 2, 3], [1, 9, 3]) == 1
    assert mod._first_mismatch([1, 2], [1, 2, 3]) == 2


def test_build_long_mixes_creates_expected_count():
    mod = _load_module()
    base = [
        mod.SuiteSample(
            sample_id=f"s{i}",
            subset="test-clean",
            speaker_id=f"spk{i % 2}",
            language="English",
            audio_path=None,
            source_sample_ids=[],
            audio=np.ones(1600, dtype=np.float32),
            sample_rate=16000,
        )
        for i in range(4)
    ]
    mixes = mod._build_long_mixes(base, long_mixes=2, long_mix_segments=3, silence_sec=0.1)
    assert len(mixes) == 2
    assert all(m.sample_id.startswith("longmix-") for m in mixes)
    assert all(len(m.source_sample_ids) == 3 for m in mixes)
    assert all(m.audio is not None and m.audio.size > 0 for m in mixes)
