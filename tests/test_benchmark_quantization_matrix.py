"""Tests for scripts/benchmark_quantization_matrix.py helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_module():
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "benchmark_quantization_matrix.py"
    )
    module_name = "benchmark_quantization_matrix_script"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_configs_parses_expected_profiles():
    mod = _load_module()
    configs = mod._parse_configs("fp16,4:64,8:64")
    assert [c.label for c in configs] == ["fp16", "4bit-g64", "8bit-g64"]
    assert [c.bits for c in configs] == [None, 4, 8]
    assert [c.group_size for c in configs] == [None, 64, 64]


def test_parse_configs_rejects_empty_list():
    mod = _load_module()
    with pytest.raises(ValueError, match="No configs specified"):
        mod._parse_configs("")
