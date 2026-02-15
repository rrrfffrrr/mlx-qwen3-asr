#!/usr/bin/env python3
"""Compatibility wrapper for scripts/eval/benchmark_quantization_matrix.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_IMPL_PATH = Path(__file__).resolve().parent / "eval" / "benchmark_quantization_matrix.py"
_SPEC = importlib.util.spec_from_file_location("benchmark_quantization_matrix_impl", _IMPL_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Failed to load implementation module: {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

for _name, _value in vars(_IMPL).items():
    if _name.startswith("__") and _name not in {"__doc__"}:
        continue
    globals()[_name] = _value

if __name__ == "__main__":
    raise SystemExit(_IMPL.main())
