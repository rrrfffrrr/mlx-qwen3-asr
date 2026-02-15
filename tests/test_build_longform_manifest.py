"""Unit tests for scripts/build_longform_manifest.py helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_module():
    path = Path("scripts/build_longform_manifest.py")
    spec = importlib.util.spec_from_file_location("build_longform_manifest", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_manifest_reads_required_fields(tmp_path):
    mod = _load_module()
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"stub")
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(
        '{"sample_id":"s1","subset":"x","speaker_id":"spk","language":"English","audio_path":"'
        + str(audio)
        + '","reference_text":"hello"}\n',
        encoding="utf-8",
    )
    rows = mod._parse_manifest(manifest)
    assert len(rows) == 1
    assert rows[0].sample_id == "s1"
    assert rows[0].language == "English"


def test_build_concat_respects_target(monkeypatch):
    mod = _load_module()

    rows = [
        mod.ManifestRow(
            sample_id=f"s{i}",
            subset="x",
            speaker_id="spk",
            language="English",
            audio_path=Path(f"/tmp/{i}.wav"),
            reference_text=f"t{i}",
        )
        for i in range(3)
    ]

    def fake_read(path):  # noqa: ANN001
        base = int(path.stem)
        return np.ones(1600 + base * 100, dtype=np.float32)

    monkeypatch.setattr(mod, "_read_audio_16k", fake_read)
    merged, ids, texts, cursor = mod._build_concat(
        rows,
        target_samples=4000,
        silence_samples=200,
        start_cursor=0,
    )
    assert merged.size >= 4000
    assert len(ids) >= 2
    assert len(texts) == len(ids)
    assert cursor > 0
