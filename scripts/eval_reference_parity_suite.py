#!/usr/bin/env python3
"""Run token-level parity suite against official qwen-asr reference.

Extends single-fixture parity by supporting:
- LibriSpeech clean/other subsets
- deterministic multi-speaker sampling
- optional long mixed-speaker synthetic clips
- optional external manifest (for multilingual parity runs)
"""

from __future__ import annotations

import argparse
import json
import tarfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np

from mlx_qwen3_asr.audio import load_audio
from mlx_qwen3_asr.generate import GenerationConfig, generate
from mlx_qwen3_asr.load_models import load_model

OPENSLR_BASE = "https://www.openslr.org/resources/12"
SPLIT_ARCHIVES = {
    "test-clean": "test-clean.tar.gz",
    "test-other": "test-other.tar.gz",
}
EOS_IDS = {151643, 151645}


@dataclass(frozen=True)
class LibriSample:
    sample_id: str
    speaker_id: str
    audio_path: Path
    subset: str


@dataclass(frozen=True)
class SuiteSample:
    sample_id: str
    subset: str
    speaker_id: str
    language: Optional[str]
    audio_path: Optional[Path]
    source_sample_ids: list[str]
    # Optional inline audio payload for synthetic mixes.
    audio: Optional[np.ndarray] = None
    sample_rate: int = 16000


def _dtype_from_name(name: str) -> mx.Dtype:
    mapping = {
        "float16": mx.float16,
        "float32": mx.float32,
        "bfloat16": mx.bfloat16,
    }
    return mapping[name]


def _download_file(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as src, dst.open("wb") as out:  # noqa: S310
        out.write(src.read())


def _extract_archive(archive_path: Path, dst_dir: Path) -> None:
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=dst_dir)


def _ensure_split(data_dir: Path, subset: str) -> Path:
    archive_name = SPLIT_ARCHIVES[subset]
    split_root = data_dir / "LibriSpeech" / subset
    if split_root.exists():
        return split_root

    archive_path = data_dir / archive_name
    if not archive_path.exists():
        url = f"{OPENSLR_BASE}/{archive_name}"
        print(f"Downloading {url} -> {archive_path}")
        _download_file(url, archive_path)
    print(f"Extracting {archive_path} -> {data_dir}")
    _extract_archive(archive_path, data_dir)
    if not split_root.exists():
        raise FileNotFoundError(f"Expected split path not found after extract: {split_root}")
    return split_root


def _speaker_id_from_sample_id(sample_id: str) -> str:
    parts = sample_id.split("-", 1)
    return parts[0] if parts else sample_id


def _collect_librispeech_samples(split_root: Path, subset: str) -> list[LibriSample]:
    out: list[LibriSample] = []
    for trans_path in sorted(split_root.rglob("*.trans.txt")):
        with trans_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = line.strip()
                if not row:
                    continue
                sample_id = row.split(" ", 1)[0]
                audio_path = trans_path.parent / f"{sample_id}.flac"
                if audio_path.exists():
                    out.append(
                        LibriSample(
                            sample_id=sample_id,
                            speaker_id=_speaker_id_from_sample_id(sample_id),
                            audio_path=audio_path,
                            subset=subset,
                        )
                    )
    return out


def _select_samples(
    samples: list[LibriSample],
    max_samples: int,
    sampling: str,
) -> list[LibriSample]:
    if sampling == "sequential":
        return samples[:max_samples]
    if sampling != "speaker_round_robin":
        raise ValueError(f"Unsupported sampling strategy: {sampling}")

    by_speaker: dict[str, list[LibriSample]] = {}
    for sample in samples:
        by_speaker.setdefault(sample.speaker_id, []).append(sample)

    speakers = sorted(by_speaker)
    out: list[LibriSample] = []
    idx = 0
    while len(out) < max_samples:
        added = False
        for speaker in speakers:
            group = by_speaker[speaker]
            if idx < len(group):
                out.append(group[idx])
                added = True
                if len(out) >= max_samples:
                    break
        if not added:
            break
        idx += 1
    return out


def _read_audio_path(path: Path) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "soundfile is required for reference parity suite. "
            "Install with: pip install 'mlx-qwen3-asr[eval]'"
        ) from exc
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1).astype(np.float32)
    return audio.astype(np.float32), int(sr)


def _resample_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    return np.array(load_audio((audio, sr))).astype(np.float32)


def _trim_eos(tokens: list[int]) -> list[int]:
    out = list(tokens)
    while out and out[-1] in EOS_IDS:
        out.pop()
    return out


def _first_mismatch(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return n
    return -1


def _build_long_mixes(
    base_samples: list[SuiteSample],
    long_mixes: int,
    long_mix_segments: int,
    silence_sec: float,
) -> list[SuiteSample]:
    if not base_samples or long_mixes <= 0:
        return []
    out: list[SuiteSample] = []
    silence = np.zeros(int(round(16000 * silence_sec)), dtype=np.float32)
    cursor = 0

    for i in range(long_mixes):
        segments: list[SuiteSample] = []
        for _ in range(long_mix_segments):
            segments.append(base_samples[cursor % len(base_samples)])
            cursor += 1

        audio_parts: list[np.ndarray] = []
        source_ids: list[str] = []
        speaker_tags: list[str] = []
        for j, segment in enumerate(segments):
            if segment.audio is None:
                continue
            audio_parts.append(segment.audio.astype(np.float32))
            source_ids.append(segment.sample_id)
            speaker_tags.append(segment.speaker_id)
            if j != len(segments) - 1 and silence.size > 0:
                audio_parts.append(silence)

        if not audio_parts:
            continue
        mixed = np.concatenate(audio_parts).astype(np.float32)
        speaker_hint = ",".join(sorted(set(speaker_tags)))[:80]
        out.append(
            SuiteSample(
                sample_id=f"longmix-{i:03d}",
                subset="synthetic-longmix",
                speaker_id=speaker_hint or "mixed",
                language="English",
                audio_path=None,
                source_sample_ids=source_ids,
                audio=mixed,
                sample_rate=16000,
            )
        )
    return out


def _parse_manifest(path: Path) -> list[SuiteSample]:
    rows: list[SuiteSample] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        row = line.strip()
        if not row:
            continue
        obj = json.loads(row)
        audio_path = Path(obj["audio_path"]).expanduser().resolve()
        rows.append(
            SuiteSample(
                sample_id=str(obj.get("sample_id", f"manifest-{i:05d}")),
                subset=str(obj.get("subset", "manifest")),
                speaker_id=str(obj.get("speaker_id", "unknown")),
                language=obj.get("language"),
                audio_path=audio_path,
                source_sample_ids=[],
                audio=None,
            )
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Run multi-sample token parity suite.")
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument(
        "--dtype",
        choices=["float16", "float32", "bfloat16"],
        default="float16",
    )
    parser.add_argument(
        "--subsets",
        default="test-clean,test-other",
        help="Comma-separated LibriSpeech subsets.",
    )
    parser.add_argument("--samples-per-subset", type=int, default=5)
    parser.add_argument(
        "--sampling",
        choices=["speaker_round_robin", "sequential"],
        default="speaker_round_robin",
    )
    parser.add_argument(
        "--data-dir",
        default=str(Path.home() / ".cache" / "mlx-qwen3-asr" / "datasets"),
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--include-long-mixes", action="store_true")
    parser.add_argument("--long-mixes", type=int, default=2)
    parser.add_argument("--long-mix-segments", type=int, default=4)
    parser.add_argument("--long-mix-silence-sec", type=float, default=0.3)
    parser.add_argument(
        "--manifest-jsonl",
        default=None,
        help=(
            "Optional JSONL with fields: audio_path, optional "
            "sample_id/subset/speaker_id/language"
        ),
    )
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--fail-match-rate-below", type=float, default=None)
    args = parser.parse_args()

    try:
        import qwen_asr
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Reference parity suite requires torch and qwen-asr. "
            "Install with: pip install 'mlx-qwen3-asr[aligner]'"
        ) from exc

    started = time.perf_counter()
    data_dir = Path(args.data_dir).expanduser().resolve()
    dtype = _dtype_from_name(args.dtype)

    suites: list[SuiteSample] = []
    selected_for_long: list[SuiteSample] = []
    for subset in [s.strip() for s in args.subsets.split(",") if s.strip()]:
        if subset not in SPLIT_ARCHIVES:
            raise ValueError(f"Unsupported subset '{subset}'.")
        root = _ensure_split(data_dir, subset)
        all_samples = _collect_librispeech_samples(root, subset)
        selected = _select_samples(all_samples, max(1, args.samples_per_subset), args.sampling)
        for sample in selected:
            audio, sr = _read_audio_path(sample.audio_path)
            audio16 = _resample_16k(audio, sr)
            row = SuiteSample(
                sample_id=sample.sample_id,
                subset=subset,
                speaker_id=sample.speaker_id,
                language="English",
                audio_path=sample.audio_path,
                source_sample_ids=[],
                audio=audio16,
                sample_rate=16000,
            )
            suites.append(row)
            selected_for_long.append(row)

    if args.include_long_mixes:
        suites.extend(
            _build_long_mixes(
                selected_for_long,
                long_mixes=max(0, args.long_mixes),
                long_mix_segments=max(2, args.long_mix_segments),
                silence_sec=max(0.0, args.long_mix_silence_sec),
            )
        )

    if args.manifest_jsonl:
        manifest_rows = _parse_manifest(Path(args.manifest_jsonl).expanduser().resolve())
        for row in manifest_rows:
            if row.audio_path is None:
                continue
            audio, sr = _read_audio_path(row.audio_path)
            suites.append(
                SuiteSample(
                    sample_id=row.sample_id,
                    subset=row.subset,
                    speaker_id=row.speaker_id,
                    language=row.language,
                    audio_path=row.audio_path,
                    source_sample_ids=row.source_sample_ids,
                    audio=_resample_16k(audio, sr),
                    sample_rate=16000,
                )
            )

    if not suites:
        raise RuntimeError("No samples selected for parity suite.")

    ref = qwen_asr.Qwen3ASRModel.from_pretrained(
        args.model,
        dtype=torch.float16,
        device_map="cpu",
        max_new_tokens=args.max_new_tokens,
    )
    mlx_model, _ = load_model(args.model, dtype=dtype)

    rows: list[dict] = []
    match_count = 0
    for i, sample in enumerate(suites, start=1):
        audio = sample.audio
        if audio is None:
            continue
        t0 = time.perf_counter()
        prompt = ref._build_text_prompt(context="", force_language=sample.language)  # noqa: SLF001
        inputs = ref.processor(
            text=[prompt],
            audio=[audio],
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        inputs = inputs.to(ref.model.device).to(ref.model.dtype)

        with torch.no_grad():
            ref_out = ref.model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        ref_sec = time.perf_counter() - t0

        prompt_len = int(inputs["input_ids"].shape[1])
        ref_tokens = _trim_eos(ref_out.sequences[0, prompt_len:].tolist())

        t1 = time.perf_counter()
        input_ids = mx.array(inputs["input_ids"].cpu().numpy().astype(np.int32))
        mel = mx.array(inputs["input_features"].cpu().numpy().astype(np.float32))
        feature_lens = mx.array(
            inputs["feature_attention_mask"].sum(-1).cpu().numpy().astype(np.int32)
        )

        audio_features, _ = mlx_model.audio_tower(mel.astype(dtype), feature_lens)
        seq_len = input_ids.shape[1]
        pos = mx.arange(seq_len)[None, :]
        position_ids = mx.stack([pos, pos, pos], axis=1)

        mlx_tokens = _trim_eos(
            generate(
                model=mlx_model,
                input_ids=input_ids,
                audio_features=audio_features,
                position_ids=position_ids,
                config=GenerationConfig(
                    max_new_tokens=args.max_new_tokens,
                    temperature=0.0,
                ),
            )
        )
        mlx_sec = time.perf_counter() - t1

        is_match = mlx_tokens == ref_tokens
        if is_match:
            match_count += 1

        mismatch_idx = _first_mismatch(mlx_tokens, ref_tokens)
        rows.append(
            {
                "index": i,
                "sample_id": sample.sample_id,
                "subset": sample.subset,
                "speaker_id": sample.speaker_id,
                "language": sample.language,
                "audio_path": str(sample.audio_path) if sample.audio_path else None,
                "source_sample_ids": sample.source_sample_ids,
                "duration_sec": len(audio) / 16000.0,
                "token_count_mlx": len(mlx_tokens),
                "token_count_ref": len(ref_tokens),
                "text_match": is_match,
                "first_mismatch_index": mismatch_idx,
                "latency_sec_mlx": mlx_sec,
                "latency_sec_ref": ref_sec,
            }
        )

    total = len(rows)
    by_subset: dict[str, dict[str, float]] = {}
    for row in rows:
        b = by_subset.setdefault(
            row["subset"],
            {
                "samples": 0,
                "matches": 0,
                "match_rate": 0.0,
                "latency_sec_mlx_mean": 0.0,
                "latency_sec_ref_mean": 0.0,
            },
        )
        b["samples"] += 1
        b["matches"] += 1 if row["text_match"] else 0
        b["latency_sec_mlx_mean"] += row["latency_sec_mlx"]
        b["latency_sec_ref_mean"] += row["latency_sec_ref"]

    for subset, stats in by_subset.items():
        n = max(1, int(stats["samples"]))
        stats["match_rate"] = float(stats["matches"]) / float(n)
        stats["latency_sec_mlx_mean"] = float(stats["latency_sec_mlx_mean"]) / float(n)
        stats["latency_sec_ref_mean"] = float(stats["latency_sec_ref_mean"]) / float(n)
        by_subset[subset] = stats

    match_rate = float(match_count) / float(max(1, total))
    latency_mlx_mean = float(np.mean([r["latency_sec_mlx"] for r in rows])) if rows else 0.0
    latency_ref_mean = float(np.mean([r["latency_sec_ref"] for r in rows])) if rows else 0.0
    payload = {
        "suite": "reference-parity-suite-v1",
        "model": args.model,
        "dtype": args.dtype,
        "max_new_tokens": args.max_new_tokens,
        "subsets": [s.strip() for s in args.subsets.split(",") if s.strip()],
        "samples_per_subset": args.samples_per_subset,
        "include_long_mixes": bool(args.include_long_mixes),
        "long_mixes": args.long_mixes if args.include_long_mixes else 0,
        "manifest_jsonl": args.manifest_jsonl,
        "samples": total,
        "matches": match_count,
        "match_rate": match_rate,
        "latency_sec_mlx_mean": latency_mlx_mean,
        "latency_sec_ref_mean": latency_ref_mean,
        "elapsed_sec": time.perf_counter() - started,
        "by_subset": by_subset,
        "rows": rows,
    }

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.json_output:
        out = Path(args.json_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.fail_match_rate_below is not None and match_rate < args.fail_match_rate_below:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
