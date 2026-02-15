#!/usr/bin/env python3
"""Quality gate runner for mlx-qwen3-asr.

Usage:
  python scripts/quality_gate.py --mode fast
  RUN_REFERENCE_PARITY=1 python scripts/quality_gate.py --mode release
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

MYPY_TYPED_TARGETS = [
    "mlx_qwen3_asr/config.py",
    "mlx_qwen3_asr/chunking.py",
    "mlx_qwen3_asr/attention.py",
    "mlx_qwen3_asr/encoder.py",
    "mlx_qwen3_asr/decoder.py",
    "mlx_qwen3_asr/model.py",
]


@dataclass
class StepResult:
    name: str
    cmd: str
    passed: bool
    duration_sec: float
    returncode: int
    note: str = ""


def _run(cmd: list[str], cwd: Path) -> StepResult:
    started = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(cwd), check=False)
    elapsed = time.perf_counter() - started
    return StepResult(
        name=cmd[0],
        cmd=" ".join(shlex.quote(c) for c in cmd),
        passed=proc.returncode == 0,
        duration_sec=elapsed,
        returncode=proc.returncode,
    )


def _tracked_py_files(repo: Path) -> list[str]:
    proc = subprocess.run(
        ["git", "ls-files", "*.py"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=True,
    )
    tracked = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    return [path for path in tracked if (repo / path).exists()]


def run_gate(mode: str, repo: Path, python_bin: str) -> tuple[list[StepResult], bool]:
    steps: list[StepResult] = []

    tracked = _tracked_py_files(repo)
    if not tracked:
        steps.append(
            StepResult(
                name="ruff",
                cmd="",
                passed=False,
                duration_sec=0.0,
                returncode=1,
                note="No tracked Python files found.",
            )
        )
        return steps, False

    steps.append(_run([python_bin, "-m", "ruff", "check", *tracked], repo))
    steps.append(
        _run(
            [
                python_bin,
                "-m",
                "mypy",
                "--follow-imports=skip",
                "--ignore-missing-imports",
                *MYPY_TYPED_TARGETS,
            ],
            repo,
        )
    )
    steps.append(_run([python_bin, "-m", "pytest", "-q"], repo))

    if mode == "release":
        if not (repo / "tests" / "test_reference_parity.py").exists():
            steps.append(
                StepResult(
                    name="reference-parity",
                    cmd="pytest -q tests/test_reference_parity.py",
                    passed=False,
                    duration_sec=0.0,
                    returncode=1,
                    note="Missing tests/test_reference_parity.py",
                )
            )
        else:
            steps.append(
                _run([python_bin, "-m", "pytest", "-q", "tests/test_reference_parity.py"], repo)
            )

        if os.environ.get("RUN_QUALITY_EVAL", "1") == "1":
            quality_eval_cmd = [
                python_bin,
                str(repo / "scripts" / "eval_librispeech.py"),
                "--model",
                os.environ.get("QUALITY_EVAL_MODEL", "Qwen/Qwen3-ASR-0.6B"),
                "--dtype",
                os.environ.get("QUALITY_EVAL_DTYPE", "float16"),
                "--subset",
                os.environ.get("QUALITY_EVAL_SUBSET", "test-clean"),
                "--samples",
                os.environ.get("QUALITY_EVAL_SAMPLES", "20"),
                "--sampling",
                os.environ.get("QUALITY_EVAL_SAMPLING", "speaker_round_robin"),
                "--max-new-tokens",
                os.environ.get("QUALITY_EVAL_MAX_NEW_TOKENS", "256"),
                "--fail-wer-above",
                os.environ.get("QUALITY_EVAL_FAIL_WER_ABOVE", "0.10"),
                "--fail-cer-above",
                os.environ.get("QUALITY_EVAL_FAIL_CER_ABOVE", "0.06"),
            ]
            quality_json = os.environ.get("QUALITY_EVAL_JSON_OUTPUT")
            if quality_json:
                quality_eval_cmd.extend(["--json-output", quality_json])
            steps.append(_run(quality_eval_cmd, repo))

        if os.environ.get("RUN_MANIFEST_QUALITY_EVAL", "0") == "1":
            manifest_jsonl = os.environ.get("MANIFEST_QUALITY_EVAL_JSONL")
            if not manifest_jsonl:
                steps.append(
                    StepResult(
                        name="manifest-quality-eval",
                        cmd="scripts/eval_manifest_quality.py --manifest-jsonl <path>",
                        passed=False,
                        duration_sec=0.0,
                        returncode=1,
                        note="RUN_MANIFEST_QUALITY_EVAL=1 requires MANIFEST_QUALITY_EVAL_JSONL",
                    )
                )
            else:
                manifest_quality_cmd = [
                    python_bin,
                    str(repo / "scripts" / "eval_manifest_quality.py"),
                    "--manifest-jsonl",
                    manifest_jsonl,
                    "--model",
                    os.environ.get("MANIFEST_QUALITY_EVAL_MODEL", "Qwen/Qwen3-ASR-0.6B"),
                    "--dtype",
                    os.environ.get("MANIFEST_QUALITY_EVAL_DTYPE", "float16"),
                    "--max-new-tokens",
                    os.environ.get("MANIFEST_QUALITY_EVAL_MAX_NEW_TOKENS", "1024"),
                    "--fail-primary-above",
                    os.environ.get("MANIFEST_QUALITY_EVAL_FAIL_PRIMARY_ABOVE", "0.35"),
                ]
                fail_wer = os.environ.get("MANIFEST_QUALITY_EVAL_FAIL_WER_ABOVE")
                if fail_wer:
                    manifest_quality_cmd.extend(["--fail-wer-above", fail_wer])
                fail_cer = os.environ.get("MANIFEST_QUALITY_EVAL_FAIL_CER_ABOVE")
                if fail_cer:
                    manifest_quality_cmd.extend(["--fail-cer-above", fail_cer])
                limit = os.environ.get("MANIFEST_QUALITY_EVAL_LIMIT")
                if limit:
                    manifest_quality_cmd.extend(["--limit", limit])
                quality_json = os.environ.get("MANIFEST_QUALITY_EVAL_JSON_OUTPUT")
                if quality_json:
                    manifest_quality_cmd.extend(["--json-output", quality_json])
                steps.append(_run(manifest_quality_cmd, repo))

        if os.environ.get("RUN_ALIGNER_PARITY") == "1":
            samples = os.environ.get("ALIGNER_PARITY_SAMPLES", "10")
            steps.append(
                _run(
                    [
                        python_bin,
                        str(repo / "scripts" / "eval_aligner_parity.py"),
                        "--subset",
                        "test-clean",
                        "--samples",
                        samples,
                        "--model",
                        "Qwen/Qwen3-ForcedAligner-0.6B",
                        "--fail-text-match-rate-below",
                        "1.0",
                        "--fail-timing-mae-ms-above",
                        "60.0",
                    ],
                    repo,
                )
            )

        if os.environ.get("RUN_REFERENCE_PARITY_SUITE") == "1":
            subsets = os.environ.get(
                "REFERENCE_PARITY_SUITE_SUBSETS",
                "test-clean,test-other",
            )
            samples_per_subset = os.environ.get(
                "REFERENCE_PARITY_SUITE_SAMPLES_PER_SUBSET",
                "3",
            )
            max_new_tokens = os.environ.get("REFERENCE_PARITY_SUITE_MAX_NEW_TOKENS", "128")
            fail_match_rate = os.environ.get(
                "REFERENCE_PARITY_SUITE_FAIL_MATCH_RATE_BELOW",
                "1.0",
            )
            fail_text_match_rate = os.environ.get(
                "REFERENCE_PARITY_SUITE_FAIL_TEXT_MATCH_RATE_BELOW",
            )
            cmd = [
                python_bin,
                str(repo / "scripts" / "eval_reference_parity_suite.py"),
                "--model",
                os.environ.get("REFERENCE_PARITY_SUITE_MODEL", "Qwen/Qwen3-ASR-0.6B"),
                "--subsets",
                subsets,
                "--samples-per-subset",
                samples_per_subset,
                "--max-new-tokens",
                max_new_tokens,
                "--fail-match-rate-below",
                fail_match_rate,
            ]
            manifest_jsonl = os.environ.get("REFERENCE_PARITY_SUITE_MANIFEST_JSONL")
            if manifest_jsonl:
                cmd.extend(["--manifest-jsonl", manifest_jsonl])
            if os.environ.get("REFERENCE_PARITY_SUITE_INCLUDE_LONG_MIXES", "1") == "1":
                cmd.append("--include-long-mixes")
                cmd.extend(
                    [
                        "--long-mixes",
                        os.environ.get("REFERENCE_PARITY_SUITE_LONG_MIXES", "2"),
                        "--long-mix-segments",
                        os.environ.get("REFERENCE_PARITY_SUITE_LONG_MIX_SEGMENTS", "4"),
                        "--long-mix-silence-sec",
                        os.environ.get("REFERENCE_PARITY_SUITE_LONG_MIX_SILENCE_SEC", "0.3"),
                    ]
                )
            if os.environ.get("REFERENCE_PARITY_SUITE_INCLUDE_NOISE_VARIANTS", "0") == "1":
                cmd.append("--include-noise-variants")
                cmd.extend(
                    [
                        "--noise-snrs-db",
                        os.environ.get("REFERENCE_PARITY_SUITE_NOISE_SNRS_DB", "10,5"),
                        "--noise-seed",
                        os.environ.get("REFERENCE_PARITY_SUITE_NOISE_SEED", "20260214"),
                    ]
                )
            if fail_text_match_rate:
                cmd.extend(["--fail-text-match-rate-below", fail_text_match_rate])
            json_output = os.environ.get("REFERENCE_PARITY_SUITE_JSON_OUTPUT")
            if json_output:
                cmd.extend(["--json-output", json_output])
            steps.append(_run(cmd, repo))

    ok = all(step.passed for step in steps)
    return steps, ok


def _resolve_python(repo: Path) -> str:
    venv_py = repo / ".venv" / "bin" / "python"
    if venv_py.exists():
        return str(venv_py)
    return sys.executable


def main() -> int:
    parser = argparse.ArgumentParser(description="Run repository quality gate checks.")
    parser.add_argument(
        "--mode",
        choices=["fast", "release"],
        default="fast",
        help="fast: lint+tests, release: fast + reference parity test",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional path to write gate result JSON.",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    python_bin = _resolve_python(repo)

    started = time.perf_counter()
    steps, ok = run_gate(args.mode, repo, python_bin)
    total = time.perf_counter() - started

    print(f"\nQuality Gate ({args.mode})")
    print("=" * 32)
    for step in steps:
        status = "PASS" if step.passed else "FAIL"
        print(f"[{status}] {step.cmd} ({step.duration_sec:.2f}s)")
        if step.note:
            print(f"       note: {step.note}")
    print(f"Total: {total:.2f}s")
    print(f"Result: {'PASS' if ok else 'FAIL'}")

    if args.json_output:
        out = Path(args.json_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mode": args.mode,
            "passed": ok,
            "total_duration_sec": total,
            "steps": [asdict(s) for s in steps],
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
