# Benchmarking Protocol

Use this protocol for all runtime optimization work.

## Goal

Hold quality constant while improving runtime on Apple Silicon.

## Command

```bash
python scripts/benchmark_asr.py tests/fixtures/test_speech.wav \
  --model Qwen/Qwen3-ASR-0.6B \
  --dtype float16 \
  --warmup-runs 1 \
  --runs 5 \
  --json-output docs/benchmarks/latest.json
```

## Metrics

- `latency_sec.mean`
- `latency_sec.median`
- `rtf` (real-time factor = mean_latency / audio_duration)

Lower is better for all three.

## Method

1. Run benchmark on baseline commit.
2. Run benchmark on candidate commit with same machine/settings.
3. Compare JSON outputs.
4. If speed improves but quality gate fails, reject the optimization.

## Recommended Scenarios

- Short clip (latency sensitivity): `tests/fixtures/test_speech.wav`
- Long-form clip (stability/throughput): add a fixed long fixture and track same metrics.

## Reporting Template

Include this in PRs that claim performance gains:

```text
Machine: <chip + RAM + macOS>
Model: <id>, dtype=<dtype>
Before: mean=<x>s median=<y>s rtf=<z>
After:  mean=<x2>s median=<y2>s rtf=<z2>
Quality Gate: fast=<pass/fail>, release=<pass/fail or not run>
```

## Latest Local Finding (2026-02-14)

- Change: tokenizer instance caching across repeated `transcribe()` calls.
- Machine: Apple M4 Pro, macOS 26.2.
- Workload: `tests/fixtures/test_speech.wav`, model `Qwen/Qwen3-ASR-0.6B`, dtype `float16`, warmup=1, runs=3.
- Before: mean latency `1.7217s`, RTF `0.6796`.
- After: mean latency `0.5464s`, RTF `0.2157`.
- Delta: `-68.3%` mean latency, `-68.3%` RTF.

Raw JSON artifacts are tracked under `docs/benchmarks/`.
