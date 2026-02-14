# Benchmarks Directory

Store benchmark JSON outputs here.

Suggested naming:

- `baseline_<machine>_<model>_<dtype>.json`
- `pr<id>_<machine>_<model>_<dtype>.json`
- `latest.json` for quick local runs
- `nightly-latency.json` for scheduled runtime trend
- `nightly-librispeech.json` for scheduled quality trend

Generate with:

```bash
python scripts/benchmark_asr.py tests/fixtures/test_speech.wav \
  --model Qwen/Qwen3-ASR-0.6B \
  --runs 5 \
  --json-output docs/benchmarks/latest.json
```
