# Benchmarks Directory

Store benchmark JSON outputs here.

Suggested naming:

- `baseline_<machine>_<model>_<dtype>.json`
- `pr<id>_<machine>_<model>_<dtype>.json`
- `latest.json` for quick local runs
- `nightly-latency.json` for scheduled runtime trend
- `nightly-librispeech.json` for scheduled quality trend

## Recorded Results

Tokenizer cache optimization benchmark (2026-02-14):

- Machine: Apple M4 Pro, macOS 26.2
- Model: `Qwen/Qwen3-ASR-0.6B`, dtype `float16`
- Audio: `tests/fixtures/test_speech.wav`
- Runs: warmup=1, measured=3

Artifacts:

- `docs/benchmarks/2026-02-14-tokenizer-cache-before.json`
- `docs/benchmarks/2026-02-14-tokenizer-cache-after.json`

Generate with:

```bash
python scripts/benchmark_asr.py tests/fixtures/test_speech.wav \
  --model Qwen/Qwen3-ASR-0.6B \
  --runs 5 \
  --json-output docs/benchmarks/latest.json
```
