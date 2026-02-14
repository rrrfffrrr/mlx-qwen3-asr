# Speculative Decode Prototype Benchmark (2026-02-14)

Setup:
- Machine: Apple Silicon local dev machine
- Target model: `Qwen/Qwen3-ASR-1.7B`
- Draft model: `Qwen/Qwen3-ASR-0.6B`
- Dtype: `float16`
- Script: `scripts/benchmark_speculative.py`

## Results

Short fixture (`tests/fixtures/test_speech.wav`, ~2.53s), warmup=1 runs=3:
- Baseline mean: `1.4543s`
- Speculative mean: `2.7206s`
- Relative speed: `0.53x` (slower)
- Text parity: `true`
- Artifact: `docs/benchmarks/2026-02-14-speculative-1p7b-vs-0p6b.json`

Synthetic 10s fixture (`/tmp/test_speech_10s.wav`), warmup=1 runs=3:
- Baseline mean: `2.6755s`
- Speculative mean: `4.8957s`
- Relative speed: `0.55x` (slower)
- Text parity: `true`
- Artifact: `docs/benchmarks/2026-02-14-speculative-1p7b-vs-0p6b-10s.json`

## Decision

Keep speculative decoding as an experimental opt-in path only.

Rationale:
- Correctness target (greedy parity) is met in this prototype.
- Current end-to-end implementation is slower on tested workloads due extra
  draft-model compute overhead (including draft audio encoder pass).
- Do not switch defaults until benchmark evidence shows net speed wins.
