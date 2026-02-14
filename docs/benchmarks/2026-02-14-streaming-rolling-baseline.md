# Streaming Rolling Baseline (2026-02-14)

- Model: `Qwen/Qwen3-ASR-0.6B` (`float16`)
- Audio: `tests/fixtures/test_speech.wav` (`2.53s`)
- Settings: `chunk_size_sec=2.0`, `max_context_sec=30.0`, `unfixed_chunk_num=2`, `unfixed_token_num=5`
- Runs: `warmup=1`, `measured=3`

| Metric | Value |
|---|---:|
| Total latency mean | `1.1344s` |
| Total latency median | `1.1553s` |
| Total latency min/max | `1.0235s` / `1.2243s` |
| Per-chunk mean latency (mean over runs) | `0.2673s` |
| Per-chunk p95 latency (mean over runs) | `0.5079s` |
| Finish latency mean | `0.5997s` |
| RTF | `0.4478` |

Raw artifact: `docs/benchmarks/2026-02-14-streaming-rolling-baseline.json`
