# Tokenizer Loader Path Benchmark

- Suite: `tokenizer-loader-path-benchmark-v1`
- Audio: `tests/fixtures/test_speech.wav`
- Model: `Qwen/Qwen3-ASR-0.6B`
- Dtype: `float16`
- Runs per mode: `5`

| Loader path | Mean latency (s) | Median (s) | Min (s) | Max (s) |
|---|---:|---:|---:|---:|
| Legacy `AutoTokenizer` | 4.0081 | 3.9910 | 3.9508 | 4.1215 |
| Direct `Qwen2Tokenizer` | 2.5986 | 2.5910 | 2.5735 | 2.6367 |

Relative speedup (legacy / direct): **1.54x**

Benchmark command pattern:
- Spawn fresh Python process per run.
- Measure end-to-end single `transcribe(...)` latency with tokenizer cache cleared each process.
