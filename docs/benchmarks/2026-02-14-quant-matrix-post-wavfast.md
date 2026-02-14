# Quantization Matrix

- Source model: `Qwen/Qwen3-ASR-0.6B`
- Eval samples: `20`
- Benchmark runs: `7`
- Long clip length: `10s`

| Config | Short Mean (s) | Short RTF | Long Mean (s) | Long RTF | WER | CER | Eval RTF |
|---|---:|---:|---:|---:|---:|---:|---:|
| fp16 | 0.4996 | 0.1972 | 0.9088 | 0.0909 | 0.007317 | 0.002195 | 0.1446 |
| 4bit-g64 | 0.1187 | 0.0469 | 0.2286 | 0.0229 | 0.007317 | 0.001647 | 0.0499 |
| 4bit-g32 | 0.1305 | 0.0515 | 0.2409 | 0.0241 | 0.012195 | 0.003293 | 0.0519 |
| 8bit-g64 | 0.1446 | 0.0571 | 0.2627 | 0.0263 | 0.007317 | 0.002195 | 0.0573 |

## Relative to fp16

- `4bit-g64` long-clip speedup vs fp16: `3.98x` (WER delta `+0.000000`)
- `4bit-g32` long-clip speedup vs fp16: `3.77x` (WER delta `+0.004878`)
- `8bit-g64` long-clip speedup vs fp16: `3.46x` (WER delta `+0.000000`)
