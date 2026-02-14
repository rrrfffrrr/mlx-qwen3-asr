# Quantization Matrix

- Source model: `Qwen/Qwen3-ASR-0.6B`
- Eval samples: `20`
- Benchmark runs: `7`
- Long clip length: `10s`

| Config | Short Mean (s) | Short RTF | Long Mean (s) | Long RTF | WER | CER | Eval RTF |
|---|---:|---:|---:|---:|---:|---:|---:|
| fp16 | 0.5655 | 0.2232 | 0.9469 | 0.0947 | 0.007317 | 0.002195 | 0.1483 |
| 4bit-g64 | 0.1635 | 0.0646 | 0.2789 | 0.0279 | 0.007317 | 0.001647 | 0.0516 |
| 4bit-g32 | 0.1720 | 0.0679 | 0.2875 | 0.0288 | 0.012195 | 0.003293 | 0.0568 |
| 8bit-g64 | 0.1821 | 0.0719 | 0.3089 | 0.0309 | 0.007317 | 0.002195 | 0.0557 |

## Relative to fp16

- `4bit-g64` long-clip speedup vs fp16: `3.40x` (WER delta `+0.000000`)
- `4bit-g32` long-clip speedup vs fp16: `3.29x` (WER delta `+0.004878`)
- `8bit-g64` long-clip speedup vs fp16: `3.06x` (WER delta `+0.000000`)
