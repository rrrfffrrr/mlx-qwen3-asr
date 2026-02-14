# Forced Aligner Parity (10 samples, tokenizer parity refresh)

- Suite: `aligner-parity-v1`
- Subset: `test-clean`
- Samples: `10`
- Model: `Qwen/Qwen3-ForcedAligner-0.6B`
- Language: `English`

| Metric | Value |
|---|---:|
| Text match rate | 1.0000 |
| Timing MAE start (ms) | 0.4762 |
| Timing MAE end (ms) | 2.8571 |
| Timing MAE all (ms) | 1.6667 |
| MLX mean latency (s) | 0.5267 |
| qwen_asr mean latency (s) | 0.7731 |
| Relative speed (qwen_asr / mlx) | 1.47x |

Gate result for this run:
- `--fail-text-match-rate-below 1.0`: pass
- `--fail-timing-mae-ms-above 60.0`: pass
