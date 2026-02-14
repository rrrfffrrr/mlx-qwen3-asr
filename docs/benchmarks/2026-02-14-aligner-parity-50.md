# Forced Aligner Parity (50 samples)

- Suite: `aligner-parity-v1`
- Subset: `test-clean`
- Samples: `50`
- Model: `Qwen/Qwen3-ForcedAligner-0.6B`
- Language: `English`

| Metric | Value |
|---|---:|
| Text match rate | 1.0000 |
| Timing MAE start (ms) | 4.2579 |
| Timing MAE end (ms) | 7.1238 |
| Timing MAE all (ms) | 5.6909 |
| MLX mean latency (s) | 0.2113 |
| qwen_asr mean latency (s) | 0.5570 |
| Relative speed (qwen_asr / mlx) | 2.64x |

Gate result for this run:
- `--fail-text-match-rate-below 1.0`: pass
- `--fail-timing-mae-ms-above 60.0`: pass
