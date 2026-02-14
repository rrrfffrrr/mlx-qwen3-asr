# Reference Parity Suite (Smoke)

- Suite: `reference-parity-suite-v1`
- Model: `Qwen/Qwen3-ASR-0.6B`
- Subsets: `test-clean`, `test-other`
- Samples per subset: `2`
- Synthetic long mixes: `1` (4 segments, mixed speakers)

## Summary

| Metric | Value |
|---|---:|
| Total samples | 5 |
| Token-exact matches | 2 |
| Match rate | 0.4000 |
| Mean latency (MLX) | 2.4347s |
| Mean latency (reference) | 10.8651s |

## By Subset

| Subset | Samples | Match rate | MLX mean latency | Reference mean latency |
|---|---:|---:|---:|---:|
| test-clean | 2 | 0.5000 | 1.2919s | 5.4993s |
| test-other | 2 | 0.5000 | 1.8482s | 7.9446s |
| synthetic-longmix | 1 | 0.0000 | 5.8936s | 27.4380s |

## Interpretation

- The single-fixture reference parity test is not sufficient to claim broad
  token-exact parity.
- This smoke run found token-level divergences on harder and longer samples.
- Keep this suite non-blocking for now, use it as a gap-finding lane, and
  prioritize debugging first-mismatch patterns before promoting to required gate.
