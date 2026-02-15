# Audio Encoder Layerwise Parity (non-near 5)

- Date: 2026-02-15
- Artifact (JSON): `docs/benchmarks/2026-02-15-audio-encoder-layerwise-parity-nonnear5.json`

## Summary
- Early encoder layers are very close to reference, then cosine drift accumulates with depth.
- The worst-case sample in this subset is Japanese, where layerwise cosine drops notably by layer 18.
- This pattern supports a numeric accumulation/parity hypothesis inside encoder layers rather than a single-step decode/cache bug.

## Per-sample trend
| sample_id | lang | layer1 cosine | layer6 cosine | layer12 cosine | layer18 cosine | final cosine |
|---|---|---:|---:|---:|---:|---:|
| en_us-394135166243682296 | English | 0.998427 | 0.993330 | 0.985716 | 0.974796 | 0.951899 |
| ja_jp-8922261396806111795 | Japanese | 0.988908 | 0.973104 | 0.960815 | 0.909952 | 0.893331 |
| de_de-11906634980733046933 | German | 0.999287 | 0.995308 | 0.986913 | 0.970679 | 0.948264 |
| ar_eg-14219101187915533421 | Arabic | 0.997699 | 0.989543 | 0.981934 | 0.954082 | 0.944079 |
| hi_in-17876469696694013955 | Hindi | 0.998628 | 0.995928 | 0.985511 | 0.975061 | 0.957047 |

## Interpretation
- Residual non-near token mismatches align with gradual encoder representation drift over depth.
- Next investigation should focus on high-sensitivity encoder numerics (attention softmax path, mask value conventions, and layerwise arithmetic parity) with targeted A/B checks.
