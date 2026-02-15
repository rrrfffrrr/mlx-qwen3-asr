# Logit Parity Probe: multilingual100 mismatch rows

- Date: 2026-02-15
- Source parity artifact: `docs/benchmarks/2026-02-15-reference-parity-suite-multilingual100-post-langcanon.json`
- Probe artifact (JSON): `docs/benchmarks/2026-02-15-logit-parity-probe-multilingual100-mismatches.json`

## Summary
- Mismatch rows probed: **36**
- Near-tie rows (`ref_margin<=0.5` OR `mlx_margin<=0.5`): **31/36 (86.1%)**
- Strict top1/top2 cross-swaps: **30/36 (83.3%)**
- Interpretation: most mismatches are decode-boundary flips between the same two candidate tokens, not broad top-k divergence.

## Per-language breakdown
| language | mismatches | near-tie rows | strict top1/top2 swaps |
|---|---:|---:|---:|
| Arabic | 6 | 5 | 6 |
| Chinese | 3 | 3 | 2 |
| English | 2 | 1 | 2 |
| French | 6 | 6 | 4 |
| German | 4 | 3 | 3 |
| Hindi | 6 | 5 | 6 |
| Japanese | 4 | 3 | 2 |
| Korean | 1 | 1 | 1 |
| Russian | 1 | 1 | 1 |
| Spanish | 3 | 3 | 3 |

## Non-near-tie rows (highest triage priority)
| sample_id | language | mismatch_index | ref_margin | mlx_margin |
|---|---|---:|---:|---:|
| en_us-394135166243682296 | English | 17 | 0.718750 | 1.391224 |
| ja_jp-8922261396806111795 | Japanese | 7 | 1.453125 | 0.507343 |
| de_de-11906634980733046933 | German | 29 | 2.671875 | 3.197556 |
| ar_eg-14219101187915533421 | Arabic | 29 | 0.718750 | 1.640415 |
| hi_in-17876469696694013955 | Hindi | 61 | 1.218750 | 0.706951 |

## Recommended next action
- Do **not** force token-level parity hacks in postprocessing.
- Focus next investigation on model-path/logit-source parity for the five non-near rows first; if those collapse, multilingual token parity should rise materially without harming quality metrics.
