# Non-near 5: encoder clamp parity recheck

- Date: 2026-02-15
- Candidate artifact: `docs/benchmarks/2026-02-15-logit-parity-probe-nonnear5-post-enc-clamp.json`
- Baseline rows source: `docs/benchmarks/2026-02-15-logit-parity-probe-multilingual100-mismatches.json`

## Summary
- All tracked mismatch probe fields unchanged: **True**
- On this non-near subset, adding fp16 encoder-layer clamp did not change mismatch index or top token identity at probe step.

## Rows
| sample_id | mismatch_idx base->new | mlx_token base->new | near_tie base->new |
|---|---|---|---|
| en_us-394135166243682296 | 17 -> 17 | 220 -> 220 | False -> False |
| ja_jp-8922261396806111795 | 7 -> 7 | 11387 -> 11387 | False -> False |
| de_de-11906634980733046933 | 29 -> 29 | 58534 -> 58534 | False -> False |
| ar_eg-14219101187915533421 | 29 -> 29 | 86941 -> 86941 | False -> False |
| hi_in-17876469696694013955 | 61 -> 61 | 105 -> 105 | False -> False |
