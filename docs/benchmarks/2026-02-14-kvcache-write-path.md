# KV Cache Write Path Experiment

Date: 2026-02-14

Compared `mx.slice_update` (baseline) vs in-place slice assignment for preallocated KV-cache writes in `KVCache.update()`.

| Scenario | Baseline Mean (s) | In-place Mean (s) | Delta |
|---|---:|---:|---:|
| fp16 short | 0.5754687163964263 | 0.5613519498991082 | -2.453090167214822% |
| fp16 long 10s | 0.9613731667504908 | 0.9476531353757309 | -1.4271285957704105% |
| 4bit-g64 short | 0.16850072529923638 | 0.15978719989798265 | -5.171209432944335% |
| 4bit-g64 long 10s (mean) | 0.2728349685003195 | 0.29595140087621985 | 8.472679474688839% |

Long 10s quantized median stayed essentially unchanged (baseline 0.2721912705019349, in-place 0.2719632505031768); one outlier inflated the mean in the in-place run.

Conclusion: keep in-place writes; they are neutral-to-better overall and match upstream MLX-LM cache style.

