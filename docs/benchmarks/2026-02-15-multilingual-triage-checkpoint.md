# Multilingual Triage Checkpoint (Post Language-Canonicalization)

## Quality (Manifest multilingual-100)
- WER delta: +0.000000
- CER delta: +0.000000
- Primary error delta: +0.000000
- Latency mean delta: +0.367s

## Parity (Multilingual-100)
- Token match delta: +0.0000
- Text match delta: +0.0000
- MLX latency mean delta: +0.335s

## Precision Sensitivity (Multilingual-20)
- FP32-FP16 token match delta: +0.0000
- FP32-FP16 text match delta: +0.0000
- FP32-FP16 MLX latency mean delta: +0.026s

Conclusion: no measurable quality/parity gain from this change on current multilingual lanes; precision increase to FP32 does not improve match rates.
