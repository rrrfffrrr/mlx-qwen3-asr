# MLX vs PyTorch Quality Head-to-Head (test-other, n=100)

- model: `Qwen/Qwen3-ASR-0.6B`
- samples: `100`
- MLX WER: `0.0420`
- PyTorch WER: `0.0441`
- Delta WER (MLX-Ref): `-0.0021`
- MLX CER: `0.0209`
- PyTorch CER: `0.0214`
- Delta CER (MLX-Ref): `-0.0005`

| System | WER | CER | Mean latency (s) |
|---|---:|---:|---:|
| MLX | 0.0420 | 0.0209 | 0.7079 |
| PyTorch ref | 0.0441 | 0.0214 | 2.1404 |
