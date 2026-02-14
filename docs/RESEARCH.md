# Research: Qwen3-ASR

Verified notes for this repo, based on official artifacts as of **2026-02-14**.

## Primary Sources

- Qwen3-ASR paper: https://arxiv.org/abs/2601.21337
- Official codebase: https://github.com/QwenLM/Qwen3-ASR
- Official Python package (`qwen-asr`): https://github.com/QwenLM/Qwen3-ASR/tree/main/qwen_asr
- Apple MLX examples: https://github.com/ml-explore/mlx-examples
- Apple MLX-LM cache implementation: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/cache.py
- MLX-Audio (existing MLX Qwen3-ASR path): https://github.com/ml-explore/mlx-audio
- Swift MLX implementation: https://github.com/ivan-digital/qwen3-asr-swift
- HF model cards/configs:
  - https://huggingface.co/Qwen/Qwen3-ASR-1.7B
  - https://huggingface.co/Qwen/Qwen3-ASR-0.6B
  - https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B

## Model Family Snapshot

| Model | Core Positioning | Notable Claim (official) |
|---|---|---|
| Qwen3-ASR-1.7B | Accuracy-first ASR | SOTA among open-source ASR models |
| Qwen3-ASR-0.6B | Efficiency-first ASR | 92 ms TTFT, 2000 s/s at concurrency 128 |
| Qwen3-ForcedAligner-0.6B | NAR forced alignment | 11-language alignment, 80 ms segment resolution |

## Config Facts (from HF `config.json`)

### Qwen3-ASR-1.7B
- Audio: `encoder_layers=24`, `encoder_attention_heads=16`, `d_model=1024`, `output_dim=2048`
- Text: `hidden_size=2048`, `num_hidden_layers=28`, GQA `num_attention_heads=16`, `num_key_value_heads=8`
- Audio chunk params: `n_window=50`, `n_window_infer=800`, `conv_chunksize=500`

### Qwen3-ASR-0.6B
- Audio: `encoder_layers=18`, `encoder_attention_heads=14`, `d_model=896`, `output_dim=1024`
- Text: `hidden_size=1024`, `num_hidden_layers=28`, GQA `16/8`
- Audio chunk params: `n_window=50`, `n_window_infer=800`, `conv_chunksize=500`

### Qwen3-ForcedAligner-0.6B
- Top-level extras: `timestamp_token_id=151705`, `timestamp_segment_time=80`, `classify_num=5000`
- Audio: `encoder_layers=24`, `encoder_attention_heads=16`, `d_model=1024`, `output_dim=1024`
- Text: `hidden_size=1024`, `num_hidden_layers=28`, GQA `16/8`

## Encoder Length Formula (Critical for Correctness)

Official upstream uses per-100-frame chunk logic:

```text
input_lengths_leave = input_lengths % 100
feat_lengths = (input_lengths_leave - 1) // 2 + 1
output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
```

This means exact multiples of 100 frames map to `13 * n_chunks` tokens
(e.g., 400 -> 52, 1000 -> 130), not simple full-sequence 8x downsampling.

## Benchmarks and Claims (from paper/model cards)

- 1.7B reports LibriSpeech test-clean WER 1.51 and WenetSpeech test-net CER 4.97.
- 0.6B reports LibriSpeech test-clean WER 1.72 and WenetSpeech test-net CER 5.38.
- Forced aligner reports 42.9 ms average alignment shift (AAS), outperforming listed baselines.

### Snapshot Tables (for quick reference)

| Model | LibriSpeech clean (WER) | WenetSpeech test-net (CER) |
|---|---:|---:|
| Qwen3-ASR-1.7B | 1.51 | 4.97 |
| Qwen3-ASR-0.6B | 1.72 | 5.38 |

| Forced alignment model | AAS (ms) |
|---|---:|
| Qwen3-ForcedAligner-0.6B | 42.9 |
| NeMo Forced Aligner | 129.8 |
| WhisperX | 133.2 |

### Language Coverage Context

- 30 core languages: Chinese, English, Cantonese, Arabic, German, French,
  Spanish, Portuguese, Indonesian, Italian, Korean, Russian, Thai,
  Vietnamese, Japanese, Turkish, Hindi, Malay, Dutch, Swedish, Danish,
  Finnish, Polish, Czech, Filipino, Persian, Greek, Romanian, Hungarian,
  Macedonian.
- 22 Chinese dialects per official model card/repo docs.
- Official release highlights support for singing voice recognition as well.

## Paper Metadata

- Title: *Qwen3-ASR Technical Report*
- arXiv: https://arxiv.org/abs/2601.21337
- Submitted: **January 29, 2026**; revised version (v2): **January 30, 2026**

## Mac/MLX Implementation Landscape (2026-02-14)

- `ml-explore/mlx-audio` includes a Qwen3-ASR path and is currently the most
  visible upstream MLX implementation.
- `ivan-digital/qwen3-asr-swift` provides a native Swift + MLX implementation
  with published ASR latency claims and packaged 0.6B/1.7B quantized variants.
- Small wrappers/servers exist (GitHub search), but most are thin API layers
  on top of MLX-Audio or custom forks, not full independently validated cores.

## Forced Aligner and Timestamp Facts

- Official Qwen timestamping is a separate model:
  `Qwen/Qwen3-ForcedAligner-0.6B`.
- In official `qwen-asr`, `return_time_stamps=True` requires initializing a
  `forced_aligner`; otherwise it raises an explicit error.
- Official streaming path does **not** support timestamps in that stack.

## Swift Timestamp Status (Evidence-Based)

- In `ivan-digital/qwen3-asr-swift`, ASR docs and source currently show
  transcription support but no ASR timestamp/forced-aligner implementation.
- The repo roadmap lists ASR streaming as pending; no published native forced
  alignment API is present in its ASR modules/docs today.

## Practical MLX Optimization Guidance from Upstream Code

- `mlx-lm`'s `KVCache` uses in-place slice writes and stepped growth strategy;
  this is the proven baseline pattern for decode-time cache efficiency.
- Quantized model loading in MLX stacks generally applies module quantization
  before weight loading and persists quantization metadata in config.
- For this repo, measured wins continue to come from:
  tokenizer/model caching, preallocated KV paths, and validated quant profiles
  (`4-bit`, `group_size=64`) rather than speculative asynchronous decode logic.
