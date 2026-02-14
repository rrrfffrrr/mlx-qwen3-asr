# mlx-qwen3-asr

[![PyPI version](https://img.shields.io/pypi/v/mlx-qwen3-asr.svg)](https://pypi.org/project/mlx-qwen3-asr/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://pypi.org/project/mlx-qwen3-asr/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

Qwen3-ASR speech recognition on Apple Silicon via [MLX](https://github.com/ml-explore/mlx).

## What is this?

**mlx-qwen3-asr** is a standalone, correct MLX port of
[Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B), the current
state-of-the-art open-source ASR model released by the Alibaba Qwen team in
January 2026. Qwen3-ASR beats Whisper-large-v3 across nearly every benchmark
and supports 52 languages and dialects (30 languages + 22 Chinese dialects)
per the official release docs.

**Why a separate port?** The existing mlx-audio port has known issues: no proper
Multi-dimensional RoPE (MRoPE), long-audio truncation bugs, and incorrect config
handling for the 1.7B model. This project provides the definitive way to run
Qwen3-ASR natively on Mac.

### Key features

- Full audio encoder + text decoder with correct interleaved MRoPE
- Supports both 1.7B and 0.6B model sizes
- Long audio chunking (up to 20 minutes per chunk)
- Forced alignment scaffolding for word-level timestamps (WIP)
- Streaming ASR support
- Multiple output formats: txt, json, srt, vtt, tsv
- 4-bit and 8-bit quantization
- Minimal dependencies: mlx, numpy, huggingface-hub, transformers

## Installation

Install from PyPI:

```bash
pip install mlx-qwen3-asr
```

For development:

```bash
git clone https://github.com/dmoon/mlx-qwen3-asr.git
cd mlx-qwen3-asr
pip install -e ".[dev]"
```

## Quick Start

### Python API

```python
from mlx_qwen3_asr import transcribe

result = transcribe("audio.wav")
print(result.text)
print(result.language)
```

With options:

```python
result = transcribe(
    "meeting.mp3",
    model="Qwen/Qwen3-ASR-1.7B",
    language="English",
    verbose=True,
)
print(result.text)
```

### Loading models explicitly

```python
from mlx_qwen3_asr import load_model, load_audio, transcribe

model, config = load_model("Qwen/Qwen3-ASR-0.6B")
audio = load_audio("speech.wav")
result = transcribe(audio, model=model)
```

## CLI Usage

Basic transcription:

```bash
mlx-qwen3-asr audio.wav
```

Specify model, language, and output format:

```bash
mlx-qwen3-asr recording.mp3 --model Qwen/Qwen3-ASR-0.6B --language English -f srt -o output/
```

Multiple files with all output formats:

```bash
mlx-qwen3-asr *.wav -f all -o transcripts/ --verbose
```

Run `mlx-qwen3-asr --help` for the full list of options.

`--timestamps` is currently unavailable and exits with a clear error until
forced alignment is fully implemented.

## API Reference

### `transcribe(audio, *, model, language, return_timestamps, forced_aligner, dtype, max_new_tokens, verbose)`

Transcribe audio to text. Accepts a file path, numpy array, `mx.array`, or
`(array, sample_rate)` tuple. Returns a `TranscriptionResult`.
`return_timestamps=True` is currently not implemented and raises
`NotImplementedError`.

### `load_model(name_or_path, *, dtype)`

Load a Qwen3-ASR model and its config from a HuggingFace repo or local path.
Returns `(model, config)`.

### `load_audio(path_or_url)`

Load and resample audio to mono 16 kHz. Returns an `mx.array`.

### `TranscriptionResult`

Frozen dataclass with fields:
- `text` (str) -- transcribed text
- `language` (str) -- detected or forced language
- `segments` (list[dict] | None) -- `None` for now (word-level timestamps are WIP)

## Supported Languages

Qwen3-ASR supports 52 languages and dialects in total. The table below lists
the 30 core languages from the official docs:

| Language | Language | Language | Language |
|----------|----------|----------|----------|
| Arabic | Cantonese | Chinese | Czech |
| Danish | Dutch | English | Filipino |
| Finnish | French | German | Greek |
| Hindi | Hungarian | Indonesian | Italian |
| Japanese | Korean | Macedonian | Malay |
| Persian | Polish | Portuguese | Romanian |
| Russian | Spanish | Swedish | Thai |
| Turkish | Vietnamese | | |

Chinese dialect support (22 dialects) is provided by the official model but is
not expanded in this table.

## Model Variants

| | Qwen3-ASR-1.7B | Qwen3-ASR-0.6B |
|---|---|---|
| **Parameters** | 1.7B | 0.6B |
| **Audio encoder layers** | 24 | 18 |
| **Text decoder layers** | 28 | 28 |
| **Audio encoder dim (`d_model`)** | 1024 | 896 |
| **Text hidden size** | 2048 | 1024 |
| **Text attention (Q/KV heads)** | GQA (16/8) | GQA (16/8) |
| **RoPE theta** | 1,000,000 | 1,000,000 |
| **Accuracy** | Higher | Slightly lower |
| **Speed** | Slower | Faster |
| **HuggingFace** | `Qwen/Qwen3-ASR-1.7B` | `Qwen/Qwen3-ASR-0.6B` |

Both models use Multi-dimensional RoPE (MRoPE), 128-bin mel spectrograms, and
the same tokenizer with a vocabulary size of 151,936.

## Benchmarks

Selected word error rates (WER, lower is better) on standard benchmarks:

| Benchmark | Qwen3-ASR-1.7B | Whisper-large-v3 |
|---|---|---|
| LibriSpeech test-clean | **1.51** | 2.02 |
| LibriSpeech test-other | **3.04** | 4.28 |
| WenetSpeech test-net | **4.97** | 9.68 |
| Fleurs (avg 30 langs) | **5.2** | 8.1 |

Numbers from the [Qwen3-ASR technical report](https://huggingface.co/Qwen/Qwen3-ASR-1.7B).

## Acknowledgments

- [Qwen team](https://github.com/QwenLM) at Alibaba for the Qwen3-ASR model
- [mlx-whisper](https://github.com/ml-explore/mlx-examples) for architecture patterns and inspiration
- [Apple MLX team](https://github.com/ml-explore/mlx) for the MLX framework

## License

Apache 2.0. See [LICENSE](LICENSE) for details.
