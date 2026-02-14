# Technical Decisions

Key technical decisions made for mlx-qwen3-asr, with rationale.

## Decision 1: Python + MLX

**Choice:** Python with Apple's MLX framework
**Alternatives considered:** Rust, Swift, C++

**Rationale:**
- MLX has no Rust bindings -- the framework is Python and C++ only
- 95% of runtime is spent in Metal/C++ kernels inside MLX -- Python overhead is negligible
- Swift port already exists (qwen3-asr-swift) -- no need to duplicate
- Python ecosystem makes it easy to integrate with HuggingFace, numpy, etc.
- Fastest path to a working implementation

## Decision 2: Standalone Package (not part of mlx-audio)

**Choice:** Independent `mlx-qwen3-asr` package
**Alternative:** Contribute to mlx-audio

**Rationale:**
- mlx-audio has critical bugs for Qwen3-ASR:
  - Uses standard nn.RoPE instead of interleaved MRoPE
  - Issue #459: long-audio truncation
  - Historical config drift risk across revisions
- mlx-audio depends on bleeding-edge packages: `transformers==5.0.0rc3`, `mlx-lm==0.30.5`
- Qwen3-ASR deserves dedicated focus -- it's SOTA and complex enough to warrant its own package
- Standalone allows us to optimize specifically for this model without compromise

## Decision 3: Stage Forced Aligner Behind a Guard

**Choice:** Ship ASR + streaming + quantization in v1; keep forced aligner as explicit WIP
**Alternative:** Expose timestamps before the aligner implementation is complete

**Rationale:**
- Core ASR quality and model correctness matter more than partial timestamp support
- A clear fail-fast guard is better UX than late runtime failures
- The dedicated aligner architecture is still valuable and kept as scaffolding
- This keeps the public surface honest while preserving implementation momentum

## Decision 4: HuggingFace Tokenizer

**Choice:** Use `transformers.AutoTokenizer` (Qwen2TokenizerFast)
**Alternative:** Reimplement BPE tokenizer from scratch

**Rationale:**
- Qwen2TokenizerFast has 151936 tokens -- reimplementing is high effort, low value
- The tokenizer is well-tested and handles all edge cases
- Accept `transformers` as a dependency (needed for tokenizer only, not model inference)
- Tokenization is not a performance bottleneck

## Decision 5: ffmpeg for Audio Loading

**Choice:** Subprocess call to ffmpeg
**Alternative:** librosa, soundfile, torchaudio

**Rationale:**
- Same approach as mlx-whisper -- proven pattern
- Handles all audio formats (mp3, wav, flac, ogg, m4a, etc.)
- No additional Python dependencies (librosa pulls in many)
- ffmpeg is universally available (`brew install ffmpeg`)
- Consistent behavior across audio formats

## Decision 6: On-the-fly Weight Remapping

**Choice:** Remap HF weights at load time (strip `thinker.` prefix, transpose Conv2d)
**Alternative:** Require pre-converted weights

**Rationale:**
- Users can point directly at HuggingFace repo -- no separate conversion step
- Remapping is fast (< 1 second) compared to model download
- Reduces friction for new users
- Still support pre-converted local weights for advanced users

## Decision 7: Interleaved MRoPE (Custom Implementation)

**Choice:** Custom InterleavedMRoPE class instead of MLX's built-in nn.RoPE
**Alternative:** Use nn.RoPE with workarounds

**Rationale:**
- MLX's nn.RoPE doesn't support 3D interleaved frequency assignment
- The interleaving pattern (stride-3 across sections [24,20,20]) is specific to Qwen3
- Incorrect RoPE produces plausible but degraded transcription -- hard to debug
- This is the #1 bug in existing implementations (mlx-audio gets this wrong)
- Correctness is non-negotiable for the core position encoding
