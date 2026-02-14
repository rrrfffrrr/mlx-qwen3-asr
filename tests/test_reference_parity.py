"""Optional token-level parity test against the official PyTorch implementation.

This test is integration-heavy (downloads model weights and requires torch +
qwen-asr). It is skipped by default unless explicitly enabled.
"""

from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx_qwen3_asr.generate import GenerationConfig, generate
from mlx_qwen3_asr.load_models import load_model


def _enabled() -> bool:
    return os.getenv("RUN_REFERENCE_PARITY", "").strip() == "1"


@pytest.mark.skipif(not _enabled(), reason="Set RUN_REFERENCE_PARITY=1 to enable.")
def test_greedy_token_parity_with_official_reference():
    torch = pytest.importorskip("torch")
    qwen_asr = pytest.importorskip("qwen_asr")

    # Keep this configurable and default to the smaller model for practicality.
    model_id = os.getenv("REFERENCE_PARITY_MODEL", "Qwen/Qwen3-ASR-0.6B")
    max_new_tokens = int(os.getenv("REFERENCE_PARITY_MAX_NEW_TOKENS", "64"))
    fixture = Path("tests/fixtures/test_speech.wav")

    # --- Official reference path (PyTorch) ---
    ref = qwen_asr.Qwen3ASRModel.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map="cpu",
        max_new_tokens=max_new_tokens,
    )
    prompt = ref._build_text_prompt(context="", force_language=None)  # noqa: SLF001
    inputs = ref.processor(
        text=[prompt],
        audio=[str(fixture)],
        return_tensors="pt",
        padding=True,
    )
    inputs = inputs.to(ref.model.device).to(ref.model.dtype)

    with torch.no_grad():
        ref_out = ref.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    prompt_len = int(inputs["input_ids"].shape[1])
    ref_tokens = ref_out.sequences[0, prompt_len:].tolist()

    # --- MLX path using identical input_ids + input_features ---
    model, _ = load_model(model_id, dtype=mx.float16)

    input_ids = mx.array(inputs["input_ids"].cpu().numpy().astype(np.int32))
    mel = mx.array(inputs["input_features"].cpu().numpy().astype(np.float32))
    feature_lens = mx.array(inputs["feature_attention_mask"].sum(-1).cpu().numpy().astype(np.int32))

    audio_features, _ = model.audio_tower(mel.astype(mx.float16), feature_lens)

    seq_len = input_ids.shape[1]
    pos = mx.arange(seq_len)[None, :]
    position_ids = mx.stack([pos, pos, pos], axis=1)

    our_tokens = generate(
        model=model,
        input_ids=input_ids,
        audio_features=audio_features,
        position_ids=position_ids,
        config=GenerationConfig(max_new_tokens=max_new_tokens, temperature=0.0),
    )

    # Normalize by trimming trailing EOS tokens from both sides.
    eos_ids = {151643, 151645}
    while ref_tokens and ref_tokens[-1] in eos_ids:
        ref_tokens.pop()
    while our_tokens and our_tokens[-1] in eos_ids:
        our_tokens.pop()

    assert our_tokens == ref_tokens
