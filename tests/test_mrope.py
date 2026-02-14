"""Tests for mlx_qwen3_asr/mrope.py."""

import mlx.core as mx
import numpy as np
import pytest

from mlx_qwen3_asr.mrope import (
    InterleavedMRoPE,
    _rotate_half,
    apply_rotary_pos_emb,
)


def _official_interleaved_reference(
    position_ids: np.ndarray,
    head_dim: int,
    base: float,
    mrope_section: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Numpy reference equivalent to official Qwen apply_interleaved_mrope."""
    inv_freq = 1.0 / (
        base ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim)
    )

    # (B, 3, L) -> (3, B, L, half_dim)
    pos = position_ids.astype(np.float32).transpose(1, 0, 2)
    freqs = pos[..., None] * inv_freq[None, None, None, :]

    freqs_t = freqs[0].copy()
    for dim, offset in enumerate((1, 2), start=1):
        idx = slice(offset, mrope_section[dim] * 3, 3)
        freqs_t[..., idx] = freqs[dim, ..., idx]

    emb = np.concatenate([freqs_t, freqs_t], axis=-1)
    return np.cos(emb), np.sin(emb)

# ---------------------------------------------------------------------------
# InterleavedMRoPE constructor
# ---------------------------------------------------------------------------


class TestInterleavedMRoPEInit:
    """Test InterleavedMRoPE constructor validation."""

    def test_default_sections(self):
        mrope = InterleavedMRoPE(head_dim=128)
        assert mrope.mrope_section == [24, 20, 20]
        assert mrope.half_dim == 64

    def test_sections_must_sum_to_half_dim(self):
        """If sections don't sum to head_dim//2, assertion should fail."""
        with pytest.raises(AssertionError, match="must equal head_dim // 2"):
            InterleavedMRoPE(head_dim=128, mrope_section=[10, 10, 10])

    def test_custom_valid_sections(self):
        mrope = InterleavedMRoPE(head_dim=64, mrope_section=[12, 10, 10])
        assert mrope.half_dim == 32
        assert sum(mrope.mrope_section) == 32


# ---------------------------------------------------------------------------
# InterleavedMRoPE output shapes
# ---------------------------------------------------------------------------


class TestInterleavedMRoPEShapes:
    """Test output shapes of InterleavedMRoPE.__call__."""

    def test_output_shape(self):
        mrope = InterleavedMRoPE(head_dim=128)
        batch, seq_len = 2, 10
        position_ids = mx.zeros((batch, 3, seq_len), dtype=mx.int32)
        cos, sin = mrope(position_ids)
        assert cos.shape == (batch, seq_len, 128)
        assert sin.shape == (batch, seq_len, 128)

    def test_output_shape_single_batch(self):
        mrope = InterleavedMRoPE(head_dim=128)
        position_ids = mx.zeros((1, 3, 5), dtype=mx.int32)
        cos, sin = mrope(position_ids)
        assert cos.shape == (1, 5, 128)
        assert sin.shape == (1, 5, 128)

    def test_output_dtype(self):
        mrope = InterleavedMRoPE(head_dim=128)
        position_ids = mx.zeros((1, 3, 5), dtype=mx.int32)
        cos, sin = mrope(position_ids, dtype=mx.float32)
        assert cos.dtype == mx.float32
        assert sin.dtype == mx.float32


class TestMRoPEParity:
    """Test MRoPE matches official interleaving behavior."""

    def test_matches_official_reference(self):
        mrope = InterleavedMRoPE(head_dim=128, base=1000000.0)
        pos_ids = np.array(
            [[[0, 1, 2, 3], [5, 6, 7, 8], [9, 10, 11, 12]]],
            dtype=np.int32,
        )

        cos, sin = mrope(mx.array(pos_ids), dtype=mx.float32)
        mx.eval(cos, sin)
        cos_np = np.array(cos)
        sin_np = np.array(sin)

        ref_cos, ref_sin = _official_interleaved_reference(
            position_ids=pos_ids,
            head_dim=128,
            base=1000000.0,
            mrope_section=[24, 20, 20],
        )

        np.testing.assert_allclose(cos_np, ref_cos, atol=1e-6, rtol=0)
        np.testing.assert_allclose(sin_np, ref_sin, atol=1e-6, rtol=0)

    def test_uncovered_indices_follow_temporal_dimension(self):
        """Indices 61/62 are not overwritten by H/W and stay temporal-derived."""
        mrope = InterleavedMRoPE(head_dim=128, base=1000000.0)
        pos_ids = mx.array([[[0], [123], [456]]], dtype=mx.int32)  # T=0
        cos, sin = mrope(pos_ids, dtype=mx.float32)
        mx.eval(cos, sin)

        cos_np = np.array(cos[0, 0, :])
        sin_np = np.array(sin[0, 0, :])

        # Since temporal position is zero, temporal-derived frequencies yield cos=1, sin=0.
        for idx in (61, 62, 125, 126):
            assert cos_np[idx] == pytest.approx(1.0, abs=1e-6)
            assert sin_np[idx] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# apply_rotary_pos_emb
# ---------------------------------------------------------------------------


class TestApplyRotaryPosEmb:
    """Test apply_rotary_pos_emb preserves tensor shapes."""

    def test_output_shapes(self):
        batch, n_heads, seq_len, head_dim = 2, 4, 10, 128
        q = mx.random.normal((batch, n_heads, seq_len, head_dim))
        k = mx.random.normal((batch, n_heads, seq_len, head_dim))
        cos = mx.random.normal((batch, seq_len, head_dim))
        sin = mx.random.normal((batch, seq_len, head_dim))

        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
        assert q_rot.shape == (batch, n_heads, seq_len, head_dim)
        assert k_rot.shape == (batch, n_heads, seq_len, head_dim)

    def test_4d_cos_sin_no_expansion(self):
        """When cos/sin are already 4D, no unsqueeze should happen."""
        batch, n_heads, seq_len, head_dim = 1, 2, 5, 8
        q = mx.random.normal((batch, n_heads, seq_len, head_dim))
        k = mx.random.normal((batch, n_heads, seq_len, head_dim))
        cos = mx.random.normal((batch, 1, seq_len, head_dim))
        sin = mx.random.normal((batch, 1, seq_len, head_dim))

        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape


# ---------------------------------------------------------------------------
# _rotate_half
# ---------------------------------------------------------------------------


class TestRotateHalf:
    """Test _rotate_half with known input."""

    def test_known_values(self):
        x = mx.array([[1.0, 2.0, 3.0, 4.0]])
        result = _rotate_half(x)
        # Mid = 2: x1 = [1, 2], x2 = [3, 4]
        # Output = [-x2, x1] = [-3, -4, 1, 2]
        expected = [[-3.0, -4.0, 1.0, 2.0]]
        np.testing.assert_array_equal(np.array(result), expected)

    def test_preserves_shape(self):
        x = mx.random.normal((2, 4, 8))
        result = _rotate_half(x)
        assert result.shape == x.shape
