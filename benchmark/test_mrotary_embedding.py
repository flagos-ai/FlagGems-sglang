import pytest
import torch

import flaggems_sglang

from .attri_util import FLOAT_DTYPES, MROTARY_EMBEDDING_BENCH_SHAPES


class FakeMRotaryEmbedding:
    """Minimal stub matching sglang's MRotaryEmbedding attributes."""

    def __init__(
        self,
        head_size,
        rotary_dim,
        max_position_embeddings,
        base,
        is_neox_style,
        dtype,
        mrope_section=None,
        mrope_interleaved=False,
        mrope_interleaved_glm=False,
        device="cuda",
    ):
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype
        self.mrope_section = mrope_section
        self.mrope_interleaved = mrope_interleaved
        self.mrope_interleaved_glm = mrope_interleaved_glm
        self.cos_sin_cache = self._compute_cos_sin_cache().to(
            device=device, dtype=dtype
        )
        self.axis_map = (
            self._compute_axis_map(device)
            if mrope_interleaved_glm and mrope_section
            else None
        )

    def _compute_inv_freq(self):
        return 1.0 / (
            self.base
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float)
                / self.rotary_dim
            )
        )

    def _compute_cos_sin_cache(self):
        inv_freq = self._compute_inv_freq()
        t = torch.arange(self.max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        return torch.cat((freqs.cos(), freqs.sin()), dim=-1)

    def _compute_axis_map(self, device="cuda"):
        num_pairs = self.rotary_dim // 2
        axis_map = torch.empty(num_pairs, dtype=torch.long, device=device)
        counts = [0, 0, 0]
        for i in range(num_pairs):
            current_ax = i % 3
            while counts[current_ax] >= self.mrope_section[current_ax]:
                current_ax = (current_ax + 1) % 3
            axis_map[i] = current_ax
            counts[current_ax] += 1
        return axis_map


def _make_inputs(N, n_qh, n_kh, head_size, max_pos, dtype, device):
    q = torch.randn(N, n_qh * head_size, device=device, dtype=dtype)
    k = torch.randn(N, n_kh * head_size, device=device, dtype=dtype)
    positions = torch.randint(
        0, max_pos, (3, N), device=device, dtype=torch.int64
    )
    return q, k, positions


@pytest.mark.parametrize("shape", MROTARY_EMBEDDING_BENCH_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.mrotary_embedding
def test_mrotary_embedding(shape, dtype, benchmark):
    (
        N,
        n_qh,
        n_kh,
        head_size,
        rotary_dim,
        is_neox,
        mrope_interleaved,
        mrope_interleaved_glm,
        section_t,
        section_h,
        section_w,
        label,
        _weight,
    ) = shape

    device = flaggems_sglang.device
    mrope_section = [section_t, section_h, section_w]

    obj = FakeMRotaryEmbedding(
        head_size=head_size,
        rotary_dim=rotary_dim,
        max_position_embeddings=8192,
        base=10000000,
        is_neox_style=is_neox,
        dtype=dtype,
        mrope_section=mrope_section,
        mrope_interleaved=mrope_interleaved,
        mrope_interleaved_glm=mrope_interleaved_glm,
        device=device,
    )

    q, k, positions = _make_inputs(
        N, n_qh, n_kh, head_size, 8192, dtype, device
    )

    def run():
        qq, kk = q.clone(), k.clone()
        return flaggems_sglang.mrotary_embedding(obj, positions, qq, kk)

    benchmark(run)
