# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pure-torch reference implementation of mrope_fused.

Used as correctness ground truth and benchmark baseline.
Source: kernel-comp-baseline/problems/rope/mrope_fused/reference_torch.py
"""

import torch


def _rows_for_axis(cos_sin_cache, positions_axis):
    return cos_sin_cache[positions_axis.long()]


def _apply_rope(x, n_h, head_size, rotary_dim, cos, sin):
    num_tokens = x.shape[0]
    half_rd = rotary_dim // 2
    x = x.view(num_tokens, n_h, head_size).clone()
    x1 = x[..., :half_rd].float()
    x2 = x[..., half_rd:rotary_dim].float()
    cos_e = cos.unsqueeze(1)
    sin_e = sin.unsqueeze(1)
    new1 = x1 * cos_e - x2 * sin_e
    new2 = x2 * cos_e + x1 * sin_e
    out = torch.cat(
        [new1.to(x.dtype), new2.to(x.dtype), x[..., rotary_dim:]], dim=-1
    )
    return out.view(num_tokens, n_h * head_size)


def reference(
    q, k, cos_sin_cache, positions, mrope_section, head_size, rotary_dim
):
    """Pure-torch mrope_fused: non-interleaved, neox-style, no axis map.

    Args:
        q: Tensor[T, n_qh * head_size]
        k: Tensor[T, n_kh * head_size]
        cos_sin_cache: Tensor[max_pos, rotary_dim]
        positions: Tensor[3, T] int64
        mrope_section: list[int, int, int]
        head_size: int
        rotary_dim: int

    Returns:
        (q_out, k_out): rotated tensors with same shape as input.
    """
    num_tokens, n_q_dim = q.shape
    n_k_dim = k.shape[1]
    n_qh = n_q_dim // head_size
    n_kh = n_k_dim // head_size
    half_rd = rotary_dim // 2

    t_end = mrope_section[0]
    h_end = t_end + mrope_section[1]

    t_row = _rows_for_axis(cos_sin_cache, positions[0])
    h_row = _rows_for_axis(cos_sin_cache, positions[1])
    w_row = _rows_for_axis(cos_sin_cache, positions[2])

    idx = torch.arange(half_rd, device=q.device)
    t_mask = idx < t_end
    h_mask = (idx >= t_end) & (idx < h_end)
    w_mask = (idx >= h_end) & (idx < half_rd)

    cos = torch.where(
        t_mask, t_row[:, :half_rd], torch.zeros_like(t_row[:, :half_rd])
    )
    cos = torch.where(h_mask, h_row[:, :half_rd], cos)
    cos = torch.where(w_mask, w_row[:, :half_rd], cos)

    sin = torch.where(
        t_mask, t_row[:, half_rd:rotary_dim], torch.zeros_like(cos)
    )
    sin = torch.where(h_mask, h_row[:, half_rd:rotary_dim], sin)
    sin = torch.where(w_mask, w_row[:, half_rd:rotary_dim], sin)

    q_out = _apply_rope(q, n_qh, head_size, rotary_dim, cos, sin)
    k_out = _apply_rope(k, n_kh, head_size, rotary_dim, cos, sin)
    return q_out, k_out
