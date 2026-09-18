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

import torch


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
    num_tokens, n_q_dim = q.shape
    n_k_dim = k.shape[1]
    n_qh = n_q_dim // head_size
    n_kh = n_k_dim // head_size
    half_rd = rotary_dim // 2

    section_h, section_w, section_t = mrope_section
    assert (
        section_h == section_w
    ), "Ernie4.5 layout assumes section_h == section_w"
    section_hw = section_h + section_w

    tpos = positions[0].long()
    hpos = positions[1].long()
    wpos = positions[2].long()

    ridx = torch.arange(half_rd, device=q.device)
    use_hw = (ridx < section_hw).unsqueeze(0)
    use_h = ((ridx % 2) == 0).unsqueeze(0)

    pos_hw = torch.where(use_h, hpos.unsqueeze(1), wpos.unsqueeze(1))
    pos = torch.where(use_hw, pos_hw, tpos.unsqueeze(1))  # (T, half_rd)

    col = ridx.unsqueeze(0).expand(num_tokens, half_rd)
    cos = cos_sin_cache[pos, col].float()
    sin = cos_sin_cache[pos, col + half_rd].float()

    q_out = _apply_rope(q, n_qh, head_size, rotary_dim, cos, sin)
    k_out = _apply_rope(k, n_kh, head_size, rotary_dim, cos, sin)
    return q_out, k_out
