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

"""Reference implementation for embedding/apply_rotary_emb."""

import torch


def reference(x, freqs_cis, positions=None, inverse=False):
    """Adjacent-pair rotary rotation driven by a real (cos, sin) table.

    Args:
        x: Tensor[rows, pairs, 2]
        freqs_cis: Tensor[n_rows, pairs, 2], last axis holds (cos, sin)
        positions: optional Tensor[n_rows] row selector into freqs_cis
        inverse: use the conjugate (sin sign flip)

    Returns:
        Rotated tensor with the shape and dtype of x.
    """
    xf = x.float()
    if positions is not None:
        table = freqs_cis[positions.long()]
    else:
        table = freqs_cis[: xf.shape[0]]
    c = table[..., 0].unsqueeze(-2).float()
    s = table[..., 1].unsqueeze(-2).float()
    if inverse:
        s = -s
    x1 = xf[..., 0::2]
    x2 = xf[..., 1::2]
    o1 = x1 * c - x2 * s
    o2 = x1 * s + x2 * c
    out = torch.stack([o1, o2], dim=-1).reshape(xf.shape)
    return out.to(x.dtype)
