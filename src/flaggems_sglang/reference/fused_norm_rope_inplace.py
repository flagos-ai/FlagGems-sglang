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

"""Reference implementation for rope/fused_norm_rope_inplace."""

import torch


def reference(kv, weight, eps, freqs_cis, positions):
    m, head_dim = kv.shape
    freqs_real = torch.view_as_real(freqs_cis).flatten(-2)  # [rows, rope_dim]
    rope_dim = freqs_real.shape[-1]
    rope_start = head_dim - rope_dim

    # RMSNorm spans the whole head dim; only the trailing rope segment is
    # rotated.
    x = kv.to(torch.float32)
    rms_inv = torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + eps)
    normed = x * rms_inv
    if weight is not None:
        normed = normed * weight.to(torch.float32)

    rows = (
        torch.arange(m, device=kv.device)
        if positions is None
        else positions.long()
    )
    f = freqs_real[rows].to(torch.float32)  # [M, rope_dim]
    f_real, f_imag = f[:, 0::2], f[:, 1::2]

    seg = normed[:, rope_start:]
    x_real, x_imag = seg[:, 0::2], seg[:, 1::2]
    # Interleaved (real, imag) pairs: out = x * f.
    out_real = x_real * f_real - x_imag * f_imag
    out_imag = x_real * f_imag + x_imag * f_real

    out = normed.clone()
    out[:, rope_start::2] = out_real
    out[:, rope_start + 1 :: 2] = out_imag
    return out.to(kv.dtype)
