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

"""Reference implementation for rope/apply_rotary_emb."""

import torch


def reference(x, freqs_cis, positions, inverse):
    freqs_real = torch.view_as_real(freqs_cis).flatten(-2).to(torch.float32)
    rows = (
        torch.arange(x.shape[0], device=x.device)
        if positions is None
        else positions.to(torch.int64)
    )
    f = freqs_real[rows]  # [tokens, rope_dim]
    if x.ndim == 3:
        # One freq row per token, shared by every head.
        f = f[:, None, :]
    f_real, f_imag = f[..., 0::2], f[..., 1::2]

    x32 = x.to(torch.float32)
    x_real, x_imag = x32[..., 0::2], x32[..., 1::2]
    if inverse:
        # Conjugate rotation: x * conj(f).
        out_real = x_real * f_real + x_imag * f_imag
        out_imag = x_imag * f_real - x_real * f_imag
    else:
        out_real = x_real * f_real - x_imag * f_imag
        out_imag = x_real * f_imag + x_imag * f_real
    return torch.stack([out_real, out_imag], dim=-1).flatten(-2).to(x.dtype)
