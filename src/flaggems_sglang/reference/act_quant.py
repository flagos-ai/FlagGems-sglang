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

"""Reference implementation for dsa/act_quant."""

import torch

_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = 448.0
_FP8_MIN = -448.0


def _exact_exp2_int(exp_tensor):
    """Exact 2**n for integer exponents, via IEEE 754 bit manipulation.

    torch.exp2 has a 1-ULP error for integer inputs on some GPUs; assembling
    the exponent field directly is bit-exact everywhere.
    """
    exp_int = exp_tensor.to(torch.int32)
    bits = (exp_int + 127).clamp(1, 254) << 23
    return bits.view(torch.float32)


def reference(x, block_size=128, scale_fmt=None):
    round_scale = scale_fmt is not None
    orig_shape = x.shape
    n = orig_shape[-1]

    x_flat = x.reshape(-1, n).float()
    m = x_flat.shape[0]
    x_g = x_flat.view(m, n // block_size, block_size)

    # The 1e-4 floor is what separates this from per_token_group_quant_fp8,
    # which floors at 1e-12: this one quantizes DSA indexer activations.
    amax = x_g.abs().amax(dim=-1).clamp(min=1e-4)
    if round_scale:
        scale = _exact_exp2_int(torch.ceil(torch.log2(amax / _FP8_MAX)))
    else:
        scale = amax / _FP8_MAX

    y = (x_g / scale.unsqueeze(-1)).clamp(_FP8_MIN, _FP8_MAX).to(_FP8_DTYPE)
    y = y.view(m, n).view(orig_shape)
    s = scale.view(*orig_shape[:-1], n // block_size)

    return y, s
