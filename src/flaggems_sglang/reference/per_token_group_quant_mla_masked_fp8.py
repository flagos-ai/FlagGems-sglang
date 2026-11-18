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

"""Reference for quantization/per_token_group_quant_mla_masked_fp8."""

import torch

_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = float(torch.finfo(_FP8_DTYPE).max)


def reference(x, group_size, eps):
    b, m, k = x.shape
    aligned_m = (m + 255) // 256 * 256  # 256 is the gemm's max block_m
    num_tiles_k = k // group_size

    x32 = x.to(torch.float32).reshape(b, m, num_tiles_k, group_size)
    absmax = x32.abs().amax(dim=-1).clamp(min=eps)
    scale = absmax / _FP8_MAX
    q = (x32 / scale[..., None]).clamp(-_FP8_MAX, _FP8_MAX)

    # Rows [m, aligned_m) are padding the kernel never writes; zero them so
    # the reference is well defined, and let the check compare only [:m].
    x_q = torch.zeros((b, aligned_m, k), dtype=_FP8_DTYPE, device=x.device)
    x_q[:, :m] = q.reshape(b, m, k).to(_FP8_DTYPE)
    x_s = torch.zeros(
        (b, aligned_m, num_tiles_k), dtype=torch.float32, device=x.device
    )
    x_s[:, :m] = scale
    masked_m = torch.full((b,), m, dtype=torch.int32, device=x.device)

    return x_q, x_s, masked_m, m, aligned_m
