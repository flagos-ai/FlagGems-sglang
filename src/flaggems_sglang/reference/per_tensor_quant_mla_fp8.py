# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import torch

__all__ = ["reference"]

FP8_MAX = 448.0


def reference(x, eps=1e-12):
    """Quantize a 3-D MLA tensor to FP8 with one tensor-wide scale.

    Written with the same divisions the kernel performs, so the encoded bytes
    are comparable without a tolerance.
    """
    xf = x.float()
    if xf.numel() == 0:
        scale = torch.full(
            (1,), float(eps) / FP8_MAX, dtype=torch.float32, device=x.device
        )
        return torch.empty_like(xf, dtype=torch.float8_e4m3fn).contiguous(), (
            scale
        )

    amax = torch.clamp_min(xf.abs().amax(), float(eps))
    scale = amax / FP8_MAX
    quantized = torch.clamp(xf / scale, -FP8_MAX, FP8_MAX).to(
        torch.float8_e4m3fn
    )
    return quantized.contiguous(), scale.reshape(1)
