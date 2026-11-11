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

"""Reference for moe/fp8_per_token_to_per_tensor_quant."""

import torch

_SCALE_GROUP = 128


def reference(x, x_scale, masked_m, output_scale):
    """Dequantize with the group scales, then requantize to one tensor scale.

    The scale of group ``g`` covers columns ``[g * 128, (g + 1) * 128)``, so
    ``x_scale`` is expanded by repeating each entry 128 times along the last
    axis. Rows at or past ``masked_m[e]`` come back as zero rather than the
    value the multiplication would give -- that band is not a real activation.

    Mirrors the kernel: one pass, no intermediate buffer, fp8 payload in and
    fp8 payload out.
    """
    dequant = x.to(torch.float32) * x_scale.repeat_interleave(
        _SCALE_GROUP, dim=2
    )
    quantized = (dequant / output_scale.reshape(())).to(torch.float8_e4m3fn)

    live = (
        torch.arange(x.shape[1], device=x.device)[None, :]
        < masked_m.to(torch.int64)[:, None]
    )
    return torch.where(live[:, :, None], quantized, quantized.new_zeros(()))
