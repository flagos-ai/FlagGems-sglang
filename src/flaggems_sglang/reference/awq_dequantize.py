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

"""Reference implementation for quantization/awq_dequantize."""

import torch

_BITS = 4
_AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]


def _reverse_awq_order(t: torch.Tensor) -> torch.Tensor:
    idx = torch.arange(t.shape[-1], dtype=torch.int32, device=t.device)
    idx = idx.view(-1, 32 // _BITS)[:, _AWQ_REVERSE_ORDER].view(-1)
    return (t[:, idx] & 0xF).contiguous()


def reference(qweight, scales, zeros):
    group_size = qweight.shape[0] // scales.shape[0]
    shifts = torch.arange(0, 32, _BITS, device=qweight.device)

    iweights = torch.bitwise_right_shift(
        qweight[:, :, None], shifts[None, None, :]
    ).to(torch.int8)
    iweights = _reverse_awq_order(iweights.view(iweights.shape[0], -1))

    zbits = torch.bitwise_right_shift(
        zeros[:, :, None], shifts[None, None, :]
    ).to(torch.int8)
    zbits = _reverse_awq_order(zbits.view(zeros.shape[0], -1))

    iweights = torch.bitwise_and(iweights, (2**_BITS) - 1)
    zbits = torch.bitwise_and(zbits, (2**_BITS) - 1)

    scales_rep = scales.repeat_interleave(group_size, dim=0)
    zeros_rep = zbits.repeat_interleave(group_size, dim=0)
    return (iweights - zeros_rep) * scales_rep
