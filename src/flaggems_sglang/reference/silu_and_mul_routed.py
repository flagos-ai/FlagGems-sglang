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

"""Reference implementation for moe/silu_and_mul_routed."""

import torch


def reference(gateup_output, topk_weights, out_dtype, use_interleaved):
    m, hidden = gateup_output.shape
    n = hidden // 2
    x = gateup_output.to(torch.float32)
    if use_interleaved:
        # rows are [g0, u0, g1, u1, ...]
        pairs = x.reshape(m, n, 2)
        gate, up = pairs[..., 0], pairs[..., 1]
    else:
        gate, up = x[:, :n], x[:, n:]

    out = gate * torch.sigmoid(gate) * up
    if topk_weights is not None:
        # The weight scale is applied last, before the single rounding cast.
        out = out * topk_weights.to(torch.float32)[:, None]
    return out.to(out_dtype or gateup_output.dtype)
