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

"""Reference implementation for rope/fused_qk_complex_rope."""

import torch


def reference(q, k, freqs_cis):
    def rot(x):
        xf = x.float()
        c = freqs_cis[..., 0].unsqueeze(-2).float()
        s = freqs_cis[..., 1].unsqueeze(-2).float()
        x1 = xf[..., 0::2]
        x2 = xf[..., 1::2]
        o1 = x1 * c - x2 * s
        o2 = x1 * s + x2 * c
        return torch.stack([o1, o2], dim=-1).reshape(xf.shape).to(x.dtype)

    return rot(q), rot(k)
