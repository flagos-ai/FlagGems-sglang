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

"""Reference implementation for activation_norm/gemma_fused_add_rmsnorm."""

import torch


def reference(x, residual, weight, eps):
    s = x.to(torch.float32) + residual.to(torch.float32)
    var = (s * s).mean(dim=-1, keepdim=True)
    out = s * torch.rsqrt(var + eps) * (1.0 + weight.to(torch.float32))
    # The pre-norm sum is returned too: the next layer's add consumes it.
    return out.to(x.dtype), s.to(x.dtype)
