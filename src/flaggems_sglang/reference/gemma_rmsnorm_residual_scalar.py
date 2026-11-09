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

"""Reference implementation for activation_norm/gemma_rmsnorm_residual_scalar."""

import torch


def reference(x, weight, residual, scalar, eps):
    x32 = x.to(torch.float32)
    var = (x32 * x32).mean(dim=-1, keepdim=True)
    # Note the weight is a plain multiplier here (no `1 +`), unlike
    # activation_norm/gemma_rmsnorm.
    out = x32 * torch.rsqrt(var + eps) * weight.to(torch.float32)
    out = (out + residual.to(torch.float32)) * scalar.to(torch.float32)
    return out.to(x.dtype)
