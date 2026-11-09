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

"""Reference implementation for activation_norm/gemma_dual_rmsnorm_residual_scalar."""

import torch


def _rmsnorm(x, weight, eps):
    var = (x * x).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(var + eps) * weight.to(torch.float32)


def reference(
    x1, weight1, x2, weight2, weight3, residual, scalar, eps1, eps2, eps3
):
    n1 = _rmsnorm(x1.to(torch.float32), weight1, eps1)
    n2 = _rmsnorm(x2.to(torch.float32), weight2, eps2)
    n3 = _rmsnorm(n1 + n2, weight3, eps3)
    out = (n3 + residual.to(torch.float32)) * scalar.to(torch.float32)
    return out.to(x1.dtype)
