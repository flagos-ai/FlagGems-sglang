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

"""Reference implementation for activation_norm/gemma_qkv_rmsnorm."""

import torch


def _head_rmsnorm(x, weight, num_heads, head_dim, eps):
    m = x.shape[0]
    x32 = x.reshape(m, num_heads, head_dim).to(torch.float32)
    var = (x32 * x32).mean(dim=-1, keepdim=True)
    out = x32 * torch.rsqrt(var + eps)
    if weight is not None:
        out = out * weight.to(torch.float32)
    return out.reshape(m, num_heads * head_dim).to(x.dtype)


def reference(
    q, k, v, q_weight, k_weight, num_q_heads, num_kv_heads, head_dim, eps
):
    q_out = q.clone()
    q_out[:] = _head_rmsnorm(q, q_weight, num_q_heads, head_dim, eps)
    if k is None or v is None:
        return q_out, k, v
    k_out = k.clone()
    v_out = v.clone()
    k_out[:] = _head_rmsnorm(k, k_weight, num_kv_heads, head_dim, eps)
    # Gemma4's V-norm has with_scale=False, so V is normalized but not scaled.
    v_out[:] = _head_rmsnorm(v, None, num_kv_heads, head_dim, eps)
    return q_out, k_out, v_out
