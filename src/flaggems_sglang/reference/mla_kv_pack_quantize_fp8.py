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

"""Reference implementation for attention/mla_kv_pack_quantize_fp8."""

import torch


def reference(k_nope, k_pe, v, k_scale_inv, v_scale_inv, fp8_dtype):
    s, num_heads, _ = k_nope.shape
    pe = k_pe.squeeze(1) if k_pe.dim() == 3 else k_pe
    # k_pe has a single head and is read once per token: broadcast it across
    # heads, then concat onto k_nope.
    pe_b = pe[:, None, :].expand(s, num_heads, pe.shape[-1])
    k = torch.cat([k_nope.to(torch.float32), pe_b.to(torch.float32)], dim=-1)

    fp8_max = torch.finfo(fp8_dtype).max
    k_out = (k * k_scale_inv).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    v_out = (
        (v.to(torch.float32) * v_scale_inv)
        .clamp(-fp8_max, fp8_max)
        .to(fp8_dtype)
    )
    return k_out, v_out
