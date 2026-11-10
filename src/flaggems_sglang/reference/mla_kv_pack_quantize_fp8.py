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

"""Reference for attention/mla_kv_pack_quantize_fp8."""

import torch


def reference(
    k_nope,
    k_pe,
    v,
    k_scale_inv=1.0,
    v_scale_inv=1.0,
    fp8_dtype=torch.float8_e4m3fn,
):
    """Materialise the packed keys, then scale both parts and cast to fp8.

    The rotary key is written once and broadcast across every head, which is
    exactly the concatenation the kernel performs. The scaling runs in fp32
    against a bf16/fp16 source, so there is a single rounding at the fp8 cast
    rather than one per intermediate step.
    """
    tokens, heads, qk_nope = k_nope.shape
    qk_rope = int(k_pe.shape[-1])

    k_pe_2d = k_pe.squeeze(1) if k_pe.ndim == 3 else k_pe

    packed = torch.empty(
        (tokens, heads, qk_nope + qk_rope),
        dtype=k_nope.dtype,
        device=k_nope.device,
    )
    packed[..., :qk_nope] = k_nope
    packed[..., qk_nope:] = k_pe_2d.unsqueeze(1).expand(-1, heads, -1)

    key = (packed.float() * k_scale_inv).to(fp8_dtype)
    value = (v.float() * v_scale_inv).to(fp8_dtype)
    return key, value
