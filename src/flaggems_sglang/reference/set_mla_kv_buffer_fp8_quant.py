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

"""Reference for set_mla_kv_buffer_fp8_quant."""

import torch


def reference(kv_buffer, loc, cache_k_nope, cache_k_rope, fp8_dtype):
    out = kv_buffer.clone()
    buffer = out if out.dtype == fp8_dtype else out.view(fp8_dtype)
    row = torch.cat([cache_k_nope, cache_k_rope], dim=-1)
    width = row.shape[-1]
    index = loc.long()
    # Out-of-range targets are skipped rather than wrapped, so the reference
    # and the kernel agree on slots that cannot name a real row.
    inside = (index >= 0) & (index < buffer.shape[0])
    if bool(inside.any()):
        # ``index_put`` will not cast for us, so the row is converted first.
        buffer[index[inside], :width] = row[inside].to(buffer.dtype)
    return out
