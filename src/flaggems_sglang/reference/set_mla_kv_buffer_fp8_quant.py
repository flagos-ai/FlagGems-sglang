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

"""Reference implementation for kvcache/set_mla_kv_buffer_fp8_quant."""

import torch


def reference(kv_buffer, loc, cache_k_nope, cache_k_rope, fp8_dtype):
    out = kv_buffer.clone()
    # The buffer is allocated in the source dtype and reinterpreted as fp8,
    # so the cast happens at the store.
    view = out.view(fp8_dtype)
    row = torch.cat([cache_k_nope, cache_k_rope], dim=-1)
    view[loc.to(torch.int64)] = row.to(fp8_dtype)
    return out
