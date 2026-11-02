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

import torch


def reference(kv_buffer, loc, cache_k_nope, cache_k_rope):
    out = kv_buffer.clone()
    row = torch.cat([cache_k_nope, cache_k_rope], dim=-1).to(out.dtype)
    out[loc.long()] = row
    return out
