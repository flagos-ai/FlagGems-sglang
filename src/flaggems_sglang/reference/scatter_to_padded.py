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

"""Reference implementation for diffusion/scatter_to_padded."""

import torch


def reference(out_unpad, inv_indices, batch_size, seqlen):
    _, num_heads, head_dim = out_unpad.shape
    hd = num_heads * head_dim
    src = out_unpad.contiguous().reshape(-1, hd)

    idx = inv_indices.to(torch.int64)
    valid = idx >= 0
    out = torch.zeros(
        batch_size * seqlen, hd, dtype=src.dtype, device=src.device
    )
    out[valid] = src[idx[valid]]
    return out.reshape(batch_size, seqlen, num_heads, head_dim)
