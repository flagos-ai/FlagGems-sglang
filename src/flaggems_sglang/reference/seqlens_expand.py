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


def reference(extend_seq_lens, seq_lens, total_len, max_q_len):
    # out[offset[i] : offset[i] + qo_len[i]] = clamp(kv_len[i] - qo_len[i] + 1 + arange(qo_len[i]), min=0)
    #   offset[i] = exclusive_cumsum(extend_seq_lens)
    # Vectorized: repeat each request's base to its qo_len rows and add the
    # in-request position.
    device = extend_seq_lens.device
    N = extend_seq_lens.shape[0]

    offsets = torch.zeros(N + 1, dtype=torch.int32, device=device)
    torch.cumsum(extend_seq_lens, dim=0, out=offsets[1:])

    req_idx = torch.repeat_interleave(
        torch.arange(N, device=device), extend_seq_lens
    )
    # torch.arange defaults to int64; keep every intermediate in int32 so the
    # output matches the spec'd int32 dtype.
    global_pos = torch.arange(total_len, device=device, dtype=torch.int32)
    local_pos = global_pos - offsets[req_idx]
    base = seq_lens - extend_seq_lens + 1
    vals = (base[req_idx] + local_pos).clamp(min=0)
    return vals.to(torch.int32)
