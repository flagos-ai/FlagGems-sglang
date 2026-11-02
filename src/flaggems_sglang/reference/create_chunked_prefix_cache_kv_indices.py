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


def reference(
    req_to_token,
    req_pool_indices,
    chunk_start_idx,
    chunk_seq_lens,
    chunk_cu_seq_lens,
    chunk_kv_indices,
):
    out = chunk_kv_indices.clone()
    for i in range(req_pool_indices.shape[0]):
        beg = int(chunk_cu_seq_lens[i])
        n = int(chunk_seq_lens[i])
        start = int(chunk_start_idx[i])
        pool = int(req_pool_indices[i])
        out[beg : beg + n] = req_to_token[pool, start : start + n]
    return out
