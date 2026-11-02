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


def reference(q, padded_q, seq_lens_q, cu_seqlens_q):
    out = padded_q.clone()
    bs = cu_seqlens_q.shape[0] - 1
    for b in range(bs):
        s = int(seq_lens_q[b])
        beg = int(cu_seqlens_q[b])
        out[b, :s] = q[beg : beg + s]
    return out
