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
    input,
    gateup_input,
    src2dst,
    topk_ids,
    a1_scales,
    num_local_experts,
    topk,
    num_tokens,
    hidden_size,
):
    out = gateup_input.clone()
    inv = 1.0 / float(a1_scales) if a1_scales is not None else 1.0
    scaled = (input.float() * inv).to(out.dtype)
    flat_ids = topk_ids.reshape(-1)
    flat_dst = src2dst.reshape(-1)
    valid = flat_ids != num_local_experts
    src_rows = scaled.repeat_interleave(topk, dim=0)
    out[flat_dst[valid].long()] = src_rows[valid]
    return out
