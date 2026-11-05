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

"""Reference for SGLang MoE top-k gating (fused softmax + top-k).

Mirrors ``sgl_kernel.topk_softmax``: a fp32 softmax over the expert axis, then
the ``k`` largest experts per token row written (descending by weight) into the
preallocated ``topk_weights`` (fp32) / ``topk_ids`` (int32) buffers. With
``renormalize`` the selected weights are divided by their own sum.

The buffers are passed for ABI parity but the pure reference returns fresh
tensors so correctness comparison does not depend on in-place effects.
"""

import torch


def reference(topk_weights, topk_ids, gating_output, renormalize=False):
    k = topk_weights.shape[1]
    probs = torch.softmax(gating_output.float(), dim=-1)
    w, ids = torch.topk(probs, k, dim=-1)
    if renormalize:
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
    return w.contiguous(), ids.to(torch.int32).contiguous()
