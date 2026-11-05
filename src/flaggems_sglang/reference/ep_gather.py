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

"""Reference implementation for moe/ep_gather."""

import torch


def reference(
    input_tensor, recv_topk_ids, recv_topk_weight, input_index, output_tensor
):
    rows = (
        input_tensor[input_index.long()].float()
        * recv_topk_weight.float()[..., None]
    )
    values = torch.where(
        (recv_topk_ids >= 0)[..., None], rows, torch.zeros_like(rows)
    )
    acc = torch.zeros(
        output_tensor.shape, dtype=torch.float32, device=input_tensor.device
    )
    for j in range(recv_topk_ids.shape[1]):
        acc += values[:, j]
    return acc.to(output_tensor.dtype)
