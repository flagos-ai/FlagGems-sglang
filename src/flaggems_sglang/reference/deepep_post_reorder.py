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


def reference(
    down_output,
    output,
    src2dst,
    topk_ids,
    topk_weights,
    topk,
    hidden_size,
    routed_scaling_factor,
):
    acc = torch.zeros(output.shape, dtype=torch.float32, device=output.device)
    for i in range(topk):
        dst = src2dst[:, i]
        valid = (dst >= 0).float()
        rows = down_output[dst.clamp(min=0).long()].float()
        w = (
            topk_weights[:, i].to(down_output.dtype).float()
            * routed_scaling_factor
        )
        acc += rows * (w * valid)[:, None]
    return acc.to(output.dtype)
