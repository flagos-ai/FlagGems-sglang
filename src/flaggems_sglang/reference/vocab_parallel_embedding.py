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
    input_,
    weight,
    org_vocab_start_index,
    org_vocab_end_index,
    num_org_vocab_padding,
    added_vocab_start_index,
    added_vocab_end_index,
):
    tok = input_.reshape(-1).long()
    in_org = (tok >= org_vocab_start_index) & (tok < org_vocab_end_index)
    in_added = (tok >= added_vocab_start_index) & (tok < added_vocab_end_index)
    valid = in_org | in_added

    added_offset = (
        added_vocab_start_index
        - (org_vocab_end_index - org_vocab_start_index)
        - num_org_vocab_padding
    )
    local_id = torch.where(
        in_org, tok - org_vocab_start_index, tok - added_offset
    )
    local_id = torch.where(valid, local_id, torch.zeros_like(local_id))

    out = weight[local_id]
    out = out * valid[:, None].to(out.dtype)
    return out.reshape(*input_.shape, weight.shape[1])
