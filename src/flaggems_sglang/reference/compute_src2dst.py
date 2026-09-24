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


def reference(reorder_ids, num_toks):
    src2dst = torch.empty(
        num_toks, dtype=torch.int32, device=reorder_ids.device
    )
    dst = torch.arange(num_toks, dtype=torch.int32, device=reorder_ids.device)
    src2dst[reorder_ids.long()] = dst
    return src2dst
