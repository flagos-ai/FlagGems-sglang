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


def reference(dst, src, dst_indices_raw, step_indices_raw):
    out = dst.clone()
    valid = step_indices_raw >= 0
    if not bool(valid.any()):
        return out
    req = torch.nonzero(valid, as_tuple=True)[0]
    dst_idx = dst_indices_raw[req].to(torch.int64)
    step_idx = step_indices_raw[req].to(torch.int64)
    # dst[:, dst_indices[i]] = src[:, i, step_indices[i]] for every valid i
    out[:, dst_idx] = src[:, req, step_idx]
    return out
