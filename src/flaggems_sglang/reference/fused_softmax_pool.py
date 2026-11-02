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


def reference(kv_score, head_dim):
    x = kv_score.to(torch.float32)
    kv, score = x[..., :head_dim], x[..., head_dim:]
    w = torch.softmax(score, dim=1)
    return (kv * w).sum(dim=1).to(kv_score.dtype)
