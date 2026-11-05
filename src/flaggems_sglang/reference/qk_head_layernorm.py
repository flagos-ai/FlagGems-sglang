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

"""Reference implementation for diffusion/qk_head_layernorm."""

import torch


def _ln(x, eps):
    return torch.nn.functional.layer_norm(
        x.to(torch.float32), (x.shape[-1],), eps=eps
    ).to(x.dtype)


def reference(q, k, eps):
    # Per-head LayerNorm, no affine: the last dim is the head dim.
    return _ln(q, eps), _ln(k, eps)
