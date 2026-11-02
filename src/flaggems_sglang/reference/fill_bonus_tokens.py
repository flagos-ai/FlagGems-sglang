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


def reference(accept_tokens, accept_lens, bonus_tokens, accept_stride):
    flat = accept_tokens.reshape(-1)
    idx = (
        torch.arange(accept_lens.shape[0], device=flat.device) * accept_stride
        + accept_lens.long()
        - 1
    )
    return flat[idx].to(bonus_tokens.dtype)
