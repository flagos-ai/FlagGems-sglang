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

"""Reference implementation for quantization/mxfp8_block_scaled_matmul."""

import torch


def reference(a, a_scale, b, b_scale, output_dtype):
    a_deq = a.to(torch.float32) * torch.exp2(
        a_scale.to(torch.float32) - 127
    ).repeat_interleave(32, dim=1)
    b_deq = b.to(torch.float32) * torch.exp2(
        b_scale.to(torch.float32) - 127
    ).repeat_interleave(32, dim=1)
    return (a_deq @ b_deq.t()).to(output_dtype)
