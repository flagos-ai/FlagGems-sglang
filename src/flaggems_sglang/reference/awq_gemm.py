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

"""Reference implementation for quantization/awq_gemm."""

import torch

from flaggems_sglang.reference.awq_dequantize import reference as _dequantize


def reference(input, qweight, scales, qzeros, split_k_iters):
    w_deq = _dequantize(qweight, scales, qzeros)
    return torch.matmul(input, w_deq)
