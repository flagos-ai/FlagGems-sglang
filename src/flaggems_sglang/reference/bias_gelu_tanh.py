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

"""Reference implementation for elementwise/bias_gelu_tanh."""

import torch


def reference(x, bias):
    # The addition is materialized in the input dtype before the activation:
    # keeping it in fp32 through the GELU would change the specified rounding.
    biased = (x.float() + bias.float()).to(x.dtype)
    return torch.nn.functional.gelu(biased, approximate="tanh")
