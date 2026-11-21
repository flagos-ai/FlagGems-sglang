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

"""Reference implementation for rope/fused_qk_complex_rope."""

import torch


def _rotate(x, freqs_cis):
    # Interleaved (real, imag) pairs along the head dim, one freq row per
    # token broadcast across heads.
    xc = torch.view_as_complex(
        x.to(torch.float32).reshape(*x.shape[:-1], -1, 2)
    )
    out = xc * freqs_cis.unsqueeze(-2)
    return torch.view_as_real(out).flatten(-2).to(x.dtype)


def reference(q, k, freqs_cis):
    return _rotate(q, freqs_cis), _rotate(k, freqs_cis)
