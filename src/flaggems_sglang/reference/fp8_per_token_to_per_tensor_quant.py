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

"""Reference implementation for moe/fp8_per_token_to_per_tensor_quant."""

_K_SCALE_BLOCK_SIZE = 128


def reference(x, x_scale, masked_m, output_scale):
    num_experts, m, k = x.shape
    out = x.new_zeros(num_experts, m, k)
    inv_out_scale = 1.0 / output_scale.reshape(()).float()

    for e in range(num_experts):
        n = int(masked_m[e].item())
        if n <= 0:
            continue
        xf = x[e, :n].float()
        scale_exp = (
            x_scale[e, :n]
            .float()
            .repeat_interleave(_K_SCALE_BLOCK_SIZE, dim=-1)
        )
        # The baseline does not clamp to the fp8 range before the cast, so
        # neither does this; the test scales stay small enough not to
        # overflow.
        val = xf * scale_exp * inv_out_scale
        out[e, :n] = val.to(out.dtype)

    return out
