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


def reference(
    x, g, weight, bias, activation="swish", eps=1e-5, is_rms_norm=True
):
    out_dtype = x.dtype
    xf = x.float()

    if is_rms_norm:
        var = (xf**2).mean(dim=-1, keepdim=True)
        x_hat = xf * (var + eps).rsqrt()
    else:
        mean = xf.mean(dim=-1, keepdim=True)
        var = ((xf - mean) ** 2).mean(dim=-1, keepdim=True)
        x_hat = (xf - mean) * (var + eps).rsqrt()

    y = x_hat
    if weight is not None:
        y = y * weight.float()
    if bias is not None:
        y = y + bias.float()

    gf = g.float()
    if activation in ("swish", "silu"):
        y = y * gf * gf.sigmoid()
    elif activation == "sigmoid":
        y = y * gf.sigmoid()

    return y.to(out_dtype)
