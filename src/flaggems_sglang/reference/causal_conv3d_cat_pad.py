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

"""Reference implementation for diffusion/causal_conv3d_cat_pad."""

import torch


def reference(x, cache_x, padding):
    w_l, w_r, h_t, h_b, d_l, d_r = padding
    # The cache frames consume part of the temporal front padding.
    d_l = d_l - cache_x.shape[2]
    cat = torch.cat([cache_x, x], dim=2)
    # F.pad's order is (w_left, w_right, h_top, h_bottom, t_front, t_back).
    return torch.nn.functional.pad(cat, (w_l, w_r, h_t, h_b, d_l, d_r))
