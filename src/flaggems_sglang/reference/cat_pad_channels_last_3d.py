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

"""Reference implementation for diffusion/cat_pad_channels_last_3d."""

import torch


def reference(x, cache_x, padding, keep_cache_t):
    pw_l, pw_r, ph_t, ph_b, pt_front, pt_back = padding
    cache_t = 0 if cache_x is None else cache_x.shape[2]

    cat = x if cache_x is None else torch.cat([cache_x, x], dim=2)
    # The cache frames consume the temporal front padding; whatever is left
    # over is zero-filled.
    front = max(pt_front - cache_t, 0)
    out = torch.nn.functional.pad(
        cat, (pw_l, pw_r, ph_t, ph_b, front, pt_back)
    )
    out = out.contiguous(memory_format=torch.channels_last_3d)

    if keep_cache_t <= 0:
        return out
    # The next chunk's compact cache: the unpadded interior of the trailing
    # keep_cache_t frames.
    keep_t = min(keep_cache_t, out.shape[2])
    h0, h1 = ph_t, ph_t + x.shape[3]
    w0, w1 = pw_l, pw_l + x.shape[4]
    cache = out[:, :, out.shape[2] - keep_t :, h0:h1, w0:w1]
    return out, cache.contiguous(memory_format=torch.channels_last_3d)
