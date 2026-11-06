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

"""Reference implementation for utility/normalize_and_patchify."""

import torch


def reference(
    image, image_scale, image_bias, patch_size, padded_height, padded_width
):
    pad_h = padded_height - image.shape[-2]
    pad_w = padded_width - image.shape[-1]
    if pad_h > 0 or pad_w > 0:
        # Pad first: the padded pixels are normalized too, so they carry the
        # bias rather than zero.
        image = torch.nn.functional.pad(image, (0, pad_w, 0, pad_h), value=0.0)

    normalized = torch.addcmul(image_bias, image, image_scale)

    b, c, h, w = normalized.shape
    gh, gw = h // patch_size, w // patch_size
    tiled = normalized.view(b, c, gh, patch_size, gw, patch_size)
    return tiled.permute(0, 2, 4, 1, 3, 5).reshape(
        b, -1, c, patch_size, patch_size
    )
