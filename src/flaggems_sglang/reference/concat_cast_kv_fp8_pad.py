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

"""Reference for kv_cache/concat_cast_kv_fp8_pad."""

import torch


def reference(out, k, k_rope, num_tokens):
    """Concatenate the NoPE latent and the rope segment, cast to fp8, and zero
    the trailing pad band.

    Mirrors the fused kernel exactly: rows ``[0, num_tokens)`` become
    ``cat([k, k_rope], -1).to(float8_e4m3fn)``, rows ``[num_tokens, total)``
    become zero bytes.
    """
    nope = k.shape[-1]
    rope = k_rope.shape[-1] if k_rope.numel() else 0
    head = nope + rope
    total_rows = out.shape[0]

    buf = torch.zeros(
        total_rows, head, dtype=torch.float8_e4m3fn, device=out.device
    )
    if num_tokens > 0 and head > 0:
        folded = torch.cat(
            [k.reshape(num_tokens, nope), k_rope.reshape(num_tokens, rope)],
            dim=-1,
        )
        buf[:num_tokens] = folded.to(torch.float8_e4m3fn)
    return buf
