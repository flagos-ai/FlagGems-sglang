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

"""Reference implementation for kvcache/absorbed_bmm_concat_cast_q_fp8."""

import torch


def reference(q_fp8_pad, q_nope, w_kc, q_rope, num_heads):
    out = q_fp8_pad.clone()
    num_tokens = q_nope.shape[0]
    n_dim = w_kc.shape[-1]

    # Absorbed bmm in fp32, rounded to bf16, then cast to fp8: the same three
    # rounding stages the kernel epilogue uses. Both roundings matter.
    nope = torch.einsum(
        "thk,hkn->thn", q_nope.to(torch.float32), w_kc.to(torch.float32)
    ).to(torch.bfloat16)

    out[:num_tokens, :num_heads, :n_dim] = nope.to(out.dtype)
    out[:num_tokens, :num_heads, n_dim:] = q_rope.to(out.dtype)
    # Rows and heads outside [:num_tokens, :num_heads] are padding the kernel
    # never writes, so they keep whatever q_fp8_pad held.
    return out
