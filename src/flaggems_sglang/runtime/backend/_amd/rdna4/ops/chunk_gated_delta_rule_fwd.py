# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GDN prefill with compact Q/K normalization on gfx1201."""

import torch
import triton
import triton.language as tl

from flaggems_sglang.ops.chunk_gated_delta_rule_fwd import (
    _chunk_gated_delta_rule_fwd,
)


@triton.jit
def _qk_norm_factor_kernel(
    q_ptr,
    k_ptr,
    q_factor,
    k_factor,
    ROWS,
    RPB: tl.constexpr,
    K: tl.constexpr,
):
    """Compute the inverse L2 norm per contiguous Q or K row."""
    src, dst = q_ptr, q_factor
    if tl.program_id(1) != 0:
        src, dst = k_ptr, k_factor
    row = tl.program_id(0) * RPB + tl.arange(0, RPB)
    col = tl.arange(0, K)
    x = tl.load(
        src + row[:, None] * K + col[None, :],
        mask=(row < ROWS)[:, None],
        other=0.0,
    ).to(tl.float32)
    norm = tl.rsqrt(tl.sum(x * x, axis=1) + 1e-6)
    tl.store(dst + row, norm, mask=row < ROWS)


def _prepare_qk_factors(q, k, v):
    """Return compact norms and the measured V tile for supported prefills."""
    batch, tokens, heads, k_dim = q.shape
    if (
        q.dtype != torch.bfloat16
        or (heads, k_dim, v.shape[-1]) != (16, 128, 128)
        or (batch, tokens) not in ((2, 2048), (1, 4096))
        or not q.is_contiguous()
        or not k.is_contiguous()
        or torch.cuda.get_device_properties(q.device).gcnArchName.split(
            ":", 1
        )[0]
        != "gfx1201"
    ):
        return None

    rows = batch * tokens * heads
    rpb = 256 // k_dim
    nq = torch.empty((rows,), dtype=torch.float32, device=q.device)
    nk = torch.empty_like(nq)
    _qk_norm_factor_kernel[(triton.cdiv(rows, rpb), 2)](
        q, k, nq, nk, rows, RPB=rpb, K=k_dim, num_warps=1
    )
    # The wider tile reduces repeated Q/K and factor reads across programs.
    return nq, nk, 32


def chunk_gated_delta_rule_fwd(
    q,
    k,
    v,
    g,
    beta,
    scale=None,
    initial_state=None,
    output_final_state=False,
    cu_seqlens=None,
    use_qk_l2norm_in_kernel=False,
):
    """Use the gfx1201 Q/K factor path for the measured BF16 shapes."""
    return _chunk_gated_delta_rule_fwd(
        q,
        k,
        v,
        g,
        beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        prepare_qk_factors=_prepare_qk_factors,
    )


__all__ = ["chunk_gated_delta_rule_fwd"]
