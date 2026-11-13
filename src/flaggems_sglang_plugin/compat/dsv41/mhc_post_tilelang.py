# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Portions adapted from SGLang (Copyright 2023-2026 SGLang Team, Apache-2.0).
"""HC=4 BF16 post-mix with independent token/hidden TileLang CTAs.

Import this module only on the TileLang execution path. The expression and
serial residual-channel order are deliberately identical to mhc_post_tilelang.
"""

import tilelang
import tilelang.language as T
import torch


def is_arch_support_pdl():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
    },
)
def _mhc_post_split_h_tilelang(
    a, b, c, d, out, hidden: int, h_blk: int = 1024, n_thr: int = 128
):
    n = T.dynamic("num_tokens")
    a: T.Tensor((n, 4, 4), T.float32)
    b: T.Tensor((n, 4, hidden), T.bfloat16)
    c: T.Tensor((n, 4), T.float32)
    d: T.Tensor((n, hidden), T.bfloat16)
    out: T.Tensor((n, 4, hidden), T.bfloat16)

    enable_pdl = is_arch_support_pdl()
    with T.Kernel(n, T.ceildiv(hidden, h_blk), threads=n_thr) as (token, tile):
        if enable_pdl:
            T.pdl_sync()
        x_shared = T.alloc_shared((4, h_blk), T.bfloat16)
        b_shared = T.alloc_shared((4, h_blk), T.bfloat16)
        d_shared = T.alloc_shared(h_blk, T.bfloat16)
        x_local = T.alloc_fragment((4, h_blk), T.float32)
        b_local = T.alloc_fragment((4, h_blk), T.float32)
        d_local = T.alloc_fragment(h_blk, T.float32)
        a_local = T.alloc_fragment((4, 4), T.float32)
        c_local = T.alloc_fragment(4, T.float32)
        T.copy(a[token, 0, 0], a_local)
        T.copy(c[token, 0], c_local)
        T.copy(b[token, 0, tile * h_blk], b_shared)
        T.copy(d[token, tile * h_blk], d_shared)
        T.copy(b_shared, b_local)
        T.copy(d_shared, d_local)
        for channel, h in T.Parallel(4, h_blk):
            x_local[channel, h] = c_local[channel] * d_local[h]
            for i in T.serial(4):
                x_local[channel, h] += a_local[i, channel] * b_local[i, h]
        T.copy(x_local, x_shared)
        T.copy(x_shared, out[token, 0, tile * h_blk])
        if enable_pdl:
            T.pdl_trigger()


def mhc_post_split_h_tilelang(x, residual, post, comb, *, block=1024, threads=128):
    """Allocate the same BF16 output as the existing mhc_post wrapper."""
    assert x.dtype == residual.dtype == torch.bfloat16
    assert post.dtype == comb.dtype == torch.float32
    assert residual.shape == (x.shape[0], 4, x.shape[1])
    assert x.shape[1] % block == 0
    assert all(t.is_contiguous() for t in (x, residual, post, comb))
    output = torch.empty_like(residual)
    _mhc_post_split_h_tilelang(
        comb,
        residual,
        post.squeeze(-1),
        x,
        output,
        hidden=x.shape[1],
        h_blk=block,
        n_thr=threads,
    )
    return output
