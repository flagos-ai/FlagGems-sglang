"""Mamba2 inter-chunk state-passing scan (Triton).

Semantics (per independent lane (b, h, d)):
    cur = initial_states[b, h, d] (float32) or 0.0
    for c in 0..C-1:
        out[b, c, h, d] = cast(cur, states.dtype)     # state BEFORE the update
        cur = cur * exp(dA_cumsum[b, h, c, L-1]) + states[b, c, h, d]
    final_states[b, h, d] = cur                        # float32

Design A: one program owns a [BLOCK_H, BLOCK_D] tile of one batch and walks the
whole chunk axis sequentially, keeping `cur` in registers, so the operator costs
exactly one streaming read of `states` plus one streaming write of `out`.

Only D is a constexpr: that makes the "divisible tile" test compile-time and
enables an entirely unmasked vector path for BLOCK_H == 1. C and H stay runtime
scalars because specializing either of them measured 3.7-4.8% slower on the
many-program workload (rounds 5 and 9). Both traffic streams are declared
non-temporal because neither is ever reused.
"""

import torch
import triton
import triton.language as tl


_CONFIGS = [
    # Mixed grid: BLOCK_H=1 with small BLOCK_D for the many-program shape,
    # BLOCK_H=2/4 with small BLOCK_D for extra row parallelism, and large
    # BLOCK_D for maximum bytes in flight when there are few programs.
    triton.Config({"BLOCK_H": 1, "BLOCK_D": 128}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK_H": 1, "BLOCK_D": 256}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK_H": 1, "BLOCK_D": 512}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK_H": 1, "BLOCK_D": 256}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_H": 1, "BLOCK_D": 512}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_H": 1, "BLOCK_D": 1024}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_H": 1, "BLOCK_D": 1024}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_H": 1, "BLOCK_D": 2048}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_H": 1, "BLOCK_D": 2048}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK_H": 2, "BLOCK_D": 128}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK_H": 2, "BLOCK_D": 256}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK_H": 2, "BLOCK_D": 256}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_H": 2, "BLOCK_D": 512}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_H": 4, "BLOCK_D": 128}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK_H": 4, "BLOCK_D": 256}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_H": 1, "BLOCK_D": 1024}, num_warps=2, num_stages=2),
]


@triton.autotune(configs=_CONFIGS, key=["C", "H", "D"])
@triton.jit
def _state_passing_kernel(
    states_ptr,
    dA_ptr,
    init_ptr,
    out_ptr,
    final_ptr,
    C,
    H,
    L,
    D: tl.constexpr,
    sB,
    sC,
    sH,
    sD,
    aB,
    aH,
    aC,
    aL,
    iB,
    iH,
    iD,
    fB,
    fH,
    fD,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_INIT: tl.constexpr,
):
    pid_d = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    cur = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)

    dA_off = pid_b * aB + offs_h * aH + (L - 1) * aL
    # states / out tiles: out is freshly allocated contiguous
    s_off = pid_b * sB + offs_h[:, None] * sH + offs_d[None, :] * sD
    o_off = pid_b * (C * H * D) + offs_h[:, None] * D + offs_d[None, :]
    i_off = pid_b * iB + offs_h[:, None] * iH + offs_d[None, :] * iD
    f_off = pid_b * fB + offs_h[:, None] * fH + offs_d[None, :] * fD

    if BLOCK_H == 1 and D % BLOCK_D == 0:
        # no partial head tile and no tail d tile -> fully unmasked vector path
        if HAS_INIT:
            cur = tl.load(init_ptr + i_off).to(tl.float32)
        for c in range(0, C):
            a = tl.load(dA_ptr + dA_off + c * aC).to(tl.float32)
            s = tl.load(
                states_ptr + s_off + c * sC, eviction_policy="evict_first"
            ).to(tl.float32)
            tl.store(
                out_ptr + o_off + c * (H * D),
                cur.to(out_ptr.dtype.element_ty),
                eviction_policy="evict_first",
            )
            cur = cur * tl.exp(a)[:, None] + s
        tl.store(final_ptr + f_off, cur)
    else:
        mask_h = offs_h < H
        mask = mask_h[:, None] & (offs_d < D)[None, :]
        if HAS_INIT:
            cur = tl.load(init_ptr + i_off, mask=mask, other=0.0).to(
                tl.float32
            )
        for c in range(0, C):
            a = tl.load(dA_ptr + dA_off + c * aC, mask=mask_h, other=0.0).to(
                tl.float32
            )
            s = tl.load(states_ptr + s_off + c * sC, mask=mask, other=0.0).to(
                tl.float32
            )
            tl.store(
                out_ptr + o_off + c * (H * D),
                cur.to(out_ptr.dtype.element_ty),
                mask=mask,
            )
            cur = cur * tl.exp(a)[:, None] + s
        tl.store(final_ptr + f_off, cur, mask=mask)


def run(states, dA_cumsum, initial_states=None):
    B, C, H, D = states.shape
    out = torch.empty((B, C, H, D), dtype=states.dtype, device=states.device)
    final_states = torch.empty(
        (B, H, D), dtype=torch.float32, device=states.device
    )
    if final_states.numel() == 0:
        return out, final_states

    L = dA_cumsum.shape[-1]
    HAS_INIT = initial_states is not None

    if HAS_INIT:
        init_ptr = initial_states
        iB = initial_states.stride(0)
        iH = initial_states.stride(1)
        iD = initial_states.stride(2)
    else:
        init_ptr = out
        iB = iH = iD = 0

    grid = lambda meta: (
        triton.cdiv(D, meta["BLOCK_D"]),
        triton.cdiv(H, meta["BLOCK_H"]),
        B,
    )
    _state_passing_kernel[grid](
        states,
        dA_cumsum,
        init_ptr,
        out,
        final_states,
        C,
        H,
        L,
        D,
        states.stride(0),
        states.stride(1),
        states.stride(2),
        states.stride(3),
        dA_cumsum.stride(0),
        dA_cumsum.stride(1),
        dA_cumsum.stride(2),
        dA_cumsum.stride(3),
        iB,
        iH,
        iD,
        final_states.stride(0),
        final_states.stride(1),
        final_states.stride(2),
        HAS_INIT=HAS_INIT,
    )
    return out, final_states


state_passing = run

__all__ = ["state_passing"]
