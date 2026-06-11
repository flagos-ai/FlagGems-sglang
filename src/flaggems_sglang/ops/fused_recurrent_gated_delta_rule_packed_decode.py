"""Unified deferred-Q+V Triton kernel for fused recurrent gated delta rule
packed decode.

v27: Q loaded alongside V after gating (Phase 3), hiding Q load latency behind
the delta rule computation without increasing register pressure during gating.

Dispatch strategy:
  B <= 4:  vtile_loop kernel — single block per (token, head) processes
           all NV V-tiles sequentially, eliminating redundant Q/K/gating loads.
  B > 4:   per_vtile kernel  — grid=(NV, B*HV) transposed, each block
           processes one V-tile with maximal SM occupancy.
"""

import logging

import torch
import triton
import triton.language as tl

from flaggems_sglang.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def _kernel_vtile_loop(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    o,
    h0,
    ht,
    ssm_state_indices,
    scale,
    stride_mixed_qkv_tok: tl.constexpr,
    stride_a_tok: tl.constexpr,
    stride_b_tok: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NV: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
):
    """V-tile loop kernel for B <= 4. Single block per (token, head) processes
    all NV V-tiles sequentially, eliminating redundant Q/K/gating loads."""

    i_nh = tl.program_id(0)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    o_k = tl.arange(0, BK)
    mask_k = o_k < K

    state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq).to(
        tl.int64
    )

    # Handle padding tokens (state_idx == -1): write zeros
    if state_idx < 0:
        p_o = o + (i_n * HV + i_hv) * V
        for i_v in range(NV):
            o_v = i_v * BV + tl.arange(0, BV)
            mask_v = o_v < V
            zero = tl.zeros([BV], dtype=tl.float32).to(o.dtype.element_ty)
            tl.store(p_o + o_v, zero, mask=mask_v)
        return

    # Load Q and K for this (token, head)
    p_mixed = mixed_qkv + i_n * stride_mixed_qkv_tok
    q_off = i_h * K + o_k
    k_off = (H * K) + i_h * K + o_k
    b_q = tl.load(p_mixed + q_off, mask=mask_k, other=0).to(tl.float32)
    b_k = tl.load(p_mixed + k_off, mask=mask_k, other=0).to(tl.float32)

    if USE_QK_L2NORM_IN_KERNEL:
        b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
    b_q = b_q * scale

    # Load gating scalars
    a_val = tl.load(a + i_n * stride_a_tok + i_hv).to(tl.float32)
    b_val = tl.load(b + i_n * stride_b_tok + i_hv).to(tl.float32)
    A_log_val = tl.load(A_log + i_hv).to(tl.float32)
    dt_bias_val = tl.load(dt_bias + i_hv).to(tl.float32)

    # Gating: softplus(dt_bias + a) * (-exp(A_log)) -> decay
    x = a_val + dt_bias_val
    softplus_x = tl.where(x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x)
    g_val = -tl.exp(A_log_val) * softplus_x
    decay = tl.exp(g_val)
    beta_val = tl.sigmoid(b_val)

    # Precompute base addresses for loop
    base_h = state_idx * stride_init_state_token + i_hv * V * K
    base_v = (2 * H * K) + i_hv * V
    base_o = (i_n * HV + i_hv) * V
    tile_stride = BV * K

    o_v0 = tl.arange(0, BV)
    state_off = o_v0[:, None] * K + o_k[None, :]

    # Process all V-tiles sequentially for this (token, head)
    for i_v in range(NV):
        o_v = i_v * BV + o_v0
        mask_v = o_v < V
        mask_h = mask_v[:, None] & mask_k[None, :]

        # Load state tile
        p_h = h0 + base_h + i_v * tile_stride + state_off
        b_h = tl.load(p_h, mask=mask_h, other=0).to(tl.float32)

        # Load V tile
        b_v = tl.load(p_mixed + base_v + o_v, mask=mask_v, other=0).to(
            tl.float32
        )

        # Delta rule: h = decay * h, v = v - sum(h * k) * beta
        b_h *= decay
        b_v -= tl.sum(b_h * b_k[None, :], 1)
        b_v *= beta_val
        b_h += b_v[:, None] * b_k[None, :]

        # Output: dot(h, q)
        b_o = tl.sum(b_h * b_q[None, :], 1)

        # Store output and updated state
        tl.store(o + base_o + o_v, b_o.to(o.dtype.element_ty), mask=mask_v)
        tl.store(
            ht + base_h + i_v * tile_stride + state_off,
            b_h.to(ht.dtype.element_ty),
            mask=mask_h,
        )


@triton.jit
def _kernel_per_vtile(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    o,
    h0,
    ht,
    ssm_state_indices,
    scale,
    stride_mixed_qkv_tok: tl.constexpr,
    stride_a_tok: tl.constexpr,
    stride_b_tok: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
):
    """Per-V-tile kernel for B > 4. Grid=(NV, B*HV) transposed.

    Staged load ordering minimizes register pressure during gating:
      Phase 1: Load state (long latency) + K + gating scalars.
      Phase 2: K l2norm (if enabled) + gating (register-only, SFU-heavy).
      Phase 3: Decay state + load V + load Q (Q loaded with V, not serial).
      Phase 4: Delta rule update (hides Q/V load latency from Phase 3).
      Phase 5: Q l2norm + scale + output.
      Phase 6: Store results.

    Key insight: loading Q alongside V after gating hides Q load latency
    behind both the V load and the delta rule computation, without the
    register pressure penalty of having Q live during the gating phase."""

    i_v = tl.program_id(0)
    i_nh = tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    o_k = tl.arange(0, BK)
    mask_k = o_k < K
    o_v = i_v * BV + tl.arange(0, BV)
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq).to(
        tl.int64
    )

    # Handle padding tokens (state_idx == -1): write zeros
    if state_idx < 0:
        p_o = o + (i_n * HV + i_hv) * V + o_v
        zero = tl.zeros([BV], dtype=tl.float32).to(o.dtype.element_ty)
        tl.store(p_o, zero, mask=mask_v)
        return

    state_base = state_idx * stride_init_state_token + i_hv * V * K
    state_off = o_v[:, None] * K + o_k[None, :]

    # Phase 1: Load state (8KB, long latency) + K (256B) + gating scalars.
    p_h = h0 + state_base + state_off
    b_h = tl.load(p_h, mask=mask_h, other=0).to(tl.float32)

    p_mixed = mixed_qkv + i_n * stride_mixed_qkv_tok
    k_off = (H * K) + i_h * K + o_k
    b_k = tl.load(p_mixed + k_off, mask=mask_k, other=0).to(tl.float32)

    a_val = tl.load(a + i_n * stride_a_tok + i_hv).to(tl.float32)
    b_val = tl.load(b + i_n * stride_b_tok + i_hv).to(tl.float32)
    A_log_val = tl.load(A_log + i_hv).to(tl.float32)
    dt_bias_val = tl.load(dt_bias + i_hv).to(tl.float32)

    # Phase 2: K l2norm (if enabled) + gating (register-only, SFU-heavy).
    if USE_QK_L2NORM_IN_KERNEL:
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)

    x = a_val + dt_bias_val
    softplus_x = tl.where(x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x)
    g_val = -tl.exp(A_log_val) * softplus_x
    decay = tl.exp(g_val)
    beta_val = tl.sigmoid(b_val)

    # Phase 3: Decay state + load V + load Q.
    # Q loaded here (not Phase 5) to hide its latency behind delta rule.
    b_h *= decay
    v_off = (2 * H * K) + i_hv * V + o_v
    q_off = i_h * K + o_k
    b_v = tl.load(p_mixed + v_off, mask=mask_v, other=0).to(tl.float32)
    b_q = tl.load(p_mixed + q_off, mask=mask_k, other=0).to(tl.float32)

    # Phase 4: Delta rule update (reduction + outer product).
    # Hides Q and V load latency from Phase 3.
    b_v -= tl.sum(b_h * b_k[None, :], 1)
    b_v *= beta_val
    b_h += b_v[:, None] * b_k[None, :]

    # Phase 5: Q l2norm + scale + output.
    if USE_QK_L2NORM_IN_KERNEL:
        b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
    b_q = b_q * scale
    b_o = tl.sum(b_h * b_q[None, :], 1)

    # Phase 6: Store results.
    p_o = o + (i_n * HV + i_hv) * V + o_v
    tl.store(p_o, b_o.to(o.dtype.element_ty), mask=mask_v)
    tl.store(
        ht + state_base + state_off,
        b_h.to(ht.dtype.element_ty),
        mask=mask_h,
    )


def fused_recurrent_gated_delta_rule_packed_decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = False,
):
    """Triton-optimized fused recurrent gated delta rule for packed decode.

    Computes the delta-rule state update and output projection for every
    (token, head) pair in a single fused kernel, with hybrid dispatch
    based on batch size for optimal SM utilization.

    Args:
        mixed_qkv: Packed QKV tensor of shape (B, 2*H*K + HV*V).
        a: Input-dependent gating logits of shape (B, HV).
        b: Input-dependent gating logits of shape (B, HV).
        A_log: Log of state transition matrix of shape (HV,).
        dt_bias: Delta bias of shape (HV,).
        scale: QK scaling factor (typically K ** -0.5).
        initial_state: State pool of shape (pool_size, HV, V, K).
        out: Output buffer of shape (B, 1, HV, V).
        ssm_state_indices: State indices per token of shape (B,).
        use_qk_l2norm_in_kernel: Whether to apply Q/K L2 norm in kernel.

    Returns:
        tuple: (out, initial_state) — updated output and state tensors.
    """
    logger.debug(
        "FLAGGEMS_SGLANG FUSED_RECURRENT_GATED_DELTA_RULE_PACKED_DECODE"
    )

    B = mixed_qkv.shape[0]
    HV, V, K = initial_state.shape[-3:]
    qkv_dim = mixed_qkv.shape[1]
    qk_dim = qkv_dim - HV * V
    q_dim = qk_dim // 2
    H = q_dim // K

    BK = triton.next_power_of_2(K)

    stride_mixed_qkv_tok = mixed_qkv.stride(0)
    stride_a_tok = a.stride(0)
    stride_b_tok = b.stride(0)
    stride_init_state_token = initial_state.stride(0)
    stride_final_state_token = initial_state.stride(0)
    stride_indices_seq = ssm_state_indices.stride(0)

    # BV is computed dynamically from V to avoid hardcoded block sizes.
    BV = min(triton.next_power_of_2(V), 32)
    NV = triton.cdiv(V, BV)

    with torch_device_fn.device(mixed_qkv.device):
        if B <= 4:
            # Small batch: vtile_loop kernel with num_warps=4 for maximum
            # per-block parallelism; single grid dimension = B * HV.
            grid = (B * HV,)
            num_warps = 4
            num_stages = 4
            _kernel_vtile_loop[grid](
                mixed_qkv=mixed_qkv,
                a=a,
                b=b,
                A_log=A_log,
                dt_bias=dt_bias,
                o=out,
                h0=initial_state,
                ht=initial_state,
                ssm_state_indices=ssm_state_indices,
                scale=scale,
                stride_mixed_qkv_tok=stride_mixed_qkv_tok,
                stride_a_tok=stride_a_tok,
                stride_b_tok=stride_b_tok,
                stride_init_state_token=stride_init_state_token,
                stride_final_state_token=stride_final_state_token,
                stride_indices_seq=stride_indices_seq,
                H=H,
                HV=HV,
                K=K,
                V=V,
                BK=BK,
                BV=BV,
                NV=NV,
                SOFTPLUS_THRESHOLD=20.0,
                USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
                num_warps=num_warps,
                num_stages=num_stages,
            )
        else:
            # Large batch: per_vtile kernel with grid transposition (NV, B*HV)
            # for better L2 cache locality; num_warps=1 per block for max SM
            # occupancy since B*HV already provides sufficient parallelism.
            grid = (NV, B * HV)  # type: ignore[assignment]
            _kernel_per_vtile[grid](
                mixed_qkv=mixed_qkv,
                a=a,
                b=b,
                A_log=A_log,
                dt_bias=dt_bias,
                o=out,
                h0=initial_state,
                ht=initial_state,
                ssm_state_indices=ssm_state_indices,
                scale=scale,
                stride_mixed_qkv_tok=stride_mixed_qkv_tok,
                stride_a_tok=stride_a_tok,
                stride_b_tok=stride_b_tok,
                stride_init_state_token=stride_init_state_token,
                stride_final_state_token=stride_final_state_token,
                stride_indices_seq=stride_indices_seq,
                H=H,
                HV=HV,
                K=K,
                V=V,
                BK=BK,
                BV=BV,
                SOFTPLUS_THRESHOLD=20.0,
                USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
                num_warps=1,
                num_stages=3,
            )
    return out, initial_state
