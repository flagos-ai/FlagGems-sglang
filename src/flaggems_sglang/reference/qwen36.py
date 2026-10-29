# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Torch oracles for the explicit SGLang v0.5.11 workload interfaces.

These are portable, executable oracles, not optimized candidates. In particular
the chunk oracle retains the native BF16 intermediate boundaries and all six
return slots. See docs/qwen36_workloads.md for the pinned source and scope.
"""

from collections import namedtuple

import torch
import torch.nn.functional as F


def gemma_rmsnorm(input, weight, eps=1e-6, out=None, enable_pdl=None):
    del enable_pdl
    x = input.float()
    y = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    y = (y * (1 + weight.float())).to(input.dtype)
    if out is None:
        return y.contiguous()
    out.copy_(y)
    return out


def gemma_fused_add_rmsnorm(
    input, residual, weight, eps=1e-6, enable_pdl=None
):
    del enable_pdl
    summed = input.float() + residual.float()
    y = summed * torch.rsqrt(summed.square().mean(-1, keepdim=True) + eps)
    input.copy_((y * (1 + weight.float())).to(input.dtype))
    residual.copy_(summed.to(residual.dtype))


def _round(value, dtype):
    return value.to(dtype).float()


def _safe_exp(value):
    # Native FLA safe_exp returns zero for positive differences.
    return torch.where(value <= 0, value, -float("inf")).exp()


def chunk_gated_delta_rule_fwd(
    q,
    k,
    v,
    g,
    beta,
    scale,
    initial_state,
    initial_state_indices,
    cu_seqlens=None,
    chunk_indices=None,
):
    """Native six-slot ABI at GDN_RECOMPUTE_SUPPRESS_LEVEL < 3.

    q/k are already normalized. FP32 linear algebra is rounded at the same
    BF16 storage boundaries as the native KKT/WY/state/output pipeline.
    """
    batch, tokens, key_heads, key_dim = k.shape
    heads, value_dim = v.shape[-2:]
    if cu_seqlens is None:
        sequences = [(b, 0, tokens) for b in range(batch)]
    else:
        if batch != 1:
            raise ValueError("Packed variable-length inputs require B=1.")
        bounds = cu_seqlens.cpu().tolist()
        sequences = [(0, start, end) for start, end in zip(bounds, bounds[1:])]
    planned = [
        (seq, offset // 64)
        for seq, (_, start, end) in enumerate(sequences)
        for offset in range(0, end - start, 64)
    ]
    if chunk_indices is not None and chunk_indices.cpu().tolist() != [
        list(x) for x in planned
    ]:
        raise ValueError(
            "chunk_indices must describe the ordered 64-token chunks."
        )
    cumulative = torch.empty_like(g, dtype=torch.float32)
    output = torch.empty_like(v)
    inverse = k.new_zeros(batch, tokens, heads, 64)
    # Native varlen allocation is [B, total_chunks, HV, V, K].
    histories = k.new_empty(
        batch, len(planned) // batch, heads, value_dim, key_dim
    )
    slots = initial_state_indices.cpu().tolist()
    expand = torch.arange(heads, device=k.device) // (heads // key_heads)
    history_index = 0
    for seq, (b, start, end) in enumerate(sequences):
        state = initial_state[slots[seq]].float().clone()
        for lo in range(start, end, 64):
            hi = min(lo + 64, end)
            length = hi - lo
            kk = k[b, lo:hi].index_select(1, expand).transpose(0, 1).float()
            qq = q[b, lo:hi].index_select(1, expand).transpose(0, 1).float()
            vv = v[b, lo:hi].transpose(0, 1).float()
            bb = beta[b, lo:hi].transpose(0, 1).float()
            gg = g[b, lo:hi].transpose(0, 1).float().cumsum(-1)
            cumulative[b, lo:hi] = gg.transpose(0, 1)
            decay = _safe_exp(gg.unsqueeze(-1) - gg.unsqueeze(-2))
            lower = (kk @ kk.transpose(-1, -2)) * decay * bb.unsqueeze(-1)
            lower = torch.tril(lower, diagonal=-1)
            identity = torch.eye(length, device=k.device).expand(heads, -1, -1)
            aa = torch.linalg.solve_triangular(
                identity + lower, identity, upper=False, unitriangular=True
            ).to(k.dtype)
            inverse[b, lo:hi, :, :length] = aa.transpose(0, 1)
            ww = _round(
                aa.float()
                @ _round(
                    kk * bb.unsqueeze(-1) * gg.exp().unsqueeze(-1), k.dtype
                ),
                k.dtype,
            )
            uu = _round(
                aa.float() @ _round(vv * bb.unsqueeze(-1), v.dtype), v.dtype
            )
            stored_state = _round(state, k.dtype)
            history_column = (
                history_index if cu_seqlens is not None else (lo - start) // 64
            )
            histories[b, history_column] = stored_state.to(k.dtype)
            transformed = uu - ww @ stored_state.transpose(-1, -2)
            new_v = _round(transformed, v.dtype)
            score = torch.tril((qq @ kk.transpose(-1, -2)) * decay)
            out = (qq @ stored_state.transpose(-1, -2)) * gg.exp().unsqueeze(
                -1
            )
            out = (out + _round(score, v.dtype) @ new_v) * scale
            output[b, lo:hi] = out.transpose(0, 1).to(v.dtype)
            weighted = _round(
                transformed * _safe_exp(gg[:, -1:] - gg).unsqueeze(-1),
                k.dtype,
            )
            state = state * gg[:, -1].exp()[:, None, None]
            state = state + weighted.transpose(-1, -2) @ kk
            history_index += 1
        initial_state[slots[seq]].copy_(state.to(initial_state.dtype))
    return cumulative, output, inverse, None, histories, None


def fused_recurrent_gated_delta_rule_packed_decode(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    scale,
    initial_state,
    out,
    ssm_state_indices,
    use_qk_l2norm_in_kernel=False,
):
    heads, value_dim, key_dim = initial_state.shape[-3:]
    key_heads = (mixed_qkv.shape[-1] - heads * value_dim) // (2 * key_dim)
    expand = torch.arange(heads, device=mixed_qkv.device) // (
        heads // key_heads
    )
    width = key_heads * key_dim
    q = mixed_qkv[:, :width].reshape(-1, key_heads, key_dim).float()
    k = mixed_qkv[:, width : 2 * width].reshape(-1, key_heads, key_dim).float()
    v = mixed_qkv[:, 2 * width :].reshape(-1, heads, value_dim).float()
    if use_qk_l2norm_in_kernel:
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    q, k = q[:, expand] * scale, k[:, expand]
    gate = -A_log.float().exp() * F.softplus(
        a.float() + dt_bias.float(), threshold=20
    )
    # The native packed kernel explicitly rounds sigmoid to b.dtype.
    strength = b.float().sigmoid().to(b.dtype).float()
    for row, slot in enumerate(ssm_state_indices.cpu().tolist()):
        if slot < 0:
            out[row].zero_()
            continue
        state = initial_state[slot].float() * gate[row].exp()[:, None, None]
        delta = v[row] - (state * k[row, :, None, :]).sum(-1)
        state = (
            state
            + (delta * strength[row, :, None])[:, :, None] * k[row, :, None, :]
        )
        out[row, 0].copy_((state * q[row, :, None, :]).sum(-1).to(out.dtype))
        initial_state[slot].copy_(state.to(initial_state.dtype))
    return out, initial_state


class StandardTopKOutput(
    namedtuple("StandardTopKOutput", "topk_weights topk_ids router_logits")
):
    __slots__ = ()

    @property
    def format(self):
        # SGLang TopKOutputFormat.STANDARD is IntEnum value 1.
        return 1


def select_experts(
    hidden_states,
    router_logits,
    topk_config,
    *,
    layer_id=None,
    num_token_non_padded=None,
    expert_location_dispatch_info=None,
):
    del layer_id
    if (
        num_token_non_padded is not None
        or expert_location_dispatch_info is not None
    ):
        raise NotImplementedError(
            "Only the documented unfiltered routing ABI is covered."
        )
    if hidden_states.shape[0] != router_logits.shape[0]:
        raise ValueError("Mismatched token counts.")
    if (
        topk_config.use_grouped_topk
        or topk_config.scoring_func != "softmax"
        or topk_config.num_fused_shared_experts != 0
        or topk_config.apply_routed_scaling_factor_on_output
        or any(
            getattr(topk_config, name) is not None
            for name in (
                "topk_group",
                "num_expert_group",
                "custom_routing_function",
                "correction_bias",
                "routed_scaling_factor",
                "fused_shared_experts_scaling_factor",
                "output_format",
            )
        )
    ):
        raise NotImplementedError(
            "Only the documented ungrouped softmax routing is covered."
        )
    probabilities = router_logits.float().softmax(-1)
    weights, ids = probabilities.topk(topk_config.top_k, dim=-1, sorted=False)
    if topk_config.renormalize:
        weights = weights / weights.sum(-1, keepdim=True)
    return StandardTopKOutput(weights, ids.to(torch.int32), router_logits)


def _fused_moe_kernel_sequence(
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
    config,
    down_config,
    down_moe_use_tma,
    *,
    b1,
    b2,
    use_fp8_w8a8,
    use_int8_w8a8,
    use_int8_w8a16,
    use_int4_w4a16,
    per_channel_quant,
    w1_scale,
    w2_scale,
    w1_zp,
    w2_zp,
    a1_scale,
    a2_scale,
    block_shape,
    activation,
    is_gated,
    no_combine,
    inplace,
    apply_router_weight_on_input,
    routed_scaling_factor,
    gemm1_alpha,
    gemm1_limit,
    filter_expert,
    hooks=None,
):
    if any(
        (
            use_fp8_w8a8,
            use_int8_w8a8,
            use_int8_w8a16,
            use_int4_w4a16,
            per_channel_quant,
            down_moe_use_tma,
            no_combine,
            apply_router_weight_on_input,
            filter_expert,
        )
    ):
        raise NotImplementedError(
            "Only the documented unquantized combined MoE is covered."
        )
    if (
        activation != "silu"
        or not is_gated
        or any(
            x is not None
            for x in (
                b1,
                b2,
                w1_scale,
                w2_scale,
                w1_zp,
                w2_zp,
                a1_scale,
                a2_scale,
                block_shape,
                hooks,
                gemm1_alpha,
                gemm1_limit,
            )
        )
    ):
        raise NotImplementedError(
            "Only the documented bias-free SiLU MoE is covered."
        )
    del (
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        config,
        down_config,
    )
    source = hidden_states.float()
    pieces = []
    width = w1.shape[1] // 2
    for row in range(source.shape[0]):
        ids = topk_ids[row].long()
        projection = torch.matmul(w1[ids].float(), source[row]).to(
            hidden_states.dtype
        )
        gated = (
            F.silu(projection[:, :width].float())
            * projection[:, width:].float()
        )
        gated = gated.to(hidden_states.dtype).float()
        down = torch.matmul(w2[ids].float(), gated.unsqueeze(-1)).squeeze(-1)
        down = (down * topk_weights[row, :, None]).to(hidden_states.dtype)
        pieces.append(down.float().sum(0).to(hidden_states.dtype))
    result = torch.stack(pieces)
    if routed_scaling_factor is not None:
        result = (result.float() * routed_scaling_factor).to(result.dtype)
    if inplace:
        hidden_states.copy_(result)
        return hidden_states
    return result


def silu_and_mul_out(input, out=None):
    half = input.shape[-1] // 2
    result = (
        F.silu(input[..., :half].float()) * input[..., half:].float()
    ).to(input.dtype)
    if out is None:
        return result
    out.copy_(result)
    return out
