import torch
import triton
import triton.language as tl

_CONFIGS = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=16, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=16, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4, num_stages=3),
]


@triton.autotune(configs=_CONFIGS, key=["D", "IS_CAUSAL", "SEQ_BUCKET"])
@triton.jit
def _context_attn_fwd(
    Q,
    K,
    V,
    O,
    B_START,
    B_LEN,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vt,
    stride_vh,
    stride_vd,
    stride_ot,
    stride_oh,
    stride_od,
    qk_scale,
    SEQ_BUCKET,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    D: tl.constexpr,
    EXACT_D: tl.constexpr,
    OFF64: tl.constexpr,
):
    pid_m = tl.program_id(0)
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)
    num_pid_m = tl.num_programs(0)

    seq_start = tl.load(B_START + off_b).to(tl.int64)
    seq_len = tl.load(B_LEN + off_b).to(tl.int32)

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D

    q_base = Q + seq_start * stride_qt + off_h * stride_qh
    k_base = K + seq_start * stride_kt + off_h * stride_kh
    v_base = V + seq_start * stride_vt + off_h * stride_vh
    o_base = O + seq_start * stride_ot + off_h * stride_oh

    num_m = (seq_len + (BLOCK_M - 1)) // BLOCK_M

    for m_blk in range(pid_m, num_m, num_pid_m):
        offs_m = m_blk * BLOCK_M + tl.arange(0, BLOCK_M)
        m_mask = offs_m < seq_len

        # element offsets stay in int32 unless the tensor is too large for them
        if OFF64:
            m_off = offs_m.to(tl.int64) * stride_qt
        else:
            m_off = offs_m * stride_qt
        q_ptrs = q_base + m_off[:, None] + offs_d[None, :] * stride_qd
        if EXACT_D:
            q = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)
        else:
            q = tl.load(
                q_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0
            )
        q = q.to(tl.float32) * qk_scale

        m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)

        if IS_CAUSAL:
            hi = tl.minimum(seq_len, (m_blk + 1) * BLOCK_M)
            # every key in [0, n_full) is in range and satisfies n <= m for all rows
            n_full = ((m_blk * BLOCK_M) // BLOCK_N) * BLOCK_N
        else:
            hi = seq_len
            n_full = (seq_len // BLOCK_N) * BLOCK_N

        # interior key blocks: no causal test and no key-validity mask, so the
        # loads are unmasked whenever BLOCK_D == D
        for start_n in range(0, n_full, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            if OFF64:
                k_off = offs_n.to(tl.int64) * stride_kt
                v_off = offs_n.to(tl.int64) * stride_vt
            else:
                k_off = offs_n * stride_kt
                v_off = offs_n * stride_vt
            k_ptrs = k_base + k_off[:, None] + offs_d[None, :] * stride_kd
            v_ptrs = v_base + v_off[:, None] + offs_d[None, :] * stride_vd
            if EXACT_D:
                k_tile = tl.load(k_ptrs)
                v_tile = tl.load(v_ptrs)
            else:
                k_tile = tl.load(k_ptrs, mask=d_mask[None, :], other=0.0)
                v_tile = tl.load(v_ptrs, mask=d_mask[None, :], other=0.0)

            qk = tl.dot(
                q, tl.trans(k_tile).to(tl.float32), input_precision="ieee"
            )

            m_new = tl.maximum(m_i, tl.max(qk, 1))
            alpha = tl.exp2(m_i - m_new)
            p = tl.exp2(qk - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            # only sequences with interior blocks reach this loop, so skipping the
            # rescale when the running max is unchanged pays here
            if tl.max(m_new - m_i) > 0.0:
                acc = acc * alpha[:, None]
            acc += tl.dot(p, v_tile.to(tl.float32), input_precision="ieee")
            m_i = m_new

        # blocks that are fully inside the sequence but cross the causal diagonal:
        # only the causal test is needed, so no key-validity compare or select
        hi_valid = (hi // BLOCK_N) * BLOCK_N
        for start_n in range(n_full, hi_valid, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            if OFF64:
                k_off = offs_n.to(tl.int64) * stride_kt
                v_off = offs_n.to(tl.int64) * stride_vt
            else:
                k_off = offs_n * stride_kt
                v_off = offs_n * stride_vt
            k_ptrs = k_base + k_off[:, None] + offs_d[None, :] * stride_kd
            v_ptrs = v_base + v_off[:, None] + offs_d[None, :] * stride_vd
            if EXACT_D:
                k_tile = tl.load(k_ptrs)
                v_tile = tl.load(v_ptrs)
            else:
                k_tile = tl.load(k_ptrs, mask=d_mask[None, :], other=0.0)
                v_tile = tl.load(v_ptrs, mask=d_mask[None, :], other=0.0)

            qk = tl.dot(
                q, tl.trans(k_tile).to(tl.float32), input_precision="ieee"
            )
            qk = tl.where(
                offs_n[None, :] <= offs_m[:, None], qk, float("-inf")
            )

            m_new = tl.maximum(m_i, tl.max(qk, 1))
            alpha = tl.exp2(m_i - m_new)
            p = tl.exp2(qk - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]
            acc += tl.dot(p, v_tile.to(tl.float32), input_precision="ieee")
            m_i = m_new

        # final partial block: causal test and key-validity mask both required
        for start_n in range(hi_valid, hi, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_mask = offs_n < seq_len
            if OFF64:
                k_off = offs_n.to(tl.int64) * stride_kt
                v_off = offs_n.to(tl.int64) * stride_vt
            else:
                k_off = offs_n * stride_kt
                v_off = offs_n * stride_vt
            k_ptrs = k_base + k_off[:, None] + offs_d[None, :] * stride_kd
            v_ptrs = v_base + v_off[:, None] + offs_d[None, :] * stride_vd
            k_tile = tl.load(
                k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0
            )
            v_tile = tl.load(
                v_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0
            )

            qk = tl.dot(
                q, tl.trans(k_tile).to(tl.float32), input_precision="ieee"
            )

            if IS_CAUSAL:
                qk = tl.where(
                    offs_n[None, :] <= offs_m[:, None], qk, float("-inf")
                )
            qk = tl.where(n_mask[None, :], qk, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(qk, 1))
            alpha = tl.exp2(m_i - m_new)
            p = tl.exp2(qk - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]
            acc += tl.dot(p, v_tile.to(tl.float32), input_precision="ieee")
            m_i = m_new

        acc = acc / l_i[:, None]

        if OFF64:
            o_off = offs_m.to(tl.int64) * stride_ot
        else:
            o_off = offs_m * stride_ot
        o_ptrs = o_base + o_off[:, None] + offs_d[None, :] * stride_od
        if EXACT_D:
            tl.store(o_ptrs, acc, mask=m_mask[:, None])
        else:
            tl.store(o_ptrs, acc, mask=m_mask[:, None] & d_mask[None, :])


def run(q, k, v, b_start_loc, b_seq_len, max_input_len=None, is_causal=False):
    total_tokens, num_heads, head_dim = q.shape
    out = torch.empty(
        (total_tokens, num_heads, head_dim),
        dtype=torch.float32,
        device=q.device,
    )
    n_seq = b_start_loc.numel()
    if n_seq == 0 or total_tokens == 0:
        return out

    block_d = max(16, triton.next_power_of_2(head_dim))
    mil = int(max_input_len) if max_input_len else total_tokens
    if mil <= 256:
        seq_bucket = 0
    elif mil <= 1024:
        seq_bucket = 1
    else:
        seq_bucket = 2
    # the grid-stride loop keeps correctness for any grid size >= 1; the cap only
    # prevents a huge host bound from oversubscribing the grid with idle CTAs
    cap = min(triton.cdiv(mil, 32), triton.cdiv(total_tokens, 32))

    def grid(meta):
        gm = triton.cdiv(mil, meta["BLOCK_M"])
        return (max(1, min(gm, cap)), num_heads, n_seq)

    _context_attn_fwd[grid](
        q,
        k,
        v,
        out,
        b_start_loc,
        b_seq_len,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        1.4426950408889634 / (head_dim**0.5),
        seq_bucket,
        IS_CAUSAL=bool(is_causal),
        BLOCK_D=block_d,
        D=head_dim,
        EXACT_D=(block_d == head_dim),
        OFF64=(total_tokens * num_heads * head_dim >= 2**31),
    )
    return out


def context_attention(
    q, k, v, b_start_loc, b_seq_len, max_input_len, is_causal
):
    return run(q, k, v, b_start_loc, b_seq_len, max_input_len, is_causal)


__all__ = ["context_attention"]
