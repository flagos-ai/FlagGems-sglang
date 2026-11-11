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

"""Native SGLang collection workload for extend_attention."""

import torch

OP_NAME = "extend_attention"
DTYPES = [torch.bfloat16]
IS_INPLACE = True

MUTATED_INPUTS = (3,)

SHAPES = [
    (B, n_ctx, 32, 8, 128)
    for B, n_ctx in ((1, 2048), (8, 2048), (64, 512), (256, 256))
]
MORE_SHAPES = [(4, 256, 12, 4, 128), (4, 256, 12, 4, 80), (2, 128, 8, 8, 64)]


def native():
    """解析 sglang.kernels.ops.attention.extend_attention.extend_attention_fwd。"""
    try:
        from sglang.kernels.ops.attention.extend_attention import (
            extend_attention_fwd as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [
        {"B": B, "n_ctx": n_ctx, "H_Q": H_Q, "H_KV": H_KV, "D": D}
        for (B, n_ctx, H_Q, H_KV, D) in (*SHAPES, *MORE_SHAPES)
    ]


def build_case(shape, cur_dtype, device):
    B, n_ctx, H_Q, H_KV, D = shape
    g = torch.Generator(device=device).manual_seed(0)

    def rnd_int(lo, hi, n):
        return torch.randint(
            lo, hi, (n,), dtype=torch.int32, device=device, generator=g
        )

    b_seq_len_prefix = rnd_int(1, max(2, n_ctx // 2), B)
    b_seq_len_extend = rnd_int(1, max(2, n_ctx // 2), B)
    b_seq_len = b_seq_len_prefix + b_seq_len_extend

    b_start_loc = torch.zeros((B,), dtype=torch.int32, device=device)
    b_start_loc[1:] = torch.cumsum(b_seq_len[:-1], 0)
    b_start_loc_extend = torch.zeros((B,), dtype=torch.int32, device=device)
    b_start_loc_extend[1:] = torch.cumsum(b_seq_len_extend[:-1], 0)

    kv_indptr = torch.zeros((B + 1,), dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(b_seq_len_prefix, 0)
    kv_indices = torch.zeros(
        (int(b_seq_len_prefix.sum()),), dtype=torch.int32, device=device
    )
    for i in range(B):
        kv_indices[kv_indptr[i] : kv_indptr[i + 1]] = torch.arange(
            b_start_loc[i].item(),
            (b_start_loc[i] + b_seq_len_prefix[i]).item(),
            device=device,
        )

    total_token_num = int(b_seq_len.sum())
    extend_token_num = int(b_seq_len_extend.sum())
    k_buffer = torch.randn(
        total_token_num,
        H_KV,
        D,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(cur_dtype)
    v_buffer = torch.randn(
        total_token_num,
        H_KV,
        D,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(cur_dtype)

    k_extend = torch.empty(
        (extend_token_num, H_KV, D), dtype=cur_dtype, device=device
    )
    v_extend = torch.empty(
        (extend_token_num, H_KV, D), dtype=cur_dtype, device=device
    )
    q_extend = torch.empty(
        (extend_token_num, H_Q, D), dtype=cur_dtype, device=device
    )
    for i in range(B):
        eib = (b_start_loc[i] + b_seq_len_prefix[i]).item()
        eie = (b_start_loc[i] + b_seq_len[i]).item()
        es = b_start_loc_extend[i].item()
        ee = (b_start_loc_extend[i] + b_seq_len_extend[i]).item()
        k_extend[es:ee] = k_buffer[eib:eie]
        v_extend[es:ee] = v_buffer[eib:eie]
        q_extend[es:ee] = torch.randn(
            (ee - es, H_Q, D),
            generator=g,
            device=device,
            dtype=torch.float32,
        ).to(cur_dtype)

    qo_indptr = torch.zeros((B + 1,), dtype=torch.int32, device=device)
    qo_indptr[1:] = torch.cumsum(b_seq_len_extend, 0)
    max_len_extend = int(b_seq_len_extend.max())

    return (
        q_extend,
        k_extend,
        v_extend,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        max_len_extend,
    )


def build_inputs(binding, dtype, device):
    shape = (
        binding["B"],
        binding["n_ctx"],
        binding["H_Q"],
        binding["H_KV"],
        binding["D"],
    )
    q, k, v, kb, vb, qo, kv, ids, max_len = build_case(shape, dtype, device)
    return (
        q,
        k,
        v,
        torch.zeros_like(q),
        kb,
        vb,
        qo,
        kv,
        ids,
        None,
        True,
        None,
        max_len,
        q.shape[-1] ** -0.5,
        0.0,
    ), {}


def key_shape(binding):
    return "B%d_nctx%d_hq%d_hkv%d_d%d" % (
        binding["B"],
        binding["n_ctx"],
        binding["H_Q"],
        binding["H_KV"],
        binding["D"],
    )


def config(binding, dtype):
    B, n_ctx = binding["B"], binding["n_ctx"]
    H_Q, H_KV, D = binding["H_Q"], binding["H_KV"], binding["D"]
    dt = str(dtype)
    # 前缀/扩展长度按 seed=0 随机，故 total/extend token 数据相关，用符号描述。
    tt = "total_token_num"
    et = "extend_token_num"
    return {
        "inputs": {
            "q_extend": {"shape": [et, H_Q, D], "dtype": dt},
            "k_extend": {"shape": [et, H_KV, D], "dtype": dt},
            "v_extend": {"shape": [et, H_KV, D], "dtype": dt},
            "o_extend": {
                "shape": [et, H_Q, D],
                "dtype": dt,
                "note": "原地写回",
            },
            "k_buffer": {"shape": [tt, H_KV, D], "dtype": dt},
            "v_buffer": {"shape": [tt, H_KV, D], "dtype": dt},
            "qo_indptr": {"shape": [B + 1], "dtype": "torch.int32"},
            "kv_indptr": {"shape": [B + 1], "dtype": "torch.int32"},
            "kv_indices": {
                "shape": ["sum(prefix_len)"],
                "dtype": "torch.int32",
            },
            "custom_mask": {"scalar": None},
            "is_causal": {"scalar": True},
            "mask_indptr": {"scalar": None},
            "max_len_extend": {"note": "max(b_seq_len_extend)，数据相关"},
            "sm_scale": {"scalar": D**-0.5},
            "logit_cap": {"scalar": 0.0},
        },
        "outputs": {
            "o_extend": {"shape": [et, H_Q, D], "dtype": dt},
        },
        "dims": {
            "B": B,
            "n_ctx": n_ctx,
            "H_Q": H_Q,
            "H_KV": H_KV,
            "D": D,
            "note": "prefix/extend 长度 seed=0 随机；"
            "total_token_num/extend_token_num 数据相关",
        },
    }
