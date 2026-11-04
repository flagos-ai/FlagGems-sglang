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

"""fused_moe_gemm baseline（方案 B，sglang）。

native（spec「SGLang baseline」字段）：
    sglang.kernels.ops.moe.fused_moe_triton_kernels.invoke_fused_moe_kernel
    调用形态复刻 problems/moe/fused_moe_gemm/baseline.py：先用
    moe_align_block_size(topk_ids, BLOCK_SIZE_M=32, E) 预处理出
    sorted_token_ids / expert_ids / num_tokens_post_padded，分配零初始化的
    C = zeros(T*top_k, N)，再按固定 config / compute_type 调 kernel。
    kernel 把结果原地写入 C -> IS_INPLACE=True（采样前用 build_inputs 重建实参，
    避免 warmup 的原地累积污染）。

    注：源 moe_align_block_size 依赖已编译的 sgl_kernel 扩展，与本机定制 torch
    ABI 不兼容；这里用纯 torch 等价实现 _moe_align_block_size_torch 产出同样的
    对齐元数据，被采样的 invoke_fused_moe_kernel 本身不变。

输入构造复刻 problems/moe/fused_moe_gemm/cases.py 的 BENCH_CASES：
    A=randn(T,K).to(dtype); B=randn(E,N,K).to(dtype);
    topk_ids=randint(0,E,(T,top_k)) int32; topk_weights=rand(T,top_k) fp32
    其中 E=8, N=4096, K=4096, top_k=2。
dtype 取 benchmark/test_fused_moe_gemm.py 的 [bfloat16]。
"""

import torch

OP_NAME = "fused_moe_gemm"
DTYPES = [torch.bfloat16]
IS_INPLACE = True  # kernel 原地写入 C

_BLOCK_SIZE_M = 32
_SHAPES = [(t, 8, 4096, 4096, 2) for t in (1, 8, 64, 512, 4096)]


def native():
    """解析 invoke_fused_moe_kernel；解析不到返回 None。"""
    try:
        from sglang.kernels.ops.moe.fused_moe_triton_kernels import (
            invoke_fused_moe_kernel as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [
        {"T": t, "E": e, "N": n, "K": k, "top_k": tk}
        for (t, e, n, k, tk) in _SHAPES
    ]


def _moe_align_block_size_torch(topk_ids, block_size, num_experts, device):
    """纯 torch 复刻 sglang moe_align_block_size 的对齐语义（无 sgl_kernel 依赖）。

    sglang 的 moe_align_block_size 依赖已编译的 sgl_kernel 扩展，而本机的定制
    torch 与其预编译 wheel ABI 不兼容。该函数只负责为 invoke_fused_moe_kernel
    预处理出 sorted_token_ids / expert_ids / num_tokens_post_padded 这组元数据
    （被采样的 kernel 本身不变），语义与源实现一致：按 expert 分桶、每桶 padding
    到 block_size 的整数倍，padding 槽填哨兵值 topk_ids.numel()。
    """
    flat = topk_ids.flatten()
    numel = flat.numel()
    counts = torch.bincount(flat.to(torch.int64), minlength=num_experts)
    padded = ((counts + block_size - 1) // block_size) * block_size
    num_tokens_post_padded = torch.zeros(1, dtype=torch.int32, device=device)
    num_tokens_post_padded[0] = int(padded.sum().item())

    if numel < num_experts + 1:
        max_tokens = numel * block_size
    else:
        max_tokens = numel + (num_experts + 1) * (block_size - 1)

    sentinel = numel
    sorted_token_ids = torch.full(
        (max_tokens,), sentinel, dtype=torch.int32, device=device
    )
    num_blocks_total = int((padded // block_size).sum().item())
    expert_ids = torch.zeros(max(num_blocks_total, 1), dtype=torch.int32, device=device)

    # 按 expert 升序把真实 token 填进各自桶（桶起点按 padding 后的偏移），
    # 并给每个 block 标注其 expert id。
    order = torch.argsort(flat, stable=True)
    offsets = torch.zeros(num_experts + 1, dtype=torch.int64, device=device)
    offsets[1:] = torch.cumsum(padded, dim=0)
    block_cursor = 0
    pos = 0
    sorted_cpu = sorted_token_ids
    for e in range(num_experts):
        c = int(counts[e].item())
        start = int(offsets[e].item())
        if c > 0:
            sorted_cpu[start:start + c] = order[pos:pos + c].to(torch.int32)
            pos += c
        nblk = int((padded[e] // block_size).item())
        if nblk > 0:
            expert_ids[block_cursor:block_cursor + nblk] = e
            block_cursor += nblk
    return sorted_token_ids, expert_ids, num_tokens_post_padded


def build_inputs(binding, dtype, device):
    import triton.language as tl

    T, E, N, K, top_k = (
        binding["T"], binding["E"], binding["N"], binding["K"], binding["top_k"],
    )
    g = torch.Generator(device=device).manual_seed(0)
    A = torch.randn(T, K, generator=g, device=device, dtype=torch.float32).to(dtype)
    B = torch.randn(E, N, K, generator=g, device=device, dtype=torch.float32).to(dtype)
    topk_ids = torch.randint(
        0, E, (T, top_k), dtype=torch.int32, device=device, generator=g
    )
    topk_weights = torch.rand(T, top_k, device=device, generator=g, dtype=torch.float32)

    sorted_token_ids, expert_ids, num_tokens_post_padded = _moe_align_block_size_torch(
        topk_ids, _BLOCK_SIZE_M, E, device
    )
    C = torch.zeros(T * top_k, N, dtype=A.dtype, device=A.device)
    compute_type = tl.bfloat16 if A.dtype == torch.bfloat16 else tl.float16
    config = dict(
        BLOCK_SIZE_M=_BLOCK_SIZE_M,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=32,
        GROUP_SIZE_M=8,
        num_warps=4,
        num_stages=3,
    )
    args = (
        A, B, None, C, None, None, None,
        topk_weights, topk_ids,
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        True, top_k, config, compute_type,
        False, False, False, False, False,
    )
    return args, {}


def key_shape(binding):
    return "T{T}_E{E}_N{N}_K{K}_tk{top_k}".format(**binding)


def config(binding, dtype):
    T, E, N, K, top_k = (
        binding["T"], binding["E"], binding["N"], binding["K"], binding["top_k"],
    )
    dt = str(dtype)
    return {
        "inputs": {
            "A": {"shape": [T, K], "dtype": dt},
            "B": {"shape": [E, N, K], "dtype": dt},
            "C": {"shape": [T * top_k, N], "dtype": dt, "note": "原地写回"},
            "topk_weights": {"shape": [T, top_k], "dtype": "torch.float32"},
            "topk_ids": {"shape": [T, top_k], "dtype": "torch.int32"},
        },
        "outputs": {
            "C": {"shape": [T * top_k, N], "dtype": dt},
        },
        "dims": {"T": T, "E": E, "N": N, "K": K, "top_k": top_k},
    }
