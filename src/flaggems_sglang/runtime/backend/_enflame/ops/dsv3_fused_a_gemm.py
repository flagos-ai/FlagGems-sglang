# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

# 2026-11-07 20:22 工具链回归（gcu400 / triton 3.6 + enflame.git72408c41）使本 op 的
# Triton kernel（2-D tile load/store + tl.dot + tl.trans）全部无法新鲜编译：
#   - rank-2 tile 访存（masked/无 mask/转置形态）→ memref_ext.slice_(pad_)start 8 vs 7
#   - tl.dot（ptr/bptr/纯值操作数混合、ieee/tf32、fp16/fp32）→ gcu.matmul 8 vs 5
#   - join 树 + tl.trans / 3-D 广播外积 → slice_start / 静默错值
#   - 纯 rank-1 elementwise GEMV（qk_head/dcp 同款安全原语）→ 精确但 80-766µs
#     （每 load/sum op ~1.3µs 固定成本 × N×(1+m) op 串行链），全部劣于 ATen。
# 详见 compiler_issues/sglang_dsv3_fused_a_gemm.md（触发矩阵 + 复现脚本索引）。
# 绕过方案：dispatch 到 ATen bf16 GEMM（topsprof 实证单次调用恰 1 个 kernel：
# op::matmul_bias_kernel_mle128_kle36864_gcu400，m≤128 小 M 专用路径）。
# bench core (m,7168,2112) bf16：31.5-32.1µs（旧 triton 基线，现已不可编译）→ 27.2µs
# 全 m 档平坦，精度对 fp32 参考逐 case 0 mismatch（bf16 容差内）。


def dsv3_fused_a_gemm(mat_a, mat_b):
    if mat_a.dtype == mat_b.dtype:
        return torch.matmul(mat_a, mat_b)
    return (mat_a.float() @ mat_b.float()).to(mat_a.dtype)


__all__ = ["dsv3_fused_a_gemm"]
