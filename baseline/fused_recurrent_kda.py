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

"""fused_recurrent_kda baseline（方案 B，sglang）。

native：sglang.kernels.ops.attention.fla.kda.fused_recurrent_kda_fwd
    调用形态复刻 problems/fla/fused_recurrent_kda/baseline.py：
        state = initial_state.clone()
        fused_recurrent_kda_fwd(
            q, k, v, g, beta, scale, state,
            inplace_final_state=True, use_qk_l2norm_in_kernel=...)
    inplace_final_state=True 把 final state 原地写回传入的 state（故 baseline
    先 clone initial_state）。IS_INPLACE=True（第 7 个实参被原地写回）。

输入构造复刻 problems/.../cases.py 的 BENCH_CASES：
    q/k = randn(b, t, h, kdim).bf16；v = randn(b, t, h, vdim).bf16
    g = -0.05*rand(b, t, h, kdim) fp32；beta = rand(b, t, h).bf16（headwise=False）
    scale = kdim**-0.5；initial_state = zeros(b, h, vdim, kdim) fp32
    l2norm=False；BENCH: (b,64,8,128,128) for b in (1,4,16,64)。seed=0。
dtype 取 cases.py 的 [bfloat16]。
"""

import torch

OP_NAME = "fused_recurrent_kda"
DTYPES = [torch.bfloat16]
IS_INPLACE = True  # inplace_final_state=True，state 实参被原地写回

# (b, t, h, kdim, vdim)
_CASES = [(b, 64, 8, 128, 128) for b in (1, 4, 16, 64)]


def native():
    """解析 sglang.kernels.ops.attention.fla.kda.fused_recurrent_kda_fwd。"""
    try:
        from sglang.kernels.ops.attention.fla.kda import fused_recurrent_kda_fwd as op
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [
        {"b": b, "t": t, "h": h, "kdim": kd, "vdim": vd}
        for (b, t, h, kd, vd) in _CASES
    ]


def build_inputs(binding, dtype, device):
    b, t, h, kdim, vdim = (
        binding["b"], binding["t"], binding["h"], binding["kdim"], binding["vdim"],
    )
    g = torch.Generator(device=device).manual_seed(0)

    def _t(*shape):
        return torch.randn(*shape, dtype=torch.float32, device=device, generator=g).to(dtype)

    q = _t(b, t, h, kdim).contiguous()
    k = _t(b, t, h, kdim).contiguous()
    v = _t(b, t, h, vdim).contiguous()
    gate = (-0.05 * torch.rand(b, t, h, kdim, dtype=torch.float32, device=device, generator=g)).contiguous()
    beta = torch.rand(b, t, h, dtype=torch.float32, device=device, generator=g).to(dtype).contiguous()
    scale = kdim ** -0.5
    initial_state = torch.zeros(b, h, vdim, kdim, dtype=torch.float32, device=device)
    state = initial_state.clone()
    return (q, k, v, gate, beta, scale, state), {
        "inplace_final_state": True,
        "use_qk_l2norm_in_kernel": False,
    }


def key_shape(binding):
    return "b%d_t%d_h%d_k%d_v%d" % (
        binding["b"], binding["t"], binding["h"], binding["kdim"], binding["vdim"],
    )


def config(binding, dtype):
    b, t, h, kdim, vdim = (
        binding["b"], binding["t"], binding["h"], binding["kdim"], binding["vdim"],
    )
    dt = str(dtype)
    return {
        "inputs": {
            "q": {"shape": [b, t, h, kdim], "dtype": dt},
            "k": {"shape": [b, t, h, kdim], "dtype": dt},
            "v": {"shape": [b, t, h, vdim], "dtype": dt},
            "g": {"shape": [b, t, h, kdim], "dtype": "torch.float32"},
            "beta": {"shape": [b, t, h], "dtype": dt},
            "scale": {"scalar": kdim ** -0.5},
            "initial_state": {"shape": [b, h, vdim, kdim], "dtype": "torch.float32",
                              "note": "原地写回（inplace_final_state=True）"},
            "inplace_final_state": {"scalar": True},
            "use_qk_l2norm_in_kernel": {"scalar": False},
        },
        "outputs": {
            "o": {"shape": [b, t, h, vdim], "dtype": dt},
            "final_state": {"shape": [b, h, vdim, kdim], "dtype": "torch.float32"},
        },
        "dims": {"b": b, "t": t, "h": h, "kdim": kdim, "vdim": vdim},
    }
