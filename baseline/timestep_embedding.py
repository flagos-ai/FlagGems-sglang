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

"""timestep_embedding baseline（方案 B，sglang）。

native：复刻 problems/diffusion/timestep_embedding/baseline.py：
    sglang.kernels.ops.diffusion.timestep_embedding.timestep_embedding
        timestep_embedding(t, dim, flip_sin_to_cos, downscale_freq_shift,
                           scale, max_period)
    返回新张量 -> IS_INPLACE=False。

输入构造复刻 cases.py 的 BENCH_CASES：
    t = rand(batch) * 1000（fp32）; 标量 flip=False/shift=0.0/scale=1.0/
    max_period=10000（BENCH 缺省）。BENCH 遍历 batch∈{1,8,64,512}, dim∈{256,1024}。
t 为 fp32，非 bf16；DTYPES 仍记 [bfloat16] 以统一遍历键（输入 dtype 见 config）。
"""

import torch

OP_NAME = "timestep_embedding"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_FLIP = False
_SHIFT = 0.0
_SCALE = 1.0
_MAX_PERIOD = 10000
_SHAPES = [(b, d) for b in (1, 8, 64, 512) for d in (256, 1024)]


def native():
    """解析 timestep_embedding。

    sglang 0.5.21 把该符号挪到了 sglang.kernels.ops.diffusion.modulate.timestep_embedding_jit，故先按 spec 路径再回退已安装路径。
    """
    try:
        from sglang.kernels.ops.diffusion.timestep_embedding import (
            timestep_embedding as op,
        )
    except Exception:
        try:
            from sglang.kernels.ops.diffusion.modulate.timestep_embedding_jit import (
                timestep_embedding as op,
            )
        except Exception:
            return None
    return op if callable(op) else None


def grid():
    return [{"batch": b, "dim": d} for (b, d) in _SHAPES]


def build_inputs(binding, dtype, device):
    batch, dim = binding["batch"], binding["dim"]
    g = torch.Generator(device=device).manual_seed(9)
    t = torch.rand(batch, dtype=torch.float32, device=device, generator=g) * 1000
    return (t, dim, _FLIP, _SHIFT, _SCALE, _MAX_PERIOD), {}


def key_shape(binding):
    return f"b{binding['batch']}_dim{binding['dim']}"


def config(binding, dtype):
    batch, dim = binding["batch"], binding["dim"]
    return {
        "inputs": {
            "t": {"shape": [batch], "dtype": "torch.float32"},
            "dim": {"scalar": dim},
            "flip_sin_to_cos": {"scalar": _FLIP},
            "downscale_freq_shift": {"scalar": _SHIFT},
            "scale": {"scalar": _SCALE},
            "max_period": {"scalar": _MAX_PERIOD},
        },
        "outputs": {
            "emb": {"shape": [batch, dim], "dtype": "torch.float32"},
        },
        "dims": {"batch": batch, "dim": dim},
    }
