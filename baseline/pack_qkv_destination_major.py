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

"""pack_qkv_destination_major baseline（方案 B，sglang）。

native：复刻 problems/diffusion/pack_qkv_destination_major/baseline.py：
    sglang.kernels.ops.diffusion.triton.ulysses_qkv.pack_qkv_destination_major
        out = None if out is None else out.clone()
        pack_qkv_destination_major(q, k, v, world_size, out)
    BENCH_CASES 的 pass_out 缺省 False -> out=None，kernel 自行分配返回。
    out=None 故不原地写回 -> IS_INPLACE=False。

输入构造复刻 cases.py 的 BENCH_CASES：
    q/k/v = randn(rows, global_heads, head_size).bf16（contiguous）; out=None;
    BENCH 固定 global_heads=32, head_size=128, world_size=8，遍历 rows∈{64,1024,4096,16384}。
dtype 取 cases.py 的 [bfloat16]。
"""

import torch

OP_NAME = "pack_qkv_destination_major"
DTYPES = [torch.bfloat16]
IS_INPLACE = False  # BENCH 中 out=None，kernel 自行分配

_SHAPES = [(r, 32, 128, 8) for r in (64, 1024, 4096, 16384)]


def native():
    """解析 pack_qkv_destination_major。

    sglang 0.5.21 把该符号挪到了 sglang.kernels.ops.diffusion.layout.ulysses_qkv_triton，故先按 spec 路径再回退已安装路径。
    """
    try:
        from sglang.kernels.ops.diffusion.triton.ulysses_qkv import (
            pack_qkv_destination_major as op,
        )
    except Exception:
        try:
            from sglang.kernels.ops.diffusion.layout.ulysses_qkv_triton import (
                pack_qkv_destination_major as op,
            )
        except Exception:
            return None
    return op if callable(op) else None


def grid():
    return [
        {"rows": r, "global_heads": gh, "head_size": hs, "world_size": ws}
        for (r, gh, hs, ws) in _SHAPES
    ]


def build_inputs(binding, dtype, device):
    rows, global_heads = binding["rows"], binding["global_heads"]
    head_size, world_size = binding["head_size"], binding["world_size"]
    g = torch.Generator(device=device).manual_seed(0)

    def _t():
        return torch.randn(
            rows, global_heads, head_size, dtype=torch.float32, device=device, generator=g
        ).to(dtype)

    # BENCH pass_out 缺省 False -> out=None（baseline 对 None 不 clone）。
    return (_t().contiguous(), _t().contiguous(), _t().contiguous(), world_size, None), {}


def key_shape(binding):
    r, gh = binding["rows"], binding["global_heads"]
    return f"r{r}_gh{gh}_hs{binding['head_size']}_ws{binding['world_size']}"


def config(binding, dtype):
    rows, global_heads = binding["rows"], binding["global_heads"]
    head_size, world_size = binding["head_size"], binding["world_size"]
    dt = str(dtype)
    return {
        "inputs": {
            "q": {"shape": [rows, global_heads, head_size], "dtype": dt},
            "k": {"shape": [rows, global_heads, head_size], "dtype": dt},
            "v": {"shape": [rows, global_heads, head_size], "dtype": dt},
            "world_size": {"scalar": world_size},
            "out": {"scalar": None},
        },
        "outputs": {
            "out": {"shape": [world_size, rows, global_heads // world_size, 3 * head_size],
                    "dtype": dt},
        },
        "dims": {
            "rows": rows, "global_heads": global_heads,
            "head_size": head_size, "world_size": world_size,
        },
    }
