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

"""Collection workload for the original standalone SGLang KKT kernel."""

import torch

OP_NAME = "chunk_scaled_dot_kkt"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

SHAPES = [
    (8, 16, 64, 8, 32, 128, 1),
    (32, 4, 64, 8, 32, 128, 1),
]
MORE_SHAPES = [
    (1, 1, 16, 2, 2, 32, 1),
    (2, 3, 16, 2, 4, 32, 0),
    (3, 2, 32, 4, 8, 64, 1),
]


def native():
    try:
        from sglang.kernels.ops.attention.fla.chunk_scaled_dot_kkt import (
            chunk_scaled_dot_kkt_fwd as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op


def binding_for_case(shape):
    return dict(
        zip(
            ("batch", "nchunks", "chunk_size", "hg", "h", "k_dim", "use_g"),
            shape,
        )
    )


def grid():
    return [binding_for_case(shape) for shape in (*SHAPES, *MORE_SHAPES)]


def build_case(shape, cur_dtype, device):
    batch, nchunks, chunk_size, hg, h, k_dim, use_g = shape
    g = torch.Generator(device=device).manual_seed(0)
    t = nchunks * chunk_size
    k = torch.randn(
        batch,
        t,
        hg,
        k_dim,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(cur_dtype)
    beta = torch.rand(
        batch, t, h, generator=g, device=device, dtype=torch.float32
    )
    g_cumsum = None
    if use_g:
        raw = (
            -torch.rand(
                batch, t, h, generator=g, device=device, dtype=torch.float32
            )
            * 0.1
        )
        g_cumsum = (
            raw.view(batch, nchunks, chunk_size, h)
            .cumsum(dim=2)
            .view(batch, t, h)
        )
    return k, beta, g_cumsum, {"chunk_size": chunk_size}


def build_inputs(binding, dtype, device):
    shape = (
        binding["batch"],
        binding["nchunks"],
        binding["chunk_size"],
        binding["hg"],
        binding["h"],
        binding["k_dim"],
        binding["use_g"],
    )
    *args, kwargs = build_case(shape, dtype, device)
    return tuple(args), kwargs


def key_shape(binding):
    return [
        binding["batch"],
        binding["nchunks"],
        binding["chunk_size"],
        binding["hg"],
        binding["h"],
        binding["k_dim"],
        binding["use_g"],
    ]


def config(binding, dtype):
    batch, nchunks, chunk_size, hg, h, k_dim, use_g = (
        binding["batch"],
        binding["nchunks"],
        binding["chunk_size"],
        binding["hg"],
        binding["h"],
        binding["k_dim"],
        binding["use_g"],
    )

    def tensor(shape, tensor_dtype=str(dtype)):
        return {"shape": shape, "dtype": tensor_dtype}

    return {
        "inputs": {
            "k": tensor([batch, nchunks * chunk_size, hg, k_dim]),
            "beta": tensor([batch, nchunks * chunk_size, h], "torch.float32"),
            "g_cumsum": (
                tensor([batch, nchunks * chunk_size, h], "torch.float32")
                if use_g
                else {"scalar": None}
            ),
            "chunk_size": {"scalar": chunk_size},
        },
        "outputs": {
            "out": tensor(
                [batch, nchunks * chunk_size, h, chunk_size], "torch.float32"
            )
        },
        "dims": dict(binding),
    }
