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

"""Benchmark for dsa/triton_sparse_mla_fwd."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark
from flaggems_sglang.utils.rng_utils import create_generator

_DIM = 576
_D_V = 512
_HEADS = 16
_NUM_PAGES = 8192
# (seq, topk) — 16 heads, 8192 pages. The baseline autotunes on
# (topk, heads, dim), so the first call on a new shape pauses to tune.
SHAPES = [(seq, topk) for seq in (1, 16, 128) for topk in (128, 512, 2048)]
MORE_SHAPES = [(4, 64), (64, 256), (32, 128)]


def _input_fn(shape, cur_dtype, device):
    seq, topk = shape
    g = create_generator(device, seed=0)

    def _q(width):
        return (
            0.5
            * torch.randn(
                seq,
                _HEADS,
                width,
                dtype=torch.float32,
                device=device,
                generator=g,
            )
        ).to(cur_dtype)

    kv = (
        0.5
        * torch.randn(
            _NUM_PAGES,
            1,
            _DIM,
            dtype=torch.float32,
            device=device,
            generator=g,
        )
    ).to(cur_dtype)
    idx = torch.randint(
        0,
        _NUM_PAGES,
        (seq, 1, topk),
        device=device,
        generator=g,
        dtype=torch.int32,
    )
    yield dict(
        q_nope=_q(_D_V),
        q_rope=_q(_DIM - _D_V),
        kv=kv,
        indices=idx,
        sm_scale=(_DIM**-0.5),
        d_v=_D_V,
    )


@pytest.mark.triton_sparse_mla_fwd
def test_perf_triton_sparse_mla_fwd():
    gems_op = flaggems_sglang.get_op("triton_sparse_mla_fwd")
    if gems_op is None:
        pytest.skip("dsa/triton_sparse_mla_fwd not implemented yet")
    bench = OpBenchmark(
        op_name="triton_sparse_mla_fwd",
        torch_op=get_reference("triton_sparse_mla_fwd"),
        input_fn=_input_fn,
        dtypes=[torch.float8_e4m3fn],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="seq, topk",
    )
    bench.set_gems(gems_op)
    bench.run()
