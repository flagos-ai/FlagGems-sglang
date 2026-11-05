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

"""Benchmark for lora/gate_up_lora_b."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference._lora_batch_utils import make_batch_info

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark, is_case_listing

# Shapes match kernel-comp-baseline/problems/lora/gate_up_lora_b. The first
# entry is the per-segment token count.
SHAPES = [
    ((64,) * 8, 4, 32, 4096, "none"),
    ((256,) * 4, 2, 64, 4096, "none"),
]
MORE_SHAPES = [
    ((5,), 1, 16, 64, "none"),
    ((3, 7, 0, 12), 2, 16, 128, "none"),
    ((9, 4), 2, 32, 256, "shuffled"),
]


def _input_fn(shape, cur_dtype, device):
    seg_lens, num_lora, r, output_dim, permutation = shape
    g = torch.Generator(device=device).manual_seed(0)
    s = sum(seg_lens)

    x = torch.randn(
        s, 2 * r, generator=g, device=device, dtype=torch.float32
    ).to(cur_dtype)
    weights = torch.randn(
        num_lora,
        2 * output_dim,
        r,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(cur_dtype)
    base_output = torch.randn(
        s, 2 * output_dim, generator=g, device=device, dtype=torch.float32
    ).to(cur_dtype)
    batch_info = make_batch_info(
        list(seg_lens),
        [i % num_lora for i in range(len(seg_lens))],
        lora_ranks=[r] * num_lora,
        scalings=[0.5 + 0.25 * i for i in range(num_lora)],
        permutation=permutation,
    )
    # ``batch_info`` is a dataclass, which unpack_to_args_kwargs would drop
    # from the positional args, so route it (and the tail) through kwargs.
    yield x, weights, dict(
        batch_info=batch_info,
        output_dim=output_dim,
        base_output=base_output,
    )


def _case_fn(shape, cur_dtype):
    (seg_lens, num_lora, r, output_dim, permutation) = shape
    yield BenchmarkCasePlan(
        shape={
            "x": (sum(seg_lens), 2 * r),
            "weights": (num_lora, 2 * output_dim, r),
            "base_output": (sum(seg_lens), 2 * output_dim),
            "batch_info.seg_indptr": (len(seg_lens) + 1,),
            "batch_info.seg_lens": (len(seg_lens),),
            "batch_info.weight_indices": (len(seg_lens),),
            "batch_info.lora_ranks": (num_lora,),
            "batch_info.scalings": (num_lora,),
            "batch_info.permutation": (
                (sum(seg_lens),) if permutation != "none" else None
            ),
        },
        params={
            "configuration": shape,
            "seed": 0,
            "permutation_rng": (
                "global_torch_rng" if permutation == "shuffled" else None
            ),
            "seg_lens": seg_lens,
            "num_lora": num_lora,
            "r": r,
            "output_dim": output_dim,
            "permutation": permutation,
        },
        builder_args=(shape,),
    )


@pytest.mark.gate_up_lora_b
def test_perf_gate_up_lora_b():
    # This op has a reference but no Triton implementation yet.
    gems_op = None
    if not is_case_listing():
        gems_op = flaggems_sglang.get_op("gate_up_lora_b")
        if gems_op is None:
            pytest.skip("lora/gate_up_lora_b not implemented yet")
    bench = OpBenchmark(
        op_name="gate_up_lora_b",
        torch_op=get_reference("gate_up_lora_b"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="seg_lens, num_lora, r, output_dim, permutation",
    )
    bench.set_gems(gems_op)
    bench.run()
