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


"""Paired SGLang DSA native FP32 pipeline and Hopper FlashTopK."""

from functools import partial

import pytest
import torch

import flaggems_sglang
from baseline import flash_topk as workload

from .attri_util import FLASH_TOPK_BENCH_SHAPES, FLASH_TOPK_MORE_SHAPES
from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark, is_case_listing


def _case(shape, dtype):
    requests, queries_per_request, context = shape
    rows = requests * queries_per_request
    yield BenchmarkCasePlan(
        shape={"query": (rows, 64, 128), "out": (rows, 2048)},
        params={
            "requests": requests,
            "queries_per_request": queries_per_request,
            "context": context,
            "seed": 0,
        },
        builder_args=(shape,),
    )


def _build(plan, dtype, device, *, prepare):
    (shape,) = plan.builder_args
    return workload.build_inputs(
        shape,
        dtype,
        device,
        flaggems_sglang.flash_topk_workspace_size,
        prepare,
    )


def _candidate(*args, native_metadata, **kwargs):
    return flaggems_sglang.flash_topk(*args, **kwargs)


def _unavailable(*args, **kwargs):
    raise RuntimeError("FlashTopK baseline needs SGLang DSA v2 and DeepGEMM")


@pytest.mark.flash_topk
def test_perf_flash_topk():
    if is_case_listing():
        origin, prepare = _unavailable, _unavailable
    else:
        if not torch.cuda.is_available():
            pytest.skip("FlashTopK benchmark requires CUDA")
        if not getattr(
            flaggems_sglang, "supports_flash_topk", lambda _: False
        )("cuda"):
            pytest.skip("FlashTopK benchmark requires SM90 with FlagTree TLE")
        native = workload.native()
        if native is None:
            pytest.skip(
                "FlashTopK benchmark requires native SGLang DSA v2 and DeepGEMM"
            )
        origin, prepare = native.run, native.prepare
    bench = OpBenchmark(
        op_name="flash_topk",
        torch_op=origin,
        case_fn=_case,
        build_inputs_fn=partial(_build, prepare=prepare),
        dtypes=workload.DTYPES,
        shapes=FLASH_TOPK_BENCH_SHAPES,
        more_shapes=FLASH_TOPK_MORE_SHAPES,
        shape_desc="requests, queries per request, visible context",
    )
    bench.set_gems(_candidate)
    bench.run()
