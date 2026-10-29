# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Case listing, exact replay and timing for Qwen3.6 gemma_fused_add_rmsnorm."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference_candidate import reference_mode_enabled
from flaggems_sglang.testing.qwen36_workloads import (
    baseline_key,
    build_inputs,
    describe,
    parameters,
    workloads,
)

from . import conftest
from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark, is_case_listing

OPERATOR = "gemma_fused_add_rmsnorm"
CORE = workloads(OPERATOR)
ALL = workloads(OPERATOR, comprehensive=True)
SHAPES = [(index,) for index in range(len(CORE))]
MORE_SHAPES = [(index,) for index in range(len(CORE), len(ALL))]


def _input_fn(shape, cur_dtype, device):
    yield (build_inputs(OPERATOR, ALL[shape[0]], cur_dtype, device),)


def _case_fn(shape, cur_dtype):
    del cur_dtype
    row = ALL[shape[0]]
    yield BenchmarkCasePlan(
        shape=describe(OPERATOR, row),
        params=dict(parameters(OPERATOR, row), configuration=shape),
        builder_args=(shape,),
    )


def _baseline_key(**kwargs):
    return baseline_key(OPERATOR, kwargs)


@pytest.mark.gemma_fused_add_rmsnorm
def test_perf_gemma_fused_add_rmsnorm():
    bench = OpBenchmark(
        op_name=OPERATOR,
        torch_op=get_reference(OPERATOR),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="explicit Qwen3.6 workload index",
        baseline_key_fn=_baseline_key,
    )
    if not (
        is_case_listing()
        or reference_mode_enabled()
        or conftest.Config.reference_only
    ):
        try:
            bench.set_gems_from_module(flaggems_sglang)
        except AttributeError:
            pytest.skip(f"{OPERATOR}: no public candidate on this backend")
    bench.run()
