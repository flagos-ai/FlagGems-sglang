# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tensor-free listing, exact replay and real mRoPE candidate timing."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference.mrotary_embedding import (
    WORKLOADS,
    baseline_key,
    build_inputs,
    case_parameters,
    describe_inputs,
)
from flaggems_sglang.reference_candidate import reference_mode_enabled
from flaggems_sglang.testing.reset_timing import reset_factory

from . import conftest
from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark, is_case_listing

SHAPES = [(i,) for i in range(len(WORKLOADS))]


def _baseline_key(**kwargs):
    return baseline_key(kwargs)


def _input_fn(shape, cur_dtype, device):
    yield (build_inputs(WORKLOADS[shape[0]], cur_dtype, device),)


def _case_fn(shape, cur_dtype):
    del cur_dtype
    case = WORKLOADS[shape[0]]
    yield BenchmarkCasePlan(
        shape=describe_inputs(case),
        params=dict(case_parameters(case), configuration=shape),
        builder_args=(shape,),
    )


@pytest.mark.mrotary_embedding
def test_perf_mrotary_embedding():
    bench = OpBenchmark(
        op_name="mrotary_embedding",
        torch_op=get_reference("mrotary_embedding"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        shape_desc="Qwen3.6 TP4 mRoPE configuration",
        baseline_key_fn=_baseline_key,
        input_reset_factory=reset_factory("mrotary_embedding"),
    )
    if not (
        is_case_listing()
        or reference_mode_enabled()
        or conftest.Config.reference_only
    ):
        try:
            bench.set_gems_from_module(flaggems_sglang)
        except AttributeError:
            pytest.skip(
                "mrotary_embedding: supply a candidate with --override"
            )
    bench.run()
