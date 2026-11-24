# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import flaggems_sglang

from .cases import BenchmarkCasePlan
from .op_benchmark import OpBenchmark


def _cases(shape, dtype):
    (batch,) = shape
    yield BenchmarkCasePlan(
        shape={"candidates": (batch, 4)},
        params={"batch": batch, "width": 4},
        builder_args=(batch,),
    )


def _inputs(plan, dtype, device):
    (batch,) = plan.builder_args
    slots = torch.arange(batch * 4, dtype=torch.int64, device=device).reshape(
        batch, 4
    )
    drafts = slots.to(dtype)
    target = torch.roll(drafts, -1, dims=1).contiguous()
    target[::2, 0] = -1
    children = torch.tensor([1, 2, 3, -1], device=device).repeat(batch, 1)
    siblings = torch.full_like(children, -1)
    prediction = torch.zeros(batch * 4, dtype=torch.int32, device=device)
    accepted = torch.full((batch, 4), -1, dtype=torch.int32, device=device)
    count = torch.empty(batch, dtype=torch.int32, device=device)
    return (
        prediction,
        accepted,
        count,
        drafts,
        slots,
        children,
        siblings,
        target,
    )


@pytest.mark.eagle_greedy_verify
@pytest.mark.skipif(flaggems_sglang.device != "musa", reason="MUSA kernel")
def test_perf_eagle_greedy_verify():
    native = pytest.importorskip("sglang.kernels.ops.speculative.spec_tree")

    # The pre-fusion framework invokes the native verifier and then allocates
    # the committed-token count with an out-of-place add.
    def native_committed_count(*args):
        _, _, correct = native.verify_tree_greedy_func(*args, topk=1)
        return correct + 1

    bench = OpBenchmark(
        op_name="eagle_greedy_verify",
        torch_op=native_committed_count,
        case_fn=_cases,
        build_inputs_fn=_inputs,
        dtypes=[torch.int64],
        timing_reset_args=(0, 1, 2),
        shapes=[(batch,) for batch in (1, 8, 16, 32, 64, 128)],
        shape_desc="batch",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
