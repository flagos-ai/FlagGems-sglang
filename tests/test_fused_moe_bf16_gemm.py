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

import pytest
import torch

import flaggems_sglang as gems
from benchmark.musa_cases import expert_inputs
from flaggems_sglang.reference import get_reference

pytestmark = [
    pytest.mark.fused_moe_bf16_gemm,
    pytest.mark.skipif(
        gems.vendor_name != "mthreads",
        reason="MUSA-specific operator; see plugin PR #125",
    ),
]


def make_case(tokens, fuse_swiglu=False, weighted=False):
    device = gems.device
    routes, experts, hidden, width, block = 8, 2, 64, 128, 16
    a = torch.randn(
        (tokens if not weighted else tokens * routes, hidden),
        device=device,
        dtype=torch.bfloat16,
    )
    b = torch.randn((experts, width, hidden), device=device, dtype=a.dtype)
    weights = torch.rand((tokens, routes), device=device, dtype=torch.float32)
    # Alternate experts, then pad each expert independently. Every valid
    # route appears once; sentinel rows deliberately contain invalid IDs.
    groups = [list(range(e, tokens * routes, experts)) for e in range(experts)]
    ids, expert_ids = [], []
    for expert, group in enumerate(groups):
        padded = group + [tokens * routes] * ((-len(group)) % block)
        ids.extend(padded)
        expert_ids.extend([expert] * (len(padded) // block))
    indices = torch.tensor(ids, device=device, dtype=torch.int32)
    eids = torch.tensor(expert_ids, device=device, dtype=torch.int32)
    count = torch.tensor([len(ids)], device=device, dtype=torch.int32)
    out = torch.empty(
        (tokens, routes, width // 2 if fuse_swiglu else width),
        device=device,
        dtype=a.dtype,
    )
    config = dict(
        BLOCK_SIZE_M=block,
        BLOCK_SIZE_N=32,
        BLOCK_SIZE_K=32,
        GROUP_SIZE_M=1,
        num_warps=4,
        num_stages=1,
    )
    return (a, b, out, weights, indices, eids, count, config), dict(
        top_k=1 if weighted else routes,
        mul_routed_weight=weighted,
        fuse_swiglu=fuse_swiglu,
    )


@pytest.mark.parametrize("tokens", [1, 4, 7, 64])
@pytest.mark.parametrize(
    "fused,weighted", [(False, False), (True, False), (False, True)]
)
@pytest.mark.fused_moe_bf16_gemm
def test_expert_gemm_public_path(tokens, fused, weighted):
    args, kwargs = make_case(tokens, fused, weighted)
    expected = get_reference("fused_moe_bf16_gemm")(*args, **kwargs)
    assert gems.fused_moe_bf16_gemm(*args, **kwargs) is args[2]
    torch.testing.assert_close(args[2], expected, atol=1.5e-2, rtol=1.5e-2)


@pytest.mark.fused_moe_bf16_gemm
def test_expert_gemm_refreshed_graph():
    args, kwargs = make_case(4, True)
    gems.fused_moe_bf16_gemm(*args, **kwargs)
    torch.musa.synchronize()
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph):
        gems.fused_moe_bf16_gemm(*args, **kwargs)
    for phase in (0, 1, 0):
        args[0].fill_(phase + 0.5)
        expected = get_reference("fused_moe_bf16_gemm")(*args, **kwargs)
        args[2].fill_(float("nan"))
        graph.replay()
        torch.musa.synchronize()
        torch.testing.assert_close(args[2], expected, atol=1.5e-2, rtol=1.5e-2)


@pytest.mark.fused_moe_bf16_gemm
@pytest.mark.parametrize(
    "shape,changed_key,expected_value",
    [
        ((4, 0, 0), "BLOCK_SIZE_N", 64),
        ((2048, 1, 0), "BLOCK_SIZE_K", 32),
        ((8192, 1, 0), "BLOCK_SIZE_K", 32),
        ((16384, 1, 0), "BLOCK_SIZE_K", 32),
        ((8192, 0, 1), "num_warps", 16),
        ((16384, 0, 1), "num_warps", 16),
    ],
)
def test_production_profiles_reach_real_kernel(
    monkeypatch, shape, changed_key, expected_value
):
    import importlib

    module = importlib.import_module(gems.fused_moe_bf16_gemm.__module__)
    original_kernel, calls = module.fused_moe_kernel, []

    class ObservedKernel:
        def __getitem__(self, grid):
            launch = original_kernel[grid]

            def observed(*args, **kwargs):
                calls.append(dict(kwargs))
                return launch(*args, **kwargs)

            return observed

    monkeypatch.setattr(module, "fused_moe_kernel", ObservedKernel())
    values = next(expert_inputs(shape, torch.bfloat16, gems.device))
    args, kwargs = values[:-1], values[-1]
    # The oracle gets the independently specified legacy launch, rather than
    # reading the candidate's tuning YAML or its selected configuration.
    reference_kwargs = dict(
        kwargs,
        config=dict(kwargs["config"], **{changed_key: expected_value}),
        tuning_profile=None,
    )
    expected = get_reference("fused_moe_bf16_gemm")(*args, **reference_kwargs)
    gems.fused_moe_bf16_gemm(*args, **kwargs)
    assert len(calls) == 1 and calls[0][changed_key] == expected_value
    torch.testing.assert_close(args[2], expected, atol=1.5e-2, rtol=1.5e-2)


@pytest.mark.fused_moe_bf16_gemm
def test_routing_capacity_need_not_be_tile_aligned():
    args, kwargs = make_case(1)
    args = list(args)
    # Unused capacity is not the device-side valid padded count. Poison it
    # with invalid routing/expert IDs to prove no inactive block is consumed.
    args[4] = torch.cat((args[4], args[4].new_tensor([-1])))
    args[5] = torch.cat((args[5], args[5].new_tensor([-1])))
    expected = get_reference("fused_moe_bf16_gemm")(*args, **kwargs)
    args[2].fill_(float("nan"))
    gems.fused_moe_bf16_gemm(*args, **kwargs)
    torch.testing.assert_close(args[2], expected, atol=1.5e-2, rtol=1.5e-2)


@pytest.mark.fused_moe_bf16_gemm
@pytest.mark.parametrize(
    "case", ["dtype", "indices", "stride", "alias", "output_shape"]
)
def test_invalid_buffers_are_rejected_before_launch(monkeypatch, case):
    import importlib

    args, kwargs = make_case(4)
    args = list(args)
    module = importlib.import_module(gems.fused_moe_bf16_gemm.__module__)

    class ForbiddenKernel:
        def __getitem__(self, grid):
            raise AssertionError("invalid inputs reached launch")

    monkeypatch.setattr(module, "fused_moe_kernel", ForbiddenKernel())
    if case == "dtype":
        args[0] = args[0].float()
    elif case == "indices":
        args[4] = args[4].long()
    elif case == "stride":
        args[0] = args[0].T.contiguous().T
    elif case == "alias":
        storage = torch.empty(
            args[2].numel(), dtype=args[0].dtype, device=gems.device
        )
        args[0] = storage[: args[0].numel()].view_as(args[0])
        args[2] = storage.view_as(args[2])
    else:
        args[2] = args[2][..., :-1].contiguous()
    with pytest.raises(ValueError):
        gems.fused_moe_bf16_gemm(*args, **kwargs)


@pytest.mark.fused_moe_bf16_gemm
def test_empty_routing_does_not_launch(monkeypatch):
    import importlib

    args, kwargs = make_case(0)
    module = importlib.import_module(gems.fused_moe_bf16_gemm.__module__)
    monkeypatch.setattr(module, "fused_moe_kernel", None)
    assert gems.fused_moe_bf16_gemm(*args, **kwargs) is args[2]
