# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Independent CPU mathematics and mutable-input CUDA graph qualification."""

import importlib

import pytest
import torch

ATOL = 5e-4
RTOL = 2**-7


def _fixture(rows, seed):
    generator = torch.Generator().manual_seed(seed)
    query = torch.zeros((rows, 64, 576), dtype=torch.bfloat16)
    query[:, :4] = (
        torch.randn((rows, 4, 576), generator=generator).to(torch.bfloat16)
        * 0.125
    )
    cache = (
        torch.randn((2113, 1, 576), generator=generator).to(torch.bfloat16)
        * 0.125
    )
    indices = torch.randint(
        0,
        cache.shape[0],
        (rows, 1, 2048),
        generator=generator,
        dtype=torch.int32,
    )
    # Exercise invalid slots, the first and last valid physical slot, and
    # duplicate selected keys. Selection order is part of the operator input.
    indices[:, :, :5] = torch.tensor([-1, 0, 2112, 2113, 0])
    sink = torch.full((64,), -torch.inf)
    sink[:4] = torch.tensor([-torch.inf, 0.0, 2.0, torch.inf])
    lengths = torch.full((rows,), 2048, dtype=torch.int32)
    return dict(q=query, kv=cache, indices=indices, sink=sink, lengths=lengths)


def _reference(inputs):
    """Dispatch the independent library reference with FP64 CPU inputs."""
    from flaggems_sglang.reference import get_reference

    reference = get_reference("sparse_mla_active_split")
    inputs = {
        key: value.double() if value.is_floating_point() else value
        for key, value in inputs.items()
    }
    return reference(**inputs, sm_scale=576**-0.5, active_heads=4)


def _assert_math(actual, expected):
    for index, (value, reference) in enumerate(zip(actual, expected)):
        # Keep the archived operator gate; do not widen it for the migration.
        torch.testing.assert_close(
            value.cpu().double(),
            reference,
            atol=ATOL if index == 0 else 1e-5,
            rtol=RTOL if index == 0 else 1e-5,
            equal_nan=True,
        )


def _operator():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("this candidate is qualified only for SM90")
    import flaggems_sglang

    operator = flaggems_sglang.sparse_mla_active_split
    assert operator.__module__ == "flaggems_sglang.ops.sparse_mla_active_split"
    return operator


@pytest.mark.sparse_mla_active_split
@pytest.mark.parametrize("rows", [1, 4])
@pytest.mark.parametrize("length", [0, 1, 63, 64, 65, 2047, 2048])
@torch.no_grad()
def test_sparse_mla_active_split_math(rows, length):
    operator = _operator()
    inputs = _fixture(rows, seed=length + 1)
    inputs["lengths"].fill_(length)
    expected = _reference(inputs)
    device_inputs = {key: value.cuda() for key, value in inputs.items()}
    saved = {key: value.clone() for key, value in device_inputs.items()}
    outputs = operator(**device_inputs, sm_scale=576**-0.5, active_heads=4)
    _assert_math(outputs, expected)
    for _ in range(10):
        repeated = operator(
            **device_inputs, sm_scale=576**-0.5, active_heads=4
        )
        for value, first in zip(repeated, outputs):
            torch.testing.assert_close(
                value, first, atol=0, rtol=0, equal_nan=True
            )
    for key, value in device_inputs.items():
        assert torch.equal(value, saved[key]), key


@pytest.mark.sparse_mla_active_split
@pytest.mark.parametrize("rows", [1, 4])
@torch.no_grad()
def test_sparse_mla_active_split_graph_updates(rows):
    operator = _operator()
    states = [_fixture(rows, seed=seed) for seed in (19, 29)]
    states[0]["lengths"][:] = torch.tensor([0, 63, 64, 2048][:rows])
    states[1]["lengths"][:] = torch.tensor([65, 2047, 1, 0][:rows])
    states[1]["sink"][:4] = torch.tensor([1.0, -torch.inf, torch.inf, -2.0])
    expected = [_reference(state) for state in states]
    device_inputs = {key: value.cuda() for key, value in states[0].items()}
    module = importlib.import_module(operator.__module__)
    partials = module.workspace(device_inputs["q"], 4)

    # The caller retains every captured input and workspace owner throughout replay.
    def forward():
        return operator(
            **device_inputs,
            sm_scale=576**-0.5,
            active_heads=4,
            partials=partials,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            forward()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = forward()
    # Retain the original input/output owners for the entire graph lifetime.
    for state_index in (0, 1, 0):
        for key, value in states[state_index].items():
            device_inputs[key].copy_(value)
        saved = {key: value.clone() for key, value in device_inputs.items()}
        for _ in range(10):
            for output in captured:
                output.fill_(torch.nan)
            graph.replay()
            torch.cuda.synchronize()
            _assert_math(captured, expected[state_index])
        for key, value in device_inputs.items():
            assert torch.equal(value, saved[key]), key
