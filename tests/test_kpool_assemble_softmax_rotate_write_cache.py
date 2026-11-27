# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Check KPool assembly, quantization and in-place cache writes."""

from functools import partial
from types import SimpleNamespace

import pytest
import torch

from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference_candidate import (
    get_candidate,
    reference_mode_enabled,
)
from flaggems_sglang.runtime import torch_device_fn

from . import conftest as cfg

OP = "kpool_assemble_softmax_rotate_write_cache"
pytestmark = pytest.mark.kpool_assemble_softmax_rotate_write_cache
ROW_INDICES = ("req_pool_idx", "chunk_src_start", "tail_logical_base", "loc")


def _case(
    rows,
    size=4,
    source="mixed",
    masked=False,
    rounded=True,
    page_size=16,
    *,
    dtype=torch.bfloat16,
    tail_size=None,
):
    device = cfg.device
    tail_size = size if tail_size is None else tail_size
    generator = torch.Generator(device=device).manual_seed(9)

    def rand(shape, dtype):
        return torch.randn(shape, device=device, generator=generator).to(dtype)

    row = torch.arange(rows, device=device)
    requests = max(1, min(rows, 8))
    # All-tail inputs deliberately provide only one dummy chunk row.
    chunk_rows = 1 if source == "tail" else max(1, rows * size)
    count = (
        (row + 1) % (size + 1)
        if source == "mixed"
        else torch.full_like(row, size if source == "tail" else 0)
    )
    pages = (rows + 2 * page_size - 2) // page_size + 2
    inputs = dict(
        pool=SimpleNamespace(index_kpool=size, slots_per_page=page_size),
        buf=torch.full(
            (pages, page_size * 132), 0xA5, dtype=torch.uint8, device=device
        ),
        chunk_k=rand((chunk_rows, 128), torch.bfloat16),
        chunk_score=rand((chunk_rows, 128), dtype),
        tail_k=rand((requests, tail_size, 128), torch.bfloat16),
        tail_score=rand((requests, tail_size, 128), dtype),
        ape=rand((size, 128), torch.float32),
        req_pool_idx=requests - 1 - row % requests,
        n_from_tail=count.int(),
        chunk_src_start=row * size,
        tail_logical_base=(252 + row * size).int(),
        # Reverse destinations across a page boundary and leave guard slots.
        loc=(row + page_size - 1).flip(0),
        write_mask=(row % 3 != 1) if masked else None,
        round_scale=rounded,
    )
    return _pack_descriptors(inputs)


def _copy(inputs):
    device = "cpu" if cfg.TO_CPU else cfg.device
    return {
        k: v.to(device).clone() if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }


def _cache_rows(buf, page_size):
    raw = buf.view(torch.uint8)
    split = page_size * 128
    return raw[:, :split].reshape(-1, 128), raw[:, split:].reshape(-1, 4)


def _check(inputs, before, actual, expected):
    page_size = inputs["pool"].slots_per_page
    got, want, original = [
        _cache_rows(buf.cpu(), page_size) for buf in (actual, expected, before)
    ]
    loc = inputs["loc"]
    if inputs["write_mask"] is not None:
        loc = loc[inputs["write_mask"]]
    loc = loc.cpu().long()
    untouched = torch.ones(len(got[0]), dtype=torch.bool)
    untouched[loc] = False
    for field, saved in zip(got, original):
        torch.testing.assert_close(
            field[untouched], saved[untouched], rtol=0, atol=0
        )
    if not loc.numel():
        return
    keys, ref_keys = [
        fields[0][loc].view(torch.int8) for fields in (got, want)
    ]
    scales, ref_scales = [
        fields[1][loc].view(torch.float32) for fields in (got, want)
    ]
    assert (
        (keys >= -127).all()
        and (scales > 0).all()
        and torch.isfinite(scales).all()
    )
    torch.testing.assert_close(scales, ref_scales, rtol=0.01, atol=1e-10)
    # BF16 rounding before quantization may change an INT8 code by one.
    assert ((keys.int() - ref_keys.int()).abs() <= 1).all()
    x, y = keys * scales, ref_keys * ref_scales
    error = (x - y).norm(dim=1) / y.norm(dim=1).clamp_min(1e-8)
    assert (error < 0.04).all(), error.max().item()


def _check_workload(inputs):
    before = _copy(inputs)
    reference = get_reference(OP)
    expected = reference(**before)
    try:
        candidate = get_candidate(OP, reference=reference)
    except (ImportError, ModuleNotFoundError):
        pytest.skip(f"{OP} ops module not found")
    try:
        actual = candidate(**inputs)
    except NotImplementedError:
        pytest.skip(f"{OP} not yet implemented")
    if not reference_mode_enabled():
        assert actual is None
    result = inputs["buf"] if actual is None else actual
    _check(inputs, before["buf"], result, expected)
    for name, value in before.items():
        if isinstance(value, torch.Tensor) and name != "buf":
            torch.testing.assert_close(
                inputs[name].to(value.device), value, rtol=0, atol=0
            )
    return result


def _numerical_edge_case(
    zero_keys, size=4, masked=False, tail_dtype=torch.bfloat16
):
    inputs = _case(65, size, masked=masked, rounded=not zero_keys)
    if zero_keys:
        inputs["chunk_k"].zero_()
        inputs["tail_k"].zero_()
    else:
        # Different columns select different slots across the tail/chunk split.
        slots = torch.arange(size, device=cfg.device)[:, None]
        cols = torch.arange(128, device=cfg.device)[None, :]
        inputs["ape"].copy_(torch.where(slots == cols % size, 200.0, -1000.0))
        inputs["tail_score"] = inputs["tail_score"].to(tail_dtype)
    return inputs


def _masked_addresses_case(all_masked):
    inputs = _case(65, masked=True)
    mask = torch.zeros(66, device=cfg.device, dtype=torch.bool)[1:]
    if not all_masked:
        mask.copy_(inputs["write_mask"])
    inputs["write_mask"] = mask
    for name in ROW_INDICES:
        inputs[name][~mask] = -1_000_000
    return inputs


def _ring_case(size, tail_size, dtype):
    inputs = _case(65, size, tail_size=tail_size)
    inputs["tail_logical_base"].fill_(torch.iinfo(torch.int32).max)
    inputs["tail_logical_base"] = inputs["tail_logical_base"].to(dtype)
    return inputs


def _strided_case(masked):
    inputs = _case(65, masked=masked)
    for name in (
        "chunk_k",
        "chunk_score",
        "tail_k",
        "tail_score",
        "ape",
        "loc",
        "write_mask",
    ):
        value = inputs[name]
        if value is None:
            continue
        steps = (2, 2, 1) if name.startswith("tail_") else (2,) * value.ndim
        shape = tuple(n * step for n, step in zip(value.shape, steps))
        storage = value.new_full(shape, -99)
        inputs[name] = storage[
            tuple(slice(None, None, step) for step in steps)
        ].copy_(value)
    return inputs


def _pack_descriptors(inputs):
    wide = torch.stack(
        [
            inputs[name]
            for name in (
                "req_pool_idx",
                "req_pool_idx",
                "chunk_src_start",
                "req_pool_idx",
            )
        ]
    )
    narrow = torch.stack([inputs["n_from_tail"], inputs["tail_logical_base"]])
    inputs.update(
        req_pool_idx=wide[0],
        chunk_src_start=wide[2],
        n_from_tail=narrow[0],
        tail_logical_base=narrow[1],
    )
    return inputs


def _request_case(counts, tails, tail_size=4):
    size, page_size = 4, 16
    inputs = _case(sum(counts), tail_size=tail_size)
    descriptors = []
    offset, page = 0, 1
    for req, (count, tail) in enumerate(zip(counts, tails)):
        for pool in range(count):
            descriptors.append(
                (
                    req,
                    offset + max(0, pool * size - tail),
                    tail if pool == 0 else 0,
                    pool * size,
                    page * page_size + pool,
                )
            )
        offset += count * size - tail
        page += (count + page_size - 1) // page_size
    for name, values in zip(
        (
            "req_pool_idx",
            "chunk_src_start",
            "n_from_tail",
            "tail_logical_base",
            "loc",
        ),
        zip(*descriptors),
    ):
        inputs[name] = torch.tensor(
            values, dtype=inputs[name].dtype, device=cfg.device
        )
    inputs["buf"] = inputs["buf"].new_full((page, page_size * 132), 0xA5)
    for name in ("chunk_k", "chunk_score"):
        inputs[name] = inputs[name][:offset]
    return _pack_descriptors(inputs)


# K=4: empty/single rows, packed alignment, page and dispatch boundaries.
_CASE_FACTORIES = [
    partial(_case, 0),
    partial(_case, 1, source="chunk"),
    partial(_case, 2, source="tail"),
    partial(_case, 3),
    partial(_case, 17, source="chunk"),
    partial(_case, 64),
    partial(_case, 65),
    partial(_case, 257),
    partial(_case, 65, source="tail"),
    partial(_case, 65, source="chunk", rounded=False),
    partial(_case, 65, tail_size=10),
    partial(_case, 65, masked=True, page_size=1),
]
# Other supported K and score dtypes use the same public interface.
_CASE_FACTORIES += [partial(_case, 65, size) for size in (1, 3, 8, 16)]
_CASE_FACTORIES += [
    partial(_case, 65, dtype=dtype) for dtype in (torch.float16, torch.float32)
]
_CASE_FACTORIES += [
    partial(_numerical_edge_case, True),
    partial(_numerical_edge_case, False),
    partial(_numerical_edge_case, False, 16, True, torch.float16),
    partial(_masked_addresses_case, False),
    partial(_masked_addresses_case, True),
    partial(_ring_case, 4, 4, torch.int32),
    partial(_ring_case, 4, 3, torch.int32),
    partial(_ring_case, 3, 2, torch.int64),
    partial(_strided_case, False),
    partial(_strided_case, True),
    partial(_request_case, (2, 1), (1, 3)),
    partial(_request_case, (33,) + (32,) * 7, (1, 2, 3, 0) * 2),
    partial(_request_case, (17, 2), (3, 1), 10),
]
CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


@pytest.mark.parametrize("case_idx", CORRECTNESS_CASES)
def test_kpool_assemble_softmax_rotate_write_cache(case_idx):
    _check_workload(_CASE_FACTORIES[case_idx]())


@pytest.mark.parametrize(
    "rounded,a,b,scale,peak,mid",
    [
        (False, 1.5625, 1.2421875, 2**-9, 127, 15),
        (True, 0.1015625, 0.078125, 2**-12, 65, 9),
    ],
)
def test_int8_half_ties(rounded, a, b, scale, peak, mid):
    inputs = _case(65, source="chunk", rounded=rounded)
    inputs["buf"] = inputs["buf"].view(torch.int8)
    inputs["chunk_k"].zero_()
    inputs["chunk_k"][:, 1], inputs["chunk_k"][:, 2] = a, b
    inputs["chunk_score"].zero_()
    inputs["ape"].zero_()
    result = _check_workload(inputs)
    keys, scales = _cache_rows(result.cpu(), 16)
    loc = inputs["loc"].cpu()
    expected = torch.tensor([peak, -mid, mid, -peak], dtype=torch.int8).repeat(
        32
    )
    torch.testing.assert_close(
        keys[loc].view(torch.int8),
        expected.expand(len(loc), -1),
        rtol=0,
        atol=0,
    )
    assert (scales[loc].view(torch.float32) == scale).all()


@pytest.mark.parametrize(
    "second,expected_scale", [(408, 1), (412, 1), (416, 2)]
)
def test_round_scale_power_of_two_boundary(second, expected_scale):
    inputs = _case(65, source="chunk")
    inputs["chunk_k"].zero_()
    inputs["chunk_k"][:, 0] = 1024
    inputs["chunk_k"][:, 1] = second
    inputs["chunk_score"].zero_()
    inputs["ape"].zero_()
    # The BF16 Hadamard maxima are 126.5, 127, and 127.5, respectively.
    result = _check_workload(inputs)
    _, scales = _cache_rows(result.cpu(), 16)
    scale = scales[inputs["loc"].cpu()].view(torch.float32)
    assert (scale == expected_scale).all()


@pytest.mark.parametrize("masked", [False, True])
def test_graph_replay(masked):
    inputs = _case(65, masked=masked)
    reference = get_reference(OP)
    _check_workload(inputs)
    candidate = get_candidate(OP, reference=reference)
    if reference_mode_enabled():
        return
    device = inputs["buf"].device
    graph_type = getattr(
        torch_device_fn,
        "NPUGraph" if device.type == "npu" else "CUDAGraph",
        None,
    )
    if graph_type is None:
        pytest.skip("Device does not expose the graph replay API")
    current = torch_device_fn.current_stream(device)
    stream = torch_device_fn.Stream(device=device)
    stream.wait_stream(current)
    graph = graph_type()
    with torch_device_fn.stream(stream):
        candidate(**inputs)
        with torch_device_fn.graph(graph, stream=stream):
            candidate(**inputs)
    current.wait_stream(stream)
    # Replay must read updated values, source descriptors and masks.
    for count in (0, 4):
        inputs["n_from_tail"].fill_(count)
        inputs["chunk_k"].neg_()
        inputs["tail_k"].neg_()
        for name in ("req_pool_idx", "chunk_src_start", "loc"):
            inputs[name].copy_(inputs[name].roll(1))
        inputs["tail_logical_base"].add_(1)
        if masked:
            inputs["write_mask"].logical_not_()
        inputs["buf"].fill_(0xA5)
        before = inputs["buf"].clone()
        expected = reference(**_copy(inputs))
        graph.replay()
        current.synchronize()
        _check(inputs, before, inputs["buf"], expected)
