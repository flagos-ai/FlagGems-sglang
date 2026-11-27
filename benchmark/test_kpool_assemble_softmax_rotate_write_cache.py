# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""GLM53 extend latency against the supplied generic INT8 kernel."""

import random
from types import SimpleNamespace
from typing import NamedTuple

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.ops.kpool_assemble_softmax_rotate_write_cache import (
    kpool_assemble_softmax_rotate_write_cache as original,
)

from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark


class Case(NamedTuple):
    pools: tuple[int, ...]
    first_tails: tuple[int, ...]
    scattered: bool = False


# K=4, 16 slots/page, BF16 keys; no speculative decoding or context parallelism.
# Core keeps five small/boundary cases and eleven larger prefill workloads.
SHAPES = [
    Case((1,), (0,)),
    Case((1,), (3,)),
    Case((64,), (0,)),
    Case((65,), (0,)),
    Case((1,) * 65, (1, 2, 3, 0) * 16 + (3,)),
    Case((1, 2, 5, 13, 35, 72, 128, 256), (1, 2, 3, 0) * 2),
    Case(
        (1, 2, 5, 13, 35, 72, 128, 256),
        (1, 2, 3, 0) * 2,
        scattered=True,
    ),
    Case((96,) * 8, (1, 2, 3, 0) * 2),
    Case((1024,), (0,)),
    Case((1024,), (0,), scattered=True),
    Case((32,) * 32, (1, 2, 3, 0) * 8),
    Case((192,) * 8, (1, 2, 3, 0) * 2),
    Case((256,) * 8, (1, 2, 3, 0) * 2, scattered=True),
    Case((4096,), (0,)),
    Case((256,) * 16, (1, 2, 3, 0) * 4),
    Case((8192,), (0,)),
]

MORE_SHAPES = [
    Case((17,), (0,)),
    Case((32,), (0,)),
    Case((8, 8, 8, 8), (1, 2, 3, 0)),
    Case((33,) + (32,) * 7, (1, 2, 3, 0) * 2),
    Case((1,) * 64, (1, 2, 3, 0) * 16),
    Case((4,) * 32, (1, 2, 3, 0) * 8),
]


def _layout(case):
    requests = len(case.pools)
    used_pages = sum((count + 15) // 16 for count in case.pools)
    capacity, pages = requests, used_pages + 1
    request_slots = range(requests)
    physical_pages = range(1, pages)
    if case.scattered:
        # Sparse allocations with deterministic request slots and page tables.
        capacity, pages = requests * 16 + 1, used_pages * 8 + 1
        rng = random.Random(9)
        request_slots = rng.sample(range(1, capacity), requests)
        physical_pages = rng.sample(range(1, pages), used_pages)

    descriptors = []
    offset, page = 0, 0
    for req, count, tail in zip(request_slots, case.pools, case.first_tails):
        for pool in range(count):
            descriptors.append(
                (
                    req,
                    pool,
                    offset + max(0, pool * 4 - tail),
                    tail if pool == 0 else 0,
                    pool * 4,
                    physical_pages[page + pool // 16] * 16 + pool % 16,
                )
            )
        offset += count * 4 - tail
        page += (count + 15) // 16
    return tuple(zip(*descriptors)), offset, pages, capacity


def _plan(shape, cur_dtype):
    case = Case(*shape)
    columns, chunk_tokens, pages, capacity = _layout(case)
    rows, requests = sum(case.pools), len(case.pools)
    chunk, tail = (chunk_tokens, 128), (capacity, 4, 128)
    mixed = sum(count != 0 for count in case.first_tails)
    return (
        BenchmarkCasePlan(
            shape={
                "buf": (pages, 16 * 132),
                "chunk_k": chunk,
                "chunk_score": chunk,
                "tail_k": tail,
                "tail_score": tail,
                "ape": (4, 128),
                **{
                    name: (rows,)
                    for name in (
                        "req_pool_idx",
                        "n_from_tail",
                        "chunk_src_start",
                        "tail_logical_base",
                        "loc",
                    )
                },
                "write_mask": None,
            },
            params={
                "configuration": shape,
                "n_pools": rows,
                "index_kpool": 4,
                "head_dim": 128,
                "slots_per_page": 16,
                "tail_size": 4,
                "tail_extra_slots": 0,
                "source": "mixed" if mixed else "chunk",
                "requests": requests,
                "request_capacity": capacity,
                "cache_pages": pages,
                "cache_layout": (
                    "scattered" if case.scattered else "contiguous"
                ),
                "mixed_pools": mixed,
                "chunk_only_pools": rows - mixed,
                "tail_only_pools": 0,
                "descriptor_layout": "packed",
                "write_mask": None,
                "round_scale": True,
                "cache_format": "row-major int8 + float32 scale",
                "key_dtype": "torch.bfloat16",
                "score_dtype": str(cur_dtype),
                "ape_dtype": "torch.float32",
                "seed": 9,
                "baseline": "supplied_generic_int8",
            },
            builder_args=(shape,),
        ),
        columns,
    )


def _case_fn(shape, cur_dtype):
    plan, _ = _plan(shape, cur_dtype)
    yield plan


def _input_fn(shape, cur_dtype, device):
    plan, (req, pool, start, count, base, loc) = _plan(shape, cur_dtype)
    generator = torch.Generator(device=device).manual_seed(plan.params["seed"])

    def rand(name, dtype):
        return torch.randn(
            plan.shape[name], device=device, generator=generator
        ).to(dtype)

    inputs = dict(
        pool=SimpleNamespace(index_kpool=4, slots_per_page=16),
        buf=torch.full(
            plan.shape["buf"], 0xA5, dtype=torch.uint8, device=device
        ),
        chunk_k=rand("chunk_k", torch.bfloat16),
        chunk_score=rand("chunk_score", cur_dtype),
        tail_k=rand("tail_k", torch.bfloat16),
        tail_score=rand("tail_score", cur_dtype),
        ape=rand("ape", torch.float32),
        round_scale=True,
    )
    wide = torch.tensor(
        [req, pool, start, req], dtype=torch.int64, device=device
    )
    narrow = torch.tensor([count, base], dtype=torch.int32, device=device)
    inputs.update(
        req_pool_idx=wide[0],
        chunk_src_start=wide[2],
        n_from_tail=narrow[0],
        tail_logical_base=narrow[1],
        loc=torch.tensor(loc, dtype=torch.int64, device=device),
        write_mask=None,
    )
    yield (inputs,)


@pytest.mark.kpool_assemble_softmax_rotate_write_cache
def test_perf_kpool_assemble_softmax_rotate_write_cache():
    bench = OpBenchmark(
        op_name="kpool_assemble_softmax_rotate_write_cache",
        torch_op=original,
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=(torch.bfloat16, torch.float16, torch.float32),
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="(pools_per_request, first_tail_per_request, scattered)",
    )
    bench.set_gems_from_module(flaggems_sglang)
    bench.run()
