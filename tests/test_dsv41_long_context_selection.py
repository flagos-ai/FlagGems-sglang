# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU-only selection contracts for long-context SGLang indexer paths."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


class Tensor:
    def __init__(
        self, shape, dtype, *, device="cuda:0", cuda=True, stride=None, ptr=256
    ):
        self.shape = tuple(shape)
        self.ndim = len(shape)
        self.dtype = dtype
        self.device = device
        self.is_cuda = cuda
        self._stride = stride
        self.ptr = ptr

    def stride(self, dim):
        if self._stride is not None:
            return self._stride[dim]
        result = 1
        for size in self.shape[dim + 1 :]:
            result *= size
        return result

    def data_ptr(self):
        return self.ptr


@pytest.fixture
def long_context():
    path = (
        Path(__file__).parents[1]
        / "src/flaggems_sglang/runtime/backend/_nvidia/hopper/ops"
        / "dsv41_fp4_long_context.py"
    )
    spec = importlib.util.spec_from_file_location(
        "dsv41_long_context_test", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.torch = SimpleNamespace(
        bfloat16="bf16",
        int32="i32",
        int64="i64",
        uint8="u8",
        cuda=SimpleNamespace(
            get_device_properties=lambda device: SimpleNamespace(
                major=9, minor=0, multi_processor_count=132
            )
        ),
    )
    return module


def inputs(batch=108):
    return [
        Tensor((batch, 32, 128), "bf16"),
        Tensor((batch, 32), "bf16"),
        Tensor((18, 1048843), "i32"),
        Tensor((batch,), "i64"),
        Tensor((batch,), "i64"),
        Tensor((200000, 4352), "u8"),
        64,
        1,
        1048843,
        8,
        False,
        6,
    ]


@pytest.mark.parametrize(
    "batch",
    [6, 12, 18, 24, 30, 36, 42, 48, 60, 72, 84, 96, 108, 120, 144, 192],
)
def test_source_and_compact_admit_only_qualified_shapes(long_context, batch):
    args = inputs(batch)
    assert long_context.producer_family(*args) == "source"
    blocks, counts = Tensor((batch, 2048), "i32"), Tensor((batch,), "i32")
    assert long_context.compact_supported(
        *args[:4], blocks, counts, args[5], 64, 1, 8
    )
    args[7:11] = [2, 524421, 0, True]
    expected = "dense" if batch in (108, 120, 144, 192) else None
    assert long_context.producer_family(*args) == expected


@pytest.mark.parametrize(
    "index,value",
    [
        (0, Tensor((108, 31, 128), "bf16")),
        (0, Tensor((108, 32, 128), "fp16")),
        (1, Tensor((108, 32), "bf16", device="cuda:1")),
        (2, Tensor((18, 1048843), "i64")),
        (2, Tensor((18, 655627), "i32")),
        (3, Tensor((108,), "i32")),
        (4, Tensor((108,), "i32", cuda=False)),
        (5, Tensor((200000, 4352), "u8", ptr=257)),
        (6, 128),
        (7, 2),
        (8, 1048844),
        (9, 16),
        (10, True),
        (11, 1),
    ],
)
def test_each_invalid_source_contract_falls_back(long_context, index, value):
    args = inputs()
    args[index] = value
    assert long_context.producer_family(*args) is None
    assert long_context.dsv41_long_context_producer(*args) is None


def test_unqualified_device_and_batch_fall_back(long_context):
    assert long_context.producer_family(*inputs(54)) is None
    long_context.torch.cuda.get_device_properties = (
        lambda device: SimpleNamespace(
            major=9, minor=0, multi_processor_count=120
        )
    )
    assert long_context.producer_family(*inputs()) is None


def test_invalid_compact_inputs_fall_back(long_context):
    args = inputs()
    blocks, counts = Tensor((108, 2048), "i32"), Tensor((108,), "i32")
    assert long_context.compact_supported(
        *args[:4], blocks, counts, args[5], 64, 1, 8
    )
    bad = Tensor((108, 2048), "i32", device="cuda:1")
    assert not long_context.compact_supported(
        *args[:4], bad, counts, args[5], 64, 1, 8
    )
    assert (
        long_context.dsv41_long_context_compact(
            *args[:4], bad, counts, args[5], 64, 1, 8
        )
        is None
    )


def test_source_failure_after_dispatch_propagates(long_context, monkeypatch):
    import sys
    from types import ModuleType

    parent = ModuleType("dsv41_test_ops")
    parent.__path__ = []
    native = ModuleType("dsv41_test_ops.dsv41_fp4_indexer")

    def fail(*args):
        raise RuntimeError("launch failure")

    native._tle_native_fp8 = lambda: fail
    gems = ModuleType("flag_gems")
    gems.__path__ = []
    fused = ModuleType("flag_gems.fused")
    fused.__path__ = []
    dsv41 = ModuleType("flag_gems.fused.dsv41")
    dsv41.__path__ = []
    score = ModuleType("flag_gems.fused.dsv41.fp4_fp8")
    score.FP4_FP8_SCORE_ABI = 2
    for module in (parent, native, gems, fused, dsv41, score):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    long_context.__package__ = "dsv41_test_ops"
    long_context.__spec__ = None
    with pytest.raises(RuntimeError, match="launch failure"):
        long_context.dsv41_long_context_producer(*inputs())


def test_unavailable_source_falls_back_before_dispatch(
    long_context, monkeypatch
):
    import sys
    from types import ModuleType

    parent = ModuleType("dsv41_test_ops")
    parent.__path__ = []
    native = ModuleType("dsv41_test_ops.dsv41_fp4_indexer")
    native._tle_native_fp8 = lambda: None
    gems = ModuleType("flag_gems")
    gems.__path__ = []
    fused = ModuleType("flag_gems.fused")
    fused.__path__ = []
    dsv41 = ModuleType("flag_gems.fused.dsv41")
    dsv41.__path__ = []
    score = ModuleType("flag_gems.fused.dsv41.fp4_fp8")
    score.FP4_FP8_SCORE_ABI = 2
    for module in (parent, native, gems, fused, dsv41, score):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    long_context.__package__ = "dsv41_test_ops"
    long_context.__spec__ = None
    assert long_context.dsv41_long_context_producer(*inputs()) is None


def test_public_entries_preserve_native_keyword_names(long_context):
    args = inputs()
    # Exercise the unsupported path with keyword-only call sites; no GPU is used.
    args[6] = 128
    assert (
        long_context.dsv41_long_context_producer(
            q=args[0],
            weights=args[1],
            req_to_token=args[2],
            req=args[3],
            lens=args[4],
            table=args[5],
            page_size=args[6],
            ratio=args[7],
            width=args[8],
            candidate_block_size=args[9],
            write_logits=args[10],
            query_group_size=args[11],
        )
        is None
    )
    blocks, counts = Tensor((108, 2048), "i32"), Tensor((108,), "i32")
    assert (
        long_context.dsv41_long_context_compact(
            q=args[0],
            weights=args[1],
            req_to_token=args[2],
            req=args[3],
            candidate_blocks=blocks,
            candidate_lens=counts,
            table=args[5],
            page_size=128,
            ratio=1,
            candidate_block_size=8,
        )
        is None
    )
