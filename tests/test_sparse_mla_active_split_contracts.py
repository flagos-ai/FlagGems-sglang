# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU metadata contracts; accelerator capability is simulated, not qualified."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _load_operator():
    path = (
        Path(__file__).resolve().parents[1]
        / "src/flaggems_sglang/ops/sparse_mla_active_split.py"
    )
    spec = importlib.util.spec_from_file_location(
        "active_split_contract", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TensorMetadata:
    """Only static tensor facts; no storage or accelerator allocation."""

    def __init__(self, shape, strides, dtype, device, pointer):
        self.shape = shape
        self.ndim = len(shape)
        self.strides = strides
        self.dtype = dtype
        self.device = device
        self.pointer = pointer
        self.requires_grad = False
        self.offset = 0

    def stride(self):
        return self.strides

    def storage_offset(self):
        return self.offset

    def is_conj(self):
        return False

    def is_neg(self):
        return False

    def is_contiguous(self):
        return True

    def untyped_storage(self):
        return SimpleNamespace(data_ptr=lambda: self.pointer)


def _inputs(rows):
    device = SimpleNamespace(type="cuda")
    specs = (
        ((rows, 64, 576), (36864, 576, 1), torch.bfloat16),
        ((2113, 1, 576), (576, 576, 1), torch.bfloat16),
        ((rows, 1, 2048), (2048, 2048, 1), torch.int32),
        ((64,), (1,), torch.float32),
        ((rows,), (1,), torch.int32),
    )
    return [
        TensorMetadata(shape, strides, dtype, device, index + 1)
        for index, (shape, strides, dtype) in enumerate(specs)
    ]


@pytest.mark.parametrize("rows", [1, 4])
@pytest.mark.parametrize("field", ["dtype", "stride", "offset", "gradient"])
@torch.no_grad()
def test_metadata_rejects_one_changed_contract(monkeypatch, rows, field):
    module = _load_operator()
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device: (9, 0)
    )
    inputs = _inputs(rows)
    q, kv, indices, sink, lengths = inputs
    assert module.metadata(q, kv, indices, 0.125, 4, sink, lengths) is not None
    if field == "dtype":
        q.dtype = torch.float16
    elif field == "stride":
        q.strides = (36865, 576, 1)
    elif field == "offset":
        q.offset = 1
    else:
        q.requires_grad = True
    assert module.metadata(q, kv, indices, 0.125, 4, sink, lengths) is None


@torch.no_grad()
def test_workspace_alias_is_rejected_before_output_allocation(monkeypatch):
    module = _load_operator()
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device: (9, 0)
    )
    q, kv, indices, sink, lengths = _inputs(1)
    partials = [
        TensorMetadata(
            (1, 32, 4, 512), (), torch.float32, q.device, q.pointer
        ),
        TensorMetadata((1, 32, 4), (), torch.float32, q.device, 100),
        TensorMetadata((1, 32, 4), (), torch.float32, q.device, 101),
    ]
    assert module.metadata(q, kv, indices, 0.125, 4, sink, lengths) is not None

    def unexpected_allocation(*args, **kwargs):
        pytest.fail("output allocation started before alias rejection")

    monkeypatch.setattr(torch, "empty", unexpected_allocation)
    with pytest.raises(NotImplementedError, match="independent owners"):
        module.sparse_mla_active_split(
            q,
            kv,
            indices,
            0.125,
            active_heads=4,
            sink=sink,
            lengths=lengths,
            partials=partials,
        )
