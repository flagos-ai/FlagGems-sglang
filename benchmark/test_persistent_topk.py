import pytest
import torch

# Reference: vLLM's CUDA persistent_topk as performance baseline.
import vllm._C  # noqa: F401

import flaggems_sglang

from .attri_util import PERSISTENT_TOPK_BENCH_SHAPES

RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024


def persistent_topk_ref(logits, seq_lens, topk_indices, workspace, k, stride):
    """Reference implementation via vLLM CUDA kernel."""
    torch.ops._C.persistent_topk(logits, seq_lens, topk_indices, workspace, k, stride)


def _make_inputs(shape, device):
    """Create input tensors for a benchmark configuration.

    Args:
        shape: (num_rows, seq_len, k) tuple.
        device: Target device.
    """
    num_rows, seq_len, k = shape
    stride = seq_len
    logits = torch.randn(num_rows, stride, device=device, dtype=torch.float32)
    seq_lens = torch.full((num_rows,), seq_len, device=device, dtype=torch.int32)
    topk_indices = torch.zeros(num_rows, k, device=device, dtype=torch.int32)
    workspace = torch.zeros(RADIX_TOPK_WORKSPACE_SIZE, device=device, dtype=torch.uint8)
    return logits, seq_lens, topk_indices, workspace, k, stride


@pytest.mark.parametrize("shape", PERSISTENT_TOPK_BENCH_SHAPES)
@pytest.mark.persistent_topk
def test_persistent_topk(shape, benchmark):
    device = flaggems_sglang.device

    logits, seq_lens, topk_indices, workspace, k, stride = _make_inputs(shape, device)

    def run():
        idx = topk_indices.clone()
        flaggems_sglang.persistent_topk(logits, seq_lens, idx, workspace, k, stride)

    benchmark(run)
