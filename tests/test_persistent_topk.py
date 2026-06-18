import pytest
import torch

# Reference: vLLM's CUDA persistent_topk as correctness baseline.
import vllm._C  # noqa: F401

import flaggems_sglang

from . import accuracy_utils as utils
from . import conftest as cfg

RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024


def persistent_topk_ref(logits, seq_lens, topk_indices, workspace, k, stride):
    """Reference implementation via vLLM CUDA kernel."""
    torch.ops._C.persistent_topk(logits, seq_lens, topk_indices, workspace, k, stride)


def _make_inputs(shape, device):
    """Create input tensors for persistent_topk.

    Args:
        shape: (num_rows, seq_len, k) tuple.
        device: Target device.

    Returns:
        Tuple of (logits, seq_lens, topk_indices_ref, topk_indices_res,
                  workspace, k, stride).
    """
    num_rows, seq_len, k = shape
    stride = seq_len
    logits = torch.randn(num_rows, stride, device=device, dtype=torch.float32)
    seq_lens = torch.full((num_rows,), seq_len, device=device, dtype=torch.int32)
    topk_indices_ref = torch.zeros(num_rows, k, device=device, dtype=torch.int32)
    topk_indices_res = torch.zeros(num_rows, k, device=device, dtype=torch.int32)
    workspace = torch.zeros(RADIX_TOPK_WORKSPACE_SIZE, device=device, dtype=torch.uint8)
    return logits, seq_lens, topk_indices_ref, topk_indices_res, workspace, k, stride


@pytest.mark.parametrize("shape", utils.PERSISTENT_TOPK_SHAPES)
@pytest.mark.persistent_topk
def test_persistent_topk(shape):
    device = cfg.device
    (
        logits,
        seq_lens,
        topk_indices_ref,
        topk_indices_res,
        workspace,
        k,
        stride,
    ) = _make_inputs(shape, device)

    # Reference (vLLM CUDA)
    persistent_topk_ref(logits, seq_lens, topk_indices_ref, workspace, k, stride)

    # Optimized (FlagGems Triton)
    flaggems_sglang.persistent_topk(logits, seq_lens, topk_indices_res, workspace, k, stride)

    # Compare: top-k indices are unordered, so compare selected values
    num_rows = logits.shape[0]
    for i in range(num_rows):
        ref_idx = topk_indices_ref[i].long()
        res_idx = topk_indices_res[i].long()

        # The k-th largest values selected must match
        ref_vals = torch.sort(logits[i][ref_idx], descending=True).values
        res_vals = torch.sort(logits[i][res_idx], descending=True).values

        torch.testing.assert_close(res_vals, ref_vals, atol=1e-6, rtol=1e-6)
