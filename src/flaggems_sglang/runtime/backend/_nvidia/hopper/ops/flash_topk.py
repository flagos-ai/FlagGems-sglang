# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""DeepSeek V3.2 FP8 index scoring and exact SGLang DSA TopK on SM90."""

import logging
from functools import lru_cache
from typing import Optional

import torch
import triton

from ._flash_topk_select import CANDIDATES, TOPK, select

logger = logging.getLogger(__name__)

__all__ = [
    "flash_topk",
    "flash_topk_workspace_size",
    "supports_flash_topk",
]

_PAGE_SIZE = 64
_PAGE_BYTES = 8448
_MAX_CONTEXT = 262144


@lru_cache(maxsize=1)
def _scorer():
    try:
        from ._flash_topk_scoring import score_paged
    except ImportError:
        return None
    return score_paged


def supports_flash_topk(device: torch.device) -> bool:
    """Check the NVIDIA Hopper and FlagTree TLE capabilities before dispatch."""
    device = torch.device(device)
    return (
        device.type == "cuda"
        and torch.version.cuda is not None
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability(device) == (9, 0)
        and _scorer() is not None
    )


def _pitch(max_context: int) -> int:
    return triton.cdiv(max_context, 256) * 256


def flash_topk_workspace_size(num_queries: int, max_context: int) -> int:
    """Return the required number of int32 elements in the caller's scratch."""
    if num_queries < 0 or not 0 <= max_context <= _MAX_CONTEXT:
        raise ValueError("FlashTopK supports 0 to 262144 visible tokens")
    if max_context <= TOPK:
        return 0
    candidate_size = 2 * CANDIDATES + 4 if max_context > CANDIDATES else 0
    return num_queries * (_pitch(max_context) + candidate_size)


def _check_metadata(name, tensor, shape, device):
    if (
        tuple(tensor.shape) != shape
        or tensor.dtype != torch.int32
        or tensor.device != device
    ):
        raise ValueError(f"{name} must be int32 {shape} on {device}")


def flash_topk(
    query: torch.Tensor,
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
    index_page_table: torch.Tensor,
    row_to_request: torch.Tensor,
    visible_lengths: torch.Tensor,
    max_context: int,
    out: torch.Tensor,
    workspace: Optional[torch.Tensor] = None,
    *,
    prefill: bool = False,
    output_page_table: Optional[torch.Tensor] = None,
    output_offsets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Write SGLang DSA Top-2048 indices and fill unused positions with -1.

    Query is [rows,64,128] E4M3FN, weights are FP32 [rows,64] or
    [rows,64,1] and already include the query scale. The index cache is
    SGLang's uint8 [pages,8448] layout, with 64 E4M3FN keys followed by
    64 FP32 scales. Each row supplies a request index and causal visible
    length. The valid window begins at the request's first cache token.
    Pass the actual visible bound as max_context, which may be tighter
    than the page-table capacity. Zero-length CUDA Graph padding rows
    have no index-cache accesses. Invalid visible lengths and request
    indices are masked on-device to protect padded rows. The index
    table and both optional
    transforms are caller-owned, already-validated scheduling metadata.

    With output_page_table, emit physical slots using that request's
    compact attention page table. Its page IDs may differ from index-K
    page IDs. With output_offsets, emit request-local IDs plus each row's
    flattened-KV base. With neither transform, emit request-local IDs.
    Only one transform may be passed. Score inputs must be finite.
    Selection uses the FP32 cutoff; ties and output order are unordered.

    The out tensor and workspace must be disjoint from each other and
    from inputs. The same workspace may be reused in stream order;
    concurrent calls on different streams need separate storage. The
    caller owns the scratch, output and cache through CUDA Graph replay.
    This entry point supports single-device, uncompressed FP8 index
    caches. Context/tensor parallel scheduling and cache writes stay
    with the caller.
    """
    logger.debug("GEMS_NVIDIA FLASH_TOPK")
    device = query.device
    if not supports_flash_topk(device):
        raise NotImplementedError("FlashTopK requires SM90 and FlagTree TLE")
    if (
        query.ndim != 3
        or tuple(query.shape[1:]) != (64, 128)
        or query.dtype != torch.float8_e4m3fn
        or query.stride(1) != 128
        or query.stride(2) != 1
    ):
        raise ValueError(
            "query must be E4M3FN [rows,64,128] with packed heads"
        )
    rows = query.shape[0]
    if tuple(weights.shape) == (rows, 64, 1):
        weights = weights.squeeze(-1)
    if (
        tuple(weights.shape) != (rows, 64)
        or weights.dtype != torch.float32
        or weights.device != device
        or weights.stride(1) != 1
    ):
        raise ValueError("weights must be FP32 [rows,64] with packed heads")
    if (
        index_k_cache.ndim != 2
        or index_k_cache.shape[1] != _PAGE_BYTES
        or index_k_cache.dtype != torch.uint8
        or index_k_cache.device != device
        or not index_k_cache.is_contiguous()
        or index_k_cache.data_ptr() % 16
    ):
        raise ValueError("index_k_cache must be aligned uint8 [pages,8448]")
    if index_page_table.ndim != 2:
        raise ValueError("index_page_table must be a compact rank-two table")
    _check_metadata(
        "index_page_table",
        index_page_table,
        tuple(index_page_table.shape),
        device,
    )
    if not 0 <= max_context <= index_page_table.shape[1] * _PAGE_SIZE:
        raise ValueError("max_context exceeds the index page table's capacity")
    if rows and max_context > TOPK and not index_k_cache.shape[0]:
        raise ValueError("a nonempty index cache is required for scoring")
    _check_metadata("row_to_request", row_to_request, (rows,), device)
    _check_metadata("visible_lengths", visible_lengths, (rows,), device)
    _check_metadata("out", out, (rows, TOPK), device)
    if out.stride(1) != 1 or (rows > 1 and out.stride(0) < TOPK):
        raise ValueError("out must have dense, nonoverlapping rows")

    if output_page_table is not None and output_offsets is not None:
        raise ValueError("choose either paged or ragged output")
    mode = "local"
    if output_page_table is not None:
        if (
            output_page_table.ndim != 2
            or output_page_table.shape[0] != index_page_table.shape[0]
            or output_page_table.shape[1] * _PAGE_SIZE < max_context
        ):
            raise ValueError("output_page_table does not cover all requests")
        _check_metadata(
            "output_page_table",
            output_page_table,
            tuple(output_page_table.shape),
            device,
        )
        mode = "paged"
    elif output_offsets is not None:
        _check_metadata("output_offsets", output_offsets, (rows,), device)
        mode = "ragged"

    required = flash_topk_workspace_size(rows, max_context)
    if required and (
        workspace is None
        or workspace.ndim != 1
        or not workspace.is_contiguous()
        or workspace.dtype != torch.int32
        or workspace.device != device
        or workspace.numel() < required
        or workspace.data_ptr() % 8
    ):
        raise ValueError(
            f"workspace must supply {required} aligned int32 elements on {device}"
        )
    if not rows:
        return out

    scores, candidates, meta = None, None, None
    pitch = _pitch(max_context)
    with torch.cuda.device(device):
        if required:
            score_elems = rows * pitch
            scores = workspace[:score_elems].view(torch.float32)
            if max_context > CANDIDATES:
                after_candidates = score_elems + 2 * rows * CANDIDATES
                candidates = workspace[score_elems:after_candidates].view(
                    torch.int64
                )
                meta = workspace[after_candidates:required]
            _scorer()(
                query,
                index_k_cache,
                weights,
                index_page_table,
                row_to_request,
                visible_lengths,
                scores,
                max_context,
                pitch,
                prefill=prefill,
            )
        select(
            scores,
            visible_lengths,
            row_to_request,
            output_page_table,
            output_offsets,
            out,
            max_context,
            pitch,
            mode,
            index_page_table.shape[0],
            candidates,
            meta,
        )
    return out
