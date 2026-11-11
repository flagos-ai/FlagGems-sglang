# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""SGLang/Humming indexed W4A8 expert-layout qualification.

The framework owns method rebinding and Humming provider/source admission.
This boundary maps top-k route rows to the common SwiGLU/row-quant operator.
W13/W2 GEMMs and their numerical contracts remain in Humming.
"""

import torch
import triton

QUALIFIED_ROWS = frozenset(
    (
        6144,
        9216,
        12282,
        12288,
        12294,
        24570,
        24576,
        49146,
        49152,
        92160,
        98304,
        196608,
    )
)
W2_QUALIFIED_ROWS = frozenset((98304,))
W2_CONFIG = {
    "block_shape": [128, 128, 128],
    "warp_shape": [128, 32, 128],
    "use_stream_k": False,
    "use_f16_accum": False,
    "num_stages": 3,
    "num_ctas_per_sm": 2,
}


def supports(gate_up, topk_ids):
    return (
        isinstance(gate_up, torch.Tensor)
        and isinstance(topk_ids, torch.Tensor)
        and gate_up.ndim == 2
        and gate_up.shape[0] in QUALIFIED_ROWS
        and gate_up.shape[1] == 4608
        and gate_up.dtype == torch.bfloat16
        and topk_ids.ndim == 2
        and topk_ids.shape[1] == 6
        and topk_ids.numel() == gate_up.shape[0]
        and topk_ids.dtype in (torch.int32, torch.int64)
        and gate_up.is_cuda
        and topk_ids.device == gate_up.device
        and gate_up.is_contiguous()
        and topk_ids.is_contiguous()
        and triton.__version__.split("+")[0] == "3.6.0"
        and torch.cuda.get_device_capability(gate_up.device) == (9, 0)
    )


def indexed_swiglu_quant(gate_up, topk_ids, quanted, scales):
    if not supports(gate_up, topk_ids):
        raise ValueError("unsupported SGLang/Humming indexed W4A8 layout")
    from flag_gems.fused.dsv41.swiglu_row_quant import swiglu_row_quant
    from humming.ops.input import calc_scale, finalize_scale, quant_tensor

    return swiglu_row_quant(
        gate_up,
        topk_ids,
        quanted,
        scales,
        num_experts=48,
        calc_scale=calc_scale,
        finalize_scale=finalize_scale,
        quant_tensor=quant_tensor,
    )


__all__ = ["indexed_swiglu_quant"]
