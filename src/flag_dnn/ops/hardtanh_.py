import torch

from flag_dnn.ops.hardtanh import hardtanh as hardtanh_op


def hardtanh_(
    input: torch.Tensor, min_val: float = -1.0, max_val: float = 1.0
) -> torch.Tensor:
    return hardtanh_op(input, min_val, max_val, inplace=True)
