import torch

from flag_dnn.ops.threshold import threshold as threshold_op


def threshold_(
    input: torch.Tensor, threshold: float, value: float
) -> torch.Tensor:
    return threshold_op(input, threshold, value, inplace=True)
