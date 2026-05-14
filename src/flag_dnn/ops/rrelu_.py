import torch

from flag_dnn.ops.rrelu import rrelu as rrelu_op


def rrelu_(
    input: torch.Tensor,
    lower: float = 1.0 / 8,
    upper: float = 1.0 / 3,
    training: bool = False,
) -> torch.Tensor:
    return rrelu_op(input, lower, upper, training, inplace=True)
