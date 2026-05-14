import torch

from flag_dnn.ops.leaky_relu import leaky_relu as leaky_relu_op


def leaky_relu_(x: torch.Tensor, negative_slope: float = 0.01) -> torch.Tensor:
    return leaky_relu_op(x, negative_slope=negative_slope, inplace=True)
