import torch
from flag_dnn.ops.elu import _elu_impl


def elu_(input: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    return _elu_impl(
        input,
        alpha=alpha,
        scale=1.0,
        input_scale=1.0,
        inplace=True,
    )
