from typing import Optional, Union
import torch
from flag_dnn.ops.binary import binary


def add(
    input: torch.Tensor,
    other: Union[torch.Tensor, int, float],
    *,
    alpha: Union[int, float] = 1,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return binary(input, other, out=out, op_type="add", alpha=alpha)
