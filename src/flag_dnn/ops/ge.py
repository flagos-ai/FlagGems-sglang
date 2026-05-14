from typing import Optional, Union
import torch
from flag_dnn.ops.binary import binary


def ge(
    input: torch.Tensor,
    other: Union[torch.Tensor, int, float],
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return binary(input, other, out=out, op_type="ge")
