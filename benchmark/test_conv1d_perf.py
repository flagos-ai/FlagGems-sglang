from typing import Generator, Sequence, Tuple, Union

import pytest
import torch
import torch.nn.functional as F

import flag_dnn
from benchmark.performance_utils import Benchmark

Conv1dShape = Tuple[
    Tuple[int, ...],  # input shape: N, C, L or C, L
    Tuple[int, int, int],  # weight shape: OC, IC/groups, K
    bool,  # has_bias
    Union[int, Tuple[int]],  # stride
    Union[int, Tuple[int]],  # padding
    Union[int, Tuple[int]],  # dilation
    int,  # groups
]


def _single(v: int | Sequence[int]) -> int:
    if isinstance(v, int):
        return v
    if len(v) != 1:
        raise RuntimeError(f"expected length 1, but got {v}")
    return int(v[0])


def _conv_out_dim(
    input_size: int,
    pad: int,
    dilation: int,
    kernel: int,
    stride: int,
) -> int:
    return (input_size + 2 * pad - dilation * (kernel - 1) - 1) // stride + 1


def torch_conv1d(x, w, b, stride, padding, dilation, groups):
    return F.conv1d(
        x,
        w,
        b,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


def gems_conv1d_wrapper(x, w, b, stride, padding, dilation, groups):
    return flag_dnn.ops.conv1d(
        x,
        w,
        b,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


class Conv1dBenchmark(Benchmark):
    MAX_PEAK_BYTES = 4 * 1024**3
    FP64_ACTIVE_SHAPES: list[Conv1dShape] = [
        # Keep fp64 shapes smaller than the main active set because the
        # implementation uses a scalar fp64 accumulation path.
        ((16, 16, 127), (32, 16, 3), False, 1, 1, 1, 1),
        ((8, 32, 257), (64, 32, 5), True, 2, 2, 1, 1),
        ((4, 16, 513), (32, 16, 7), False, 1, 3, 1, 1),
        ((4, 32, 512), (64, 32, 1), True, 1, 0, 1, 1),
        ((4, 32, 384), (64, 16, 3), False, 1, 1, 1, 2),
        ((4, 32, 512), (32, 1, 5), False, 1, 2, 1, 32),
        ((32, 512), (64, 32, 3), True, 1, 1, 1, 1),
    ]

    def set_more_metrics(self):
        return ["gbps", "gflops"]

    def set_more_shapes(self):
        self.shapes: list[Conv1dShape] = [
            # Small and irregular lengths.
            ((16, 16, 127), (32, 16, 3), False, 1, 1, 1, 1),
            ((8, 32, 257), (64, 32, 5), True, 2, 2, 1, 1),
            # Audio / sequence style lengths.
            ((32, 64, 1024), (64, 64, 3), False, 1, 1, 1, 1),
            ((16, 64, 2048), (128, 64, 7), False, 2, 3, 1, 1),
            ((8, 128, 4096), (128, 128, 3), True, 1, 2, 2, 1),
            # 1x1 projection.
            ((32, 64, 1024), (128, 64, 1), False, 1, 0, 1, 1),
            ((16, 128, 2048), (256, 128, 1), True, 1, 0, 1, 1),
            # Group and depthwise cases.
            ((16, 64, 1024), (128, 32, 3), False, 1, 1, 1, 2),
            ((16, 128, 1024), (128, 16, 3), False, 1, 1, 1, 8),
            ((32, 64, 1024), (64, 1, 5), False, 1, 2, 1, 64),
            # Unbatched inference shape.
            ((64, 2048), (128, 64, 3), True, 1, 1, 1, 1),
        ]
        return None

    @staticmethod
    def _tensor_nbytes(shape, dtype):
        return (
            torch.empty(shape, dtype=dtype).numel()
            * torch.empty((), dtype=dtype).element_size()
        )

    def _output_shape(
        self, input_shape, weight_shape, stride, padding, dilation
    ):
        c_out, _, kernel = weight_shape
        stride_w = _single(stride)
        pad_w = _single(padding)
        dil_w = _single(dilation)
        input_l = input_shape[-1]
        out_l = _conv_out_dim(input_l, pad_w, dil_w, kernel, stride_w)
        if len(input_shape) == 2:
            return (c_out, out_l)
        return (input_shape[0], c_out, out_l)

    def _estimate_peak_bytes(self, shape, dtype):
        input_shape, weight_shape, has_bias, stride, padding, dilation, _ = (
            shape
        )
        output_shape = self._output_shape(
            input_shape, weight_shape, stride, padding, dilation
        )

        input_bytes = self._tensor_nbytes(input_shape, dtype)
        weight_bytes = self._tensor_nbytes(weight_shape, dtype)
        bias_bytes = (
            self._tensor_nbytes((weight_shape[0],), dtype) if has_bias else 0
        )
        output_bytes = self._tensor_nbytes(output_shape, dtype)
        return input_bytes + weight_bytes + bias_bytes + output_bytes

    def get_input_iter(self, cur_dtype) -> Generator:
        shapes = self.shapes
        if cur_dtype == torch.float64:
            shapes = self.FP64_ACTIVE_SHAPES

        for shape in shapes:
            if (
                self._estimate_peak_bytes(shape, cur_dtype)
                > self.MAX_PEAK_BYTES
            ):
                continue

            (
                input_shape,
                weight_shape,
                has_bias,
                stride,
                padding,
                dilation,
                groups,
            ) = shape

            x = torch.empty(
                input_shape, dtype=cur_dtype, device=self.device
            ).uniform_(-1.0, 1.0)
            w = torch.empty(
                weight_shape, dtype=cur_dtype, device=self.device
            ).uniform_(-1.0, 1.0)
            b = (
                torch.empty(
                    (weight_shape[0],), dtype=cur_dtype, device=self.device
                ).uniform_(-1.0, 1.0)
                if has_bias
                else None
            )

            yield (x, w, b, stride, padding, dilation, groups)

    def get_gbps(self, args, latency):
        x, w, b, stride, padding, dilation, _ = args
        output_shape = self._output_shape(
            tuple(x.shape), tuple(w.shape), stride, padding, dilation
        )

        input_bytes = x.numel() * x.element_size()
        weight_bytes = w.numel() * w.element_size()
        bias_bytes = 0 if b is None else b.numel() * b.element_size()
        output_bytes = (
            torch.empty(output_shape, dtype=x.dtype).numel() * x.element_size()
        )

        io_amount = input_bytes + weight_bytes + bias_bytes + output_bytes
        return io_amount / (latency * 1e-3) / 1e9

    def get_gflops(self, args, latency):
        x, w, _, stride, padding, dilation, groups = args
        c_out, _, kernel = w.shape
        stride_w = _single(stride)
        pad_w = _single(padding)
        dil_w = _single(dilation)
        out_l = _conv_out_dim(x.shape[-1], pad_w, dil_w, kernel, stride_w)
        batch = x.shape[0] if x.dim() == 3 else 1
        c_in = x.shape[-2]

        flops = 2 * batch * out_l * c_out * (c_in // groups) * kernel
        return flops / (latency * 1e-3) / 1e9


@pytest.mark.conv1d
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_perf_conv1d(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    bench = Conv1dBenchmark(
        op_name="conv1d",
        torch_op=torch_conv1d,
        gems_op=gems_conv1d_wrapper,
        dtypes=[dtype],
    )
    bench.run()
