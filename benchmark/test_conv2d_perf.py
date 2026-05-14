import os
from typing import Generator, Sequence, Tuple, Union

import pytest
import torch
import torch.nn.functional as F

import flag_dnn
from benchmark.performance_utils import Benchmark

Conv2dShape = Tuple[
    Tuple[int, int, int, int],  # input shape: N, C, H, W
    Tuple[int, int, int, int],  # weight shape: OC, IC/groups, KH, KW
    bool,  # has_bias
    Union[int, Tuple[int, int]],  # stride
    Union[int, Tuple[int, int]],  # padding
    Union[int, Tuple[int, int]],  # dilation
    int,  # groups
]


def _pair(v: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(v, int):
        return v, v
    if len(v) != 2:
        raise RuntimeError(f"expected length 2, but got {v}")
    return int(v[0]), int(v[1])


def _conv_out_dim(
    input_size: int,
    pad: int,
    dilation: int,
    kernel: int,
    stride: int,
) -> int:
    return (input_size + 2 * pad - dilation * (kernel - 1) - 1) // stride + 1


def torch_conv2d(x, w, b, stride, padding, dilation, groups):
    return F.conv2d(
        x,
        w,
        b,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


def gems_conv2d_wrapper(x, w, b, stride, padding, dilation, groups):
    return flag_dnn.ops.conv2d(
        x,
        w,
        b,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


class Conv2dBenchmark(Benchmark):
    MAX_PEAK_BYTES = 8 * 1024**3

    def set_more_metrics(self):
        return ["gbps", "gflops"]

    def set_more_shapes(self):
        self.shapes: list[Conv2dShape] = [
            # stem / 大核
            ((32, 3, 224, 224), (64, 3, 7, 7), False, 2, 3, 1, 1),
            # 常见 3x3
            ((32, 64, 56, 56), (64, 64, 3, 3), False, 1, 1, 1, 1),
            ((32, 64, 56, 56), (128, 64, 3, 3), False, 2, 1, 1, 1),
            ((32, 128, 28, 28), (128, 128, 3, 3), False, 1, 1, 1, 1),
            ((32, 256, 14, 14), (256, 256, 3, 3), False, 1, 1, 1, 1),
            ((32, 512, 7, 7), (512, 512, 3, 3), False, 1, 1, 1, 1),
            # 常见 1x1
            ((32, 64, 56, 56), (64, 64, 1, 1), False, 1, 0, 1, 1),
            ((32, 64, 56, 56), (256, 64, 1, 1), False, 1, 0, 1, 1),
            ((32, 128, 28, 28), (256, 128, 1, 1), False, 1, 0, 1, 1),
            ((32, 256, 14, 14), (512, 256, 1, 1), False, 1, 0, 1, 1),
            # 带 bias
            ((16, 64, 56, 56), (64, 64, 3, 3), True, 1, 1, 1, 1),
            ((16, 128, 28, 28), (128, 128, 1, 1), True, 1, 0, 1, 1),
            # dilation
            ((16, 64, 56, 56), (64, 64, 3, 3), False, 1, 2, 2, 1),
            ((16, 128, 28, 28), (128, 128, 3, 3), False, 1, 4, 4, 1),
            # group conv
            ((16, 64, 56, 56), (128, 32, 3, 3), False, 1, 1, 1, 2),
            ((16, 128, 28, 28), (128, 16, 3, 3), False, 1, 1, 1, 8),
            # depthwise conv
            ((16, 32, 112, 112), (32, 1, 3, 3), False, 1, 1, 1, 32),
            ((16, 64, 56, 56), (64, 1, 3, 3), False, 1, 1, 1, 64),
            ((16, 128, 28, 28), (128, 1, 5, 5), False, 1, 2, 1, 128),
        ]
        only = os.getenv("FLAGDNN_CONV2D_PERF_SHAPE_IDS")
        if only:
            selected = {int(item) for item in only.split(",") if item.strip()}
            self.shapes = [
                shape
                for idx, shape in enumerate(self.shapes)
                if idx in selected
            ]
        return None

    @staticmethod
    def _tensor_nbytes(shape, dtype):
        return (
            torch.empty(shape, dtype=dtype).numel()
            * torch.empty((), dtype=dtype).element_size()
        )

    def _output_shape(
        self,
        input_shape,
        weight_shape,
        stride,
        padding,
        dilation,
    ):
        n, _, ih, iw = input_shape
        oc, _, kh, kw = weight_shape
        stride_h, stride_w = _pair(stride)
        pad_h, pad_w = _pair(padding)
        dil_h, dil_w = _pair(dilation)

        oh = _conv_out_dim(ih, pad_h, dil_h, kh, stride_h)
        ow = _conv_out_dim(iw, pad_w, dil_w, kw, stride_w)
        return (n, oc, oh, ow)

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
        for shape in self.shapes:
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
            tuple(x.shape),
            tuple(w.shape),
            stride,
            padding,
            dilation,
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
        n, c_in, ih, iw = x.shape
        c_out, _, kh, kw = w.shape
        stride_h, stride_w = _pair(stride)
        pad_h, pad_w = _pair(padding)
        dil_h, dil_w = _pair(dilation)

        oh = _conv_out_dim(ih, pad_h, dil_h, kh, stride_h)
        ow = _conv_out_dim(iw, pad_w, dil_w, kw, stride_w)

        # FLOPs = 2 * N * OH * OW * C_out * (C_in / groups) * KH * KW
        flops = 2 * n * oh * ow * c_out * (c_in // groups) * kh * kw
        return flops / (latency * 1e-3) / 1e9


@pytest.mark.conv2d
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_perf_conv2d(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    bench = Conv2dBenchmark(
        op_name="conv2d",
        torch_op=torch_conv2d,
        gems_op=gems_conv2d_wrapper,
        dtypes=[dtype],
    )
    bench.run()
