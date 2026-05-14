import pytest
import torch
import flag_dnn

# 提取公共的测试形状和广播用例
COMPARE_CASES = [
    # 相同形状
    ((1024,), (1024,)),
    ((2, 3, 4), (2, 3, 4)),
    # 标量比较
    ((128, 256), 0.5),
    ((10, 10), 0),
    # Broadcasting 广播机制
    ((10, 1), (1, 20)),  # 互相扩展
    ((2, 3, 4), (4,)),  # 向前补齐
    ((1, 3, 1, 5), (2, 1, 4, 1)),  # 复杂高维广播
]

DTYPES = [
    torch.float32,
    torch.float64,
    torch.float16,
    torch.bfloat16,
    torch.int64,
    torch.int32,
    torch.int16,
    torch.int8,
    torch.uint8,
    torch.bool,
]


def get_test_inputs(input_shape, other_spec, dtype, device):
    if dtype == torch.bool:
        x = torch.randint(0, 2, input_shape, device=device).bool()
    elif dtype == torch.uint8:
        x = torch.randint(0, 5, input_shape, dtype=dtype, device=device)
    elif not dtype.is_floating_point:  # 其他有符号整型
        x = torch.randint(-5, 5, input_shape, dtype=dtype, device=device)
    else:  # 浮点型
        x = torch.randn(input_shape, dtype=dtype, device=device)

    if isinstance(other_spec, tuple):
        if dtype == torch.bool:
            y = torch.randint(0, 2, other_spec, device=device).bool()
        elif dtype == torch.uint8:
            y = torch.randint(0, 5, other_spec, dtype=dtype, device=device)
        elif not dtype.is_floating_point:
            y = torch.randint(-5, 5, other_spec, dtype=dtype, device=device)
        else:
            y = torch.randn(other_spec, dtype=dtype, device=device)

        if input_shape == other_spec:
            mask = torch.rand(input_shape, device=device) > 0.5
            y = torch.where(mask, x, y)
    else:
        y = torch.tensor(other_spec, dtype=dtype).item()

    return x, y


@pytest.mark.le
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("input_shape, other_spec", COMPARE_CASES)
def test_accuracy_le(dtype, input_shape, other_spec):
    x, y = get_test_inputs(input_shape, other_spec, dtype, flag_dnn.device)

    ref_out = torch.le(x, y)
    with flag_dnn.use_dnn():
        out = torch.le(x, y)

    assert out.dtype == torch.bool
    torch.testing.assert_close(out, ref_out)


@pytest.mark.le
def test_accuracy_le_with_out_param():
    x = torch.tensor([1.0, 2.0, 3.0], device=flag_dnn.device)
    y = torch.tensor([1.0, 3.0, 2.0], device=flag_dnn.device)
    ref_out = torch.empty((3,), dtype=torch.bool, device=flag_dnn.device)
    custom_out = torch.empty((3,), dtype=torch.bool, device=flag_dnn.device)
    custom_out.fill_(True)

    torch.le(x, y, out=ref_out)
    with flag_dnn.use_dnn():
        torch.le(x, y, out=custom_out)
    torch.testing.assert_close(custom_out, ref_out)
