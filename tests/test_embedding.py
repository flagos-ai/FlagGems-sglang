import pytest
import torch
import torch.nn.functional as F

import flag_dnn
from . import accuracy_utils as utils
from . import conftest as cfg


DTYPES = [torch.float32] if cfg.QUICK_MODE else utils.ALL_FLOAT_DTYPES
INDEX_DTYPES = [torch.int64, torch.int32]

CASES = [
    ((0,), 8, 4, "random"),
    ((1,), 8, 1, "random"),
    ((7,), 32, 17, "random"),
    ((2, 3), 64, 32, "repeat"),
    ((4, 5), 128, 63, "hotspot"),
    ((2, 0, 3), 16, 8, "random"),
]


def _skip_dtype(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")


def _make_indices(shape, vocab_size, index_dtype, pattern):
    numel = int(torch.tensor(shape).prod().item())
    if numel == 0:
        return torch.empty(shape, dtype=index_dtype, device=flag_dnn.device)
    if pattern == "repeat":
        base = torch.arange(
            0, min(vocab_size, 4), dtype=index_dtype, device=flag_dnn.device
        )
        reps = (numel + base.numel() - 1) // base.numel()
        return base.repeat(reps)[:numel].reshape(shape)
    if pattern == "hotspot":
        out = torch.zeros(shape, dtype=index_dtype, device=flag_dnn.device)
        if out.numel() > 1:
            out.reshape(-1)[1::4] = torch.randint(
                0,
                vocab_size,
                (out.reshape(-1)[1::4].numel(),),
                dtype=index_dtype,
                device=flag_dnn.device,
            )
        return out
    return torch.randint(
        0, vocab_size, shape, dtype=index_dtype, device=flag_dnn.device
    )


def _make_weight(vocab_size, embedding_dim, dtype):
    return torch.empty(
        (vocab_size, embedding_dim), dtype=dtype, device=flag_dnn.device
    ).normal_()


@pytest.mark.embedding
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("shape,vocab_size,embedding_dim,pattern", CASES)
def test_accuracy_embedding(
    dtype, index_dtype, shape, vocab_size, embedding_dim, pattern
):
    _skip_dtype(dtype)

    indices = _make_indices(shape, vocab_size, index_dtype, pattern)
    weight = _make_weight(vocab_size, embedding_dim, dtype)

    ref_indices = utils.to_reference(indices, ref_kind=None)
    ref_weight = utils.to_reference(weight, ref_kind="compute")
    ref = F.embedding(ref_indices, ref_weight)
    with flag_dnn.use_dnn(include=["embedding"]):
        out = F.embedding(indices, weight)

    assert out.shape == ref.shape
    assert out.dtype == dtype
    utils.gems_assert_close(out, ref, dtype, atol=0, equal_nan=True)


@pytest.mark.embedding
@pytest.mark.parametrize("dtype", DTYPES)
def test_embedding_non_contiguous_input_and_weight(dtype):
    _skip_dtype(dtype)

    indices = torch.randint(
        0, 16, (4, 6), dtype=torch.int64, device=flag_dnn.device
    ).t()
    base = torch.empty((9, 16), dtype=dtype, device=flag_dnn.device).normal_()
    weight = base.t()

    assert not indices.is_contiguous()
    assert not weight.is_contiguous()

    ref_indices = utils.to_reference(indices, ref_kind=None)
    ref_weight = utils.to_reference(weight, ref_kind="compute")
    ref = F.embedding(ref_indices, ref_weight)
    with flag_dnn.use_dnn(include=["embedding"]):
        out = F.embedding(indices, weight)

    utils.gems_assert_close(out, ref, dtype, atol=0, equal_nan=True)


@pytest.mark.embedding
@pytest.mark.parametrize("padding_idx", [0, -1, -4])
def test_embedding_padding_idx_forward_value_is_unchanged(padding_idx):
    indices = torch.tensor(
        [0, 1, 2, 3], dtype=torch.long, device=flag_dnn.device
    )
    weight = torch.arange(
        16, dtype=torch.float32, device=flag_dnn.device
    ).reshape(4, 4)

    ref_indices = utils.to_reference(indices, ref_kind=None)
    ref_weight = utils.to_reference(weight, ref_kind="compute")
    ref = F.embedding(ref_indices, ref_weight, padding_idx=padding_idx)
    with flag_dnn.use_dnn(include=["embedding"]):
        out = F.embedding(indices, weight, padding_idx=padding_idx)

    utils.gems_assert_close(out, ref, torch.float32, atol=0)


@pytest.mark.embedding
@pytest.mark.parametrize("padding_idx", [4, -5])
def test_embedding_invalid_padding_idx_raises(padding_idx):
    indices = torch.tensor([0, 1], dtype=torch.long, device=flag_dnn.device)
    weight = torch.empty((4, 4), dtype=torch.float32, device=flag_dnn.device)

    with pytest.raises(AssertionError, match="Padding_idx"):
        flag_dnn.ops.embedding(indices, weight, padding_idx=padding_idx)


@pytest.mark.embedding
def test_embedding_nan_inf_signed_zero_are_copied():
    indices = torch.tensor([0, 1], dtype=torch.long, device=flag_dnn.device)
    weight = torch.tensor(
        [[0.0, -0.0, float("inf"), float("nan")], [1.0, -2.0, 3.0, -4.0]],
        dtype=torch.float32,
        device=flag_dnn.device,
    )

    ref_indices = utils.to_reference(indices, ref_kind=None)
    ref_weight = utils.to_reference(weight, ref_kind="compute")
    ref = F.embedding(ref_indices, ref_weight)
    with flag_dnn.use_dnn(include=["embedding"]):
        out = F.embedding(indices, weight)

    utils.gems_assert_close(out, ref, torch.float32, atol=0, equal_nan=True)
    assert torch.signbit(out[0, 1]).item() == torch.signbit(ref[0, 1]).item()


@pytest.mark.embedding
def test_embedding_direct_wrapper_rejects_max_norm():
    indices = torch.tensor([0, 1], dtype=torch.long, device=flag_dnn.device)
    weight = torch.empty((4, 4), dtype=torch.float32, device=flag_dnn.device)

    with pytest.raises(NotImplementedError, match="max_norm"):
        flag_dnn.ops.embedding(indices, weight, max_norm=1.0)


@pytest.mark.embedding
def test_embedding_registered_path_rejects_max_norm():
    indices = torch.tensor([0, 1], dtype=torch.long, device=flag_dnn.device)
    weight = torch.empty((4, 4), dtype=torch.float32, device=flag_dnn.device)

    with flag_dnn.use_dnn():
        with pytest.raises(NotImplementedError, match="max_norm"):
            F.embedding(indices, weight, max_norm=1.0)


@pytest.mark.embedding
def test_embedding_rejects_training_backward_path():
    indices = torch.tensor([0, 1], dtype=torch.long, device=flag_dnn.device)
    weight = torch.empty(
        (4, 4), dtype=torch.float32, device=flag_dnn.device
    ).normal_()
    weight.requires_grad_()

    with flag_dnn.use_dnn(include=["embedding"]):
        with pytest.raises(
            NotImplementedError, match="forward inference only"
        ):
            F.embedding(indices, weight)
