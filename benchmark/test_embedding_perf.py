from typing import Generator

import pytest
import torch
import torch.nn.functional as F

import flag_dnn
from benchmark.performance_utils import Benchmark


def torch_embedding(indices, weight):
    return F.embedding(indices, weight)


def gems_embedding_wrapper(indices, weight):
    return flag_dnn.ops.embedding(indices, weight)


class EmbeddingBenchmark(Benchmark):
    MAX_PEAK_BYTES = 6 * 1024**3

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        self.shapes = [
            ((16,), 4096, 16, "random"),
            ((1024,), 32768, 16, "random"),
            ((4096,), 32768, 64, "random"),
            ((8192,), 65536, 127, "random"),
            ((64, 128), 65536, 128, "random"),
            ((32, 128), 65536, 256, "repeat"),
            ((16, 256), 65536, 768, "hotspot"),
            ((8, 1024), 131072, 64, "repeat"),
        ]
        return None

    @staticmethod
    def _numel(shape):
        n = 1
        for dim in shape:
            n *= dim
        return n

    @staticmethod
    def _elem_size(dtype):
        return torch.empty((), dtype=dtype).element_size()

    def _estimate_peak_bytes(self, shape, vocab_size, embedding_dim, dtype):
        n_indices = self._numel(shape)
        elem_size = self._elem_size(dtype)
        return (
            n_indices * 8
            + vocab_size * embedding_dim * elem_size
            + n_indices * embedding_dim * elem_size
        )

    def _make_indices(self, shape, vocab_size, distribution):
        if distribution == "repeat":
            local_vocab = min(vocab_size, 256)
            return torch.randint(
                0, local_vocab, shape, dtype=torch.long, device=self.device
            )
        if distribution == "hotspot":
            indices = torch.zeros(shape, dtype=torch.long, device=self.device)
            flat = indices.reshape(-1)
            if flat.numel() > 1:
                flat[1::8] = torch.randint(
                    0,
                    vocab_size,
                    (flat[1::8].numel(),),
                    dtype=torch.long,
                    device=self.device,
                )
            return indices
        return torch.randint(
            0, vocab_size, shape, dtype=torch.long, device=self.device
        )

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape, vocab_size, embedding_dim, distribution in self.shapes:
            if (
                self._estimate_peak_bytes(
                    shape, vocab_size, embedding_dim, cur_dtype
                )
                > self.MAX_PEAK_BYTES
            ):
                continue

            indices = self._make_indices(shape, vocab_size, distribution)
            weight = torch.empty(
                (vocab_size, embedding_dim),
                dtype=cur_dtype,
                device=self.device,
            ).normal_()

            yield indices, weight

    def get_gbps(self, args, latency):
        indices, weight = args
        n_indices = indices.numel()
        embedding_dim = weight.shape[1]
        elem_size = weight.element_size()
        io_amount = (
            indices.numel() * indices.element_size()
            + n_indices * embedding_dim * elem_size
            + n_indices * embedding_dim * elem_size
        )
        return io_amount / (latency * 1e-3) / 1e9


@pytest.mark.embedding
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_perf_embedding(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    bench = EmbeddingBenchmark(
        op_name="embedding",
        torch_op=torch_embedding,
        gems_op=gems_embedding_wrapper,
        dtypes=[dtype],
    )
    bench.run()
