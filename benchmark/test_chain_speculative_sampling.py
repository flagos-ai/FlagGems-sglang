# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmark for sampling_grammar/chain_speculative_sampling."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

# Shapes match
# kernel-comp-baseline/problems/sampling_grammar/chain_speculative_sampling.
SHAPES = [(b, 4, 32000) for b in (1, 8, 64, 512, 4096)]
MORE_SHAPES = [(4, 5, 32), (8, 8, 128), (2, 3, 16)]


def _input_fn(shape, cur_dtype, device):
    B, S, V = shape
    g = torch.Generator(device=device).manual_seed(0)

    candidates = torch.randint(
        0, V, (B, S), dtype=torch.int32, device=device, generator=g
    )
    retrive_index = torch.arange(
        B * S, dtype=torch.int64, device=device
    ).reshape(B, S)
    uniform_samples = torch.rand(B, S - 1, generator=g, device=device)
    uniform_samples_for_final_sampling = torch.rand(
        B, generator=g, device=device
    )
    target_logits = torch.randn(B, S, V, generator=g, device=device)
    target_probs = torch.softmax(target_logits, dim=-1)
    draft_logits = torch.randn(B, S - 1, V, generator=g, device=device)
    draft_probs = torch.softmax(draft_logits, dim=-1)

    # num_slots is precomputed rather than derived via .item() inside the
    # timed function: retrive_index = arange(B*S) makes it exact, and a host
    # sync inside the measured region would perturb the timing.
    yield (
        candidates,
        retrive_index,
        uniform_samples,
        uniform_samples_for_final_sampling,
        target_probs,
        draft_probs,
        B * S,
    )


@pytest.mark.chain_speculative_sampling
def test_perf_chain_speculative_sampling():
    bench = OpBenchmark(
        op_name="chain_speculative_sampling",
        torch_op=get_reference("chain_speculative_sampling"),
        input_fn=_input_fn,
        dtypes=[torch.float32],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="B, S, V",
    )
    bench.set_gems(flaggems_sglang.chain_speculative_sampling)
    bench.run()
