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

"""Native SGLang v0.5.21 collection workload for chain_speculative_sampling."""

import torch

OP_NAME = "chain_speculative_sampling"
DTYPES = [torch.float32]
IS_INPLACE = True
MUTATED_INPUTS = (0, 1, 2)

SHAPES = [(b, 4, 32000) for b in (1, 8, 64, 512, 4096)]
MORE_SHAPES = [(4, 5, 32), (8, 8, 128), (2, 3, 16)]


def native():
    try:
        from sglang.kernels.ops.speculative.reject_sampling import (
            chain_speculative_sampling_triton as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op


def binding_for_case(shape):
    return dict(zip(("B", "S", "V"), shape))


def grid():
    return [binding_for_case(shape) for shape in (*SHAPES, *MORE_SHAPES)]


def build_case(shape, cur_dtype, device):
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
    return (
        candidates,
        retrive_index,
        uniform_samples,
        uniform_samples_for_final_sampling,
        target_probs,
        draft_probs,
        B * S,
    )


def build_inputs(binding, dtype, device):
    shape = (binding["B"], binding["S"], binding["V"])
    candidates, retrieve, samples, final_samples, target, draft, slots = (
        build_case(shape, dtype, device)
    )
    batch = candidates.shape[0]
    predicts = candidates.new_zeros(slots)
    accept_index = torch.full_like(retrieve, -1)
    accepted = candidates.new_zeros(batch)
    return (
        predicts,
        accept_index,
        accepted,
        candidates,
        retrieve,
        None,
        None,
        samples,
        final_samples,
        target,
        draft,
        0.0,
        0.0,
        False,
    ), {}


def key_shape(binding):
    return [binding["B"], binding["S"], binding["V"]]


def config(binding, dtype):
    B, S, V = binding["B"], binding["S"], binding["V"]

    def tensor(shape, tensor_dtype=str(dtype)):
        return {"shape": shape, "dtype": tensor_dtype}

    return {
        "inputs": {
            "predicts": tensor([B * S], "torch.int32"),
            "accept_index": tensor([B, S], "torch.int64"),
            "accept_token_num": tensor([B], "torch.int32"),
            "candidates": tensor([B, S], "torch.int32"),
            "retrive_index": tensor([B, S], "torch.int64"),
            "uniform_samples": tensor([B, S - 1]),
            "uniform_samples_for_final_sampling": tensor([B]),
            "target_probs": tensor([B, S, V]),
            "draft_probs": tensor([B, S - 1, V]),
        },
        "outputs": {
            "predicts": tensor([B * S], "torch.int32"),
            "accept_index": tensor([B, S], "torch.int64"),
            "accept_token_num": tensor([B], "torch.int32"),
        },
        "dims": dict(binding),
    }
