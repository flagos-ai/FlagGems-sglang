# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import torch

# Only model inputs explicitly recorded by the two workload documents.
WORKLOADS = [
    dict(model=model, tokens=tokens, q_heads=heads, position_stride=tokens)
    for model, heads in (("27b", 6), ("35b", 4))
    for tokens in (20, 1)
] + [
    dict(model="35b", tokens=tokens, q_heads=4, position_stride=512)
    for tokens in (20, 1)
]


def describe_inputs(case):
    return dict(
        q=(case["tokens"], case["q_heads"] * 256),
        k=(case["tokens"], 256),
        cos_sin_cache=(262400, 64),
        positions=(3, case["tokens"]),
    )


def case_parameters(case):
    return dict(
        case,
        mrope_section=[11, 11, 10],
        head_size=256,
        rotary_dim=64,
        mrope_interleaved=True,
        mrope_interleaved_glm=False,
        is_neox_style=True,
        axis_map=None,
        positions_dtype="torch.int64",
        positions_stride=[case["position_stride"], 1],
        seed=0,
    )


def build_inputs(case, dtype, device):
    shape = describe_inputs(case)
    generator = (
        None
        if torch.device(device).type == "meta"
        else torch.Generator(device=device).manual_seed(0)
    )

    def normal(size, scale):
        return (
            torch.randn(
                size, device=device, dtype=torch.float32, generator=generator
            )
            * scale
        )

    angles = normal((262400, 32), 1.0)
    cache = torch.cat((angles.cos(), angles.sin()), -1).to(dtype)
    count = case["tokens"]
    position = torch.arange(count, device=device, dtype=torch.int64)
    if count == 1:
        position += 20
    backing = torch.zeros(
        (3, case["position_stride"]), device=device, dtype=torch.int64
    )
    positions = backing[:, :count]
    positions.copy_(position.unsqueeze(0).expand(3, -1))
    return dict(
        q=normal(shape["q"], 0.5).to(dtype),
        k=normal(shape["k"], 0.5).to(dtype),
        cos_sin_cache=cache,
        positions=positions,
        mrope_section=[11, 11, 10],
        head_size=256,
        rotary_dim=64,
        mrope_interleaved=True,
        mrope_interleaved_glm=False,
        is_neox_style=True,
        axis_map=None,
    )


def baseline_key(inputs):
    """Exact, versioned identity shared with the H800 collection recipe."""
    identity = {}
    for name, value in inputs.items():
        if isinstance(value, torch.Tensor):
            identity[name] = dict(
                shape=list(value.shape),
                stride=list(value.stride()),
                dtype=str(value.dtype),
                offset=value.storage_offset(),
            )
        else:
            identity[name] = value
    from pathlib import Path

    spec = json.loads(
        (Path(__file__).parents[1] / "testing/reset_timing.json").read_text()
    )
    identity["timing"] = spec["timing"]
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    # v3 restores original read-write inputs outside each timed call.
    return "mrotary-v3-" + hashlib.sha256(encoded.encode()).hexdigest()
