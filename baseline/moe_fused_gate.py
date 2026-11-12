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

"""Native SGLang collection workload for moe_fused_gate."""

import torch

OP_NAME = "moe_fused_gate"
DTYPES = [torch.bfloat16]
IS_INPLACE = False


def _shape(
    m,
    n,
    topk,
    scoring_func="sigmoid",
    num_fused_shared_experts=0,
    renormalize=True,
    routed_scaling_factor=1.0,
    apply_routed_scaling_factor_on_output=False,
    moe_softcapping=0.0,
    num_expert_group=1,
    topk_group=1,
):
    return (
        m,
        n,
        topk,
        scoring_func,
        num_fused_shared_experts,
        renormalize,
        routed_scaling_factor,
        apply_routed_scaling_factor_on_output,
        moe_softcapping,
        num_expert_group,
        topk_group,
    )


SHAPES = [
    _shape(
        m, 256, 8, num_fused_shared_experts=1, num_expert_group=8, topk_group=4
    )
    for m in (1, 8, 64, 512, 4096)
]
MORE_SHAPES = [
    _shape(1, 8, 2),
    _shape(
        37,
        256,
        8,
        num_fused_shared_experts=1,
        routed_scaling_factor=2.5,
        num_expert_group=8,
        topk_group=4,
    ),
    _shape(83, 64, 6, scoring_func="sqrtsoftplus"),
    _shape(
        17,
        32,
        4,
        scoring_func="softmax",
        moe_softcapping=30.0,
        renormalize=False,
    ),
    _shape(
        9,
        128,
        6,
        apply_routed_scaling_factor_on_output=True,
        routed_scaling_factor=1.7,
    ),
]


def native():
    """解析 sglang.kernels.ops.moe.moe_fused_gate.moe_fused_gate。"""
    try:
        from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate as op
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [
        {
            "m": m,
            "n": n,
            "topk": topk,
            "scoring_func": scoring_func,
            "num_fused_shared_experts": num_fused_shared_experts,
            "renormalize": renormalize,
            "routed_scaling_factor": routed_scaling_factor,
            "apply_routed_scaling_factor_on_output": apply_routed_scaling_factor_on_output,
            "moe_softcapping": moe_softcapping,
            "num_expert_group": num_expert_group,
            "topk_group": topk_group,
        }
        for (
            m,
            n,
            topk,
            scoring_func,
            num_fused_shared_experts,
            renormalize,
            routed_scaling_factor,
            apply_routed_scaling_factor_on_output,
            moe_softcapping,
            num_expert_group,
            topk_group,
        ) in (*SHAPES, *MORE_SHAPES)
    ]


def build_case(shape, cur_dtype, device):
    (
        m,
        n,
        topk,
        scoring_func,
        num_fused_shared_experts,
        renormalize,
        routed_scaling_factor,
        apply_routed_scaling_factor_on_output,
        moe_softcapping,
        num_expert_group,
        topk_group,
    ) = shape
    g = torch.Generator(device=device).manual_seed(0)
    scores = torch.randn(
        m, n, generator=g, device=device, dtype=torch.float32
    ).to(cur_dtype)
    # The bias stays fp32; the reference upcasts both operands anyway.
    bias = 0.1 * torch.randn(
        n, generator=g, device=device, dtype=torch.float32
    )
    # ``scoring_func`` is a str, which unpack_to_args_kwargs would drop from
    # the positional args, so route it (and the rest of the gate config)
    # through kwargs.
    return (
        scores,
        bias,
        topk,
        dict(
            scoring_func=scoring_func,
            num_fused_shared_experts=num_fused_shared_experts,
            renormalize=renormalize,
            routed_scaling_factor=routed_scaling_factor,
            apply_routed_scaling_factor_on_output=(
                apply_routed_scaling_factor_on_output
            ),
            moe_softcapping=moe_softcapping,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
        ),
    )


def build_inputs(binding, dtype, device):
    shape = (
        binding["m"],
        binding["n"],
        binding["topk"],
        binding["scoring_func"],
        binding["num_fused_shared_experts"],
        binding["renormalize"],
        binding["routed_scaling_factor"],
        binding["apply_routed_scaling_factor_on_output"],
        binding["moe_softcapping"],
        binding["num_expert_group"],
        binding["topk_group"],
    )
    *args, kwargs = build_case(shape, dtype, device)
    return tuple(args), kwargs


def key_shape(binding):
    return "case_" + "_".join(
        str(binding[name])
        for name in (
            "m",
            "n",
            "topk",
            "scoring_func",
            "num_fused_shared_experts",
            "renormalize",
            "routed_scaling_factor",
            "apply_routed_scaling_factor_on_output",
            "moe_softcapping",
            "num_expert_group",
            "topk_group",
        )
    )


def config(binding, dtype):
    m, n, topk, nfse, neg, tkg = (
        binding["m"],
        binding["n"],
        binding["topk"],
        binding["num_fused_shared_experts"],
        binding["num_expert_group"],
        binding["topk_group"],
    )
    dt = str(dtype)
    # kernel 输出每 token top-(topk+num_fused_shared_experts) 的权重与索引。
    out_k = topk
    return {
        "inputs": {
            "scores": {"shape": [m, n], "dtype": dt},
            "bias": {"shape": [n], "dtype": "torch.float32"},
            "topk": {"scalar": topk},
            "scoring_func": {"scalar": binding["scoring_func"]},
            "num_fused_shared_experts": {"scalar": nfse},
            "renormalize": {"scalar": binding["renormalize"]},
            "routed_scaling_factor": {
                "scalar": binding["routed_scaling_factor"]
            },
            "apply_routed_scaling_factor_on_output": {
                "scalar": binding["apply_routed_scaling_factor_on_output"]
            },
            "moe_softcapping": {"scalar": binding["moe_softcapping"]},
            "num_expert_group": {"scalar": neg},
            "topk_group": {"scalar": tkg},
        },
        "outputs": {
            "topk_weights": {"shape": [m, out_k], "dtype": "torch.float32"},
            "topk_ids": {"shape": [m, out_k], "dtype": "torch.int32"},
        },
        "dims": {
            "m": m,
            "n": n,
            "topk": topk,
            "num_fused_shared_experts": nfse,
            "num_expert_group": neg,
            "topk_group": tkg,
        },
    }
