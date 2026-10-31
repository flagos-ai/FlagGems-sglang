# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Explicit, tensor-free Qwen3.6 workload plans and lazy input construction.

Shapes come from the two model workload documents. Seeds and scaled synthetic
values are test design choices: no original tensor dump was supplied.
"""

import os
import sys
from dataclasses import asdict, dataclass

OPERATORS = (
    "gemma_rmsnorm",
    "silu_and_mul_out",
    "gemma_fused_add_rmsnorm",
    "chunk_gated_delta_rule_fwd",
    "fused_recurrent_gated_delta_rule_packed_decode",
    "select_experts",
    "_fused_moe_kernel_sequence",
)
MODELS = {
    "27b": dict(hidden=5120, q_heads=6, value_heads=12, pool=1381),
    "35b": dict(hidden=2048, q_heads=4, value_heads=8, pool=3193),
}


@dataclass
class TopKConfig:
    # Structural equivalent of SGLang v0.5.11 TopKConfig. No SGLang dependency
    # is needed on the target backend to construct the documented input ABI.
    top_k: int = 8
    use_grouped_topk: bool = False
    topk_group: object = None
    num_expert_group: object = None
    renormalize: bool = True
    num_fused_shared_experts: int = 0
    custom_routing_function: object = None
    correction_bias: object = None
    torch_native: bool = False
    routed_scaling_factor: object = None
    apply_routed_scaling_factor_on_output: bool = False
    fused_shared_experts_scaling_factor: object = None
    output_format: object = None
    scoring_func: str = "softmax"


def workloads(operator, comprehensive=False):
    rows = []
    for model in ("27b", "35b"):
        if (
            operator in {"select_experts", "_fused_moe_kernel_sequence"}
            and model != "35b"
        ):
            continue
        tokens = (20,) if operator == "chunk_gated_delta_rule_fwd" else (20, 1)
        if operator == "fused_recurrent_gated_delta_rule_packed_decode":
            tokens = (1,)
        roles = (
            ("hidden", "q", "k")
            if operator == "gemma_rmsnorm"
            else ("default",)
        )
        for count in tokens:
            for role in roles:
                rows.append(
                    dict(
                        model=model,
                        tokens=count,
                        role=role,
                        seed=0,
                        variant="documented",
                    )
                )
    if comprehensive:
        if operator == "chunk_gated_delta_rule_fwd":
            for count in (63, 64, 65, 129):
                rows.append(
                    dict(
                        model="35b",
                        tokens=count,
                        role="default",
                        seed=1,
                        variant="chunk_boundary",
                        pool=3,
                    )
                )
        elif operator == "fused_recurrent_gated_delta_rule_packed_decode":
            rows.append(
                dict(
                    model="35b",
                    tokens=2,
                    role="default",
                    seed=1,
                    variant="negative_slot",
                    pool=3,
                )
            )
        elif operator == "gemma_rmsnorm":
            rows.append(
                dict(
                    model="35b",
                    tokens=3,
                    role="k",
                    seed=1,
                    variant="out_buffer",
                )
            )
        else:
            rows.append(
                dict(
                    model="35b",
                    tokens=3,
                    role="default",
                    seed=1,
                    variant="additional",
                )
            )
    return rows


def describe(operator, row):
    cfg = MODELS[row["model"]]
    count, hidden, heads = row["tokens"], cfg["hidden"], cfg["value_heads"]
    pool = row.get("pool", cfg["pool"])
    if operator == "silu_and_mul_out":
        width = 8704 if row["model"] == "27b" else 256
        return dict(
            input=(count, width),
            out=(count, width // 2) if row["variant"] != "out_none" else None,
        )
    if operator == "gemma_rmsnorm":
        role = row["role"]
        width = hidden if role == "hidden" else 256
        size = count * cfg["q_heads"] if role == "q" else count
        return dict(
            input=(size, width),
            weight=(width,),
            out=(size, width) if row["variant"] == "out_buffer" else None,
        )
    if operator == "gemma_fused_add_rmsnorm":
        return dict(
            input=(count, hidden), residual=(count, hidden), weight=(hidden,)
        )
    if operator == "chunk_gated_delta_rule_fwd":
        return dict(
            q=(1, count, 4, 128),
            k=(1, count, 4, 128),
            v=(1, count, heads, 128),
            g=(1, count, heads),
            beta=(1, count, heads),
            initial_state=(pool, heads, 128, 128),
            initial_state_indices=(1,),
            cu_seqlens=(2,),
            chunk_indices=((count + 63) // 64, 2),
        )
    if operator == "fused_recurrent_gated_delta_rule_packed_decode":
        return dict(
            mixed_qkv=(count, 1024 + heads * 128),
            a=(count, heads),
            b=(count, heads),
            A_log=(heads,),
            dt_bias=(heads,),
            initial_state=(pool, heads, 128, 128),
            out=(count, 1, heads, 128),
            ssm_state_indices=(count,),
        )
    if operator == "select_experts":
        return dict(hidden_states=(count, 2048), router_logits=(count, 256))
    if operator == "_fused_moe_kernel_sequence":
        # Capacities follow moe_align_block_size. Valid prefix depends on IDs.
        capacity = count * 8 * 16
        return dict(
            hidden_states=(count, 2048),
            w1=(256, 256, 2048),
            w2=(256, 2048, 128),
            topk_weights=(count, 8),
            topk_ids=(count, 8),
            sorted_token_ids=(capacity,),
            expert_ids=(capacity // 16,),
            num_tokens_post_padded=(1,),
        )
    raise ValueError(operator)


def parameters(operator, row):
    """Record literals, layouts and mixed dtypes without building a Tensor."""
    params = dict(row)
    params["source"] = "SGLang v0.5.11; Qwen3.6 TP=4 workload"
    params["input_distribution"] = (
        "seeded FP32 N(0,0.5), cast to BF16; see builder for weights/gates"
    )
    params["input_dtypes"] = {
        name: "torch.bfloat16"
        for name, shape in describe(operator, row).items()
        if shape is not None
    }
    if operator.startswith("gemma"):
        params.update(eps=1e-6, enable_pdl=None, weight_std=0.1)
        if operator == "gemma_rmsnorm" and row["role"] == "k":
            params["input_stride"] = [
                3584 if row["model"] == "27b" else 2560,
                1,
            ]
    if operator == "chunk_gated_delta_rule_fwd":
        params.update(
            scale=128**-0.5,
            normalized_qk=True,
            l2_eps=1e-6,
            chunk_size=64,
            suppression_level=0,
            initial_state_indices=[1],
            cu_seqlens=[0, row["tokens"]],
            state_fill=0.03125,
            active_state_std=0.1,
        )
        for name in ("g", "beta", "initial_state"):
            params["input_dtypes"][name] = "torch.float32"
        for name in ("initial_state_indices", "cu_seqlens", "chunk_indices"):
            params["input_dtypes"][name] = "torch.int32"
    if operator == "fused_recurrent_gated_delta_rule_packed_decode":
        params.update(
            scale=128**-0.5,
            use_qk_l2norm_in_kernel=True,
            ssm_state_indices=(
                [-1, 1] if row["variant"] == "negative_slot" else [1]
            ),
            state_fill=0.03125,
            active_state_std=0.1,
        )
        for name in ("A_log", "initial_state"):
            params["input_dtypes"][name] = "torch.float32"
        params["input_dtypes"]["ssm_state_indices"] = "torch.int32"
    if operator == "select_experts":
        params.update(
            topk_config=asdict(TopKConfig()),
            layer_id=0,
            num_token_non_padded=None,
            expert_location_dispatch_info=None,
            logits_range=[-8.0, 7.9375],
        )
    if operator == "_fused_moe_kernel_sequence":
        params.update(
            num_experts=256,
            intermediate_size=128,
            top_k=8,
            config=dict(
                BLOCK_SIZE_M=16,
                BLOCK_SIZE_N=32,
                BLOCK_SIZE_K=64,
                GROUP_SIZE_M=1,
            ),
            down_config=None,
            down_moe_use_tma=False,
            activation="silu",
            is_gated=True,
            inplace=True,
            no_combine=False,
            apply_router_weight_on_input=False,
            routed_scaling_factor=None,
            filter_expert=False,
            quantized=False,
        )
        for name in (
            "topk_ids",
            "sorted_token_ids",
            "expert_ids",
            "num_tokens_post_padded",
        ):
            params["input_dtypes"][name] = "torch.int32"
        params["input_dtypes"]["topk_weights"] = "torch.float32"
    return params


def build_inputs(operator, row, dtype, device):
    import torch

    shape = describe(operator, row)
    generator = (
        None
        if str(device) == "meta"
        else torch.Generator(device=device).manual_seed(row["seed"])
    )

    def normal(size, scale=0.5, target_dtype=dtype):
        return (
            torch.randn(
                size, dtype=torch.float32, device=device, generator=generator
            )
            * scale
        ).to(target_dtype)

    def state_pool(size):
        state = torch.full(size, 0.03125, device=device, dtype=torch.float32)
        # A non-symmetric active slot exposes wrong-slot reads and V/K swaps.
        state[1].copy_(normal(size[1:], 0.1, torch.float32))
        return state

    def routing(count):
        # Exactly representable, separated BF16 scores avoid an unspecified
        # tie-breaking contract while making expert order vary across rows.
        values = (
            torch.arange(-128, 128, device=device, dtype=torch.float32) / 16
        )
        vendor = os.environ.get("DNN_VENDOR") or getattr(
            sys.modules.get("flaggems_sglang"), "vendor_name", None
        )
        # The handoff documents unsupported device randperm on Kunlunxin.
        # Only fixture index generation moves to CPU; reference math stays
        # on the vendor device and fixed KS3 replay never regenerates inputs.
        permutation_device = (
            "cpu"
            if vendor == "kunlunxin" and str(device) != "meta"
            else device
        )
        permutation_generator = (
            torch.Generator(device="cpu").manual_seed(row["seed"])
            if permutation_device == "cpu" and vendor == "kunlunxin"
            else generator
        )
        return torch.stack(
            [
                values[
                    torch.randperm(
                        256,
                        device=permutation_device,
                        generator=permutation_generator,
                    ).to(device)
                ]
                for _ in range(count)
            ]
        ).to(dtype)

    if operator == "silu_and_mul_out":
        return dict(
            input=normal(shape["input"]),
            out=(
                torch.empty(shape["out"], dtype=dtype, device=device)
                if shape["out"]
                else None
            ),
        )
    if operator == "gemma_rmsnorm":
        value = normal(shape["input"])
        if row["role"] == "k":
            stride = 3584 if row["model"] == "27b" else 2560
            backing = torch.full(
                (row["tokens"], stride), -7.0, device=device, dtype=dtype
            )
            offset = stride - 512
            backing[:, offset : offset + 256].copy_(value)
            value = backing[:, offset : offset + 256]
        return dict(
            input=value,
            weight=normal(shape["weight"], 0.1),
            eps=1e-6,
            out=torch.empty_like(value) if shape["out"] else None,
            enable_pdl=None,
        )
    if operator == "gemma_fused_add_rmsnorm":
        return dict(
            input=normal(shape["input"]),
            residual=normal(shape["residual"]),
            weight=normal(shape["weight"], 0.1),
            eps=1e-6,
            enable_pdl=None,
        )
    if operator == "chunk_gated_delta_rule_fwd":

        def normalized(size):
            x = normal(size).float()
            return (
                x * torch.rsqrt(x.square().sum(-1, keepdim=True) + 1e-6)
            ).to(dtype)

        count = row["tokens"]
        return dict(
            q=normalized(shape["q"]),
            k=normalized(shape["k"]),
            v=normal(shape["v"]),
            g=-normal(shape["g"], 0.1, torch.float32).abs() - 0.01,
            beta=normal(shape["beta"], 1.0, torch.float32).sigmoid(),
            scale=128**-0.5,
            initial_state=state_pool(shape["initial_state"]),
            initial_state_indices=torch.tensor(
                [1], dtype=torch.int32, device=device
            ),
            cu_seqlens=torch.tensor(
                [0, count], dtype=torch.int32, device=device
            ),
            chunk_indices=torch.tensor(
                [[0, i] for i in range((count + 63) // 64)],
                dtype=torch.int32,
                device=device,
            ),
        )
    if operator == "fused_recurrent_gated_delta_rule_packed_decode":
        count = row["tokens"]
        slots = [-1, 1] if row["variant"] == "negative_slot" else [1] * count
        return dict(
            mixed_qkv=normal(shape["mixed_qkv"]),
            a=normal(shape["a"]),
            b=normal(shape["b"]),
            A_log=normal(shape["A_log"], 0.1, torch.float32),
            dt_bias=normal(shape["dt_bias"], 0.1),
            scale=128**-0.5,
            initial_state=state_pool(shape["initial_state"]),
            out=torch.full(
                shape["out"], float("nan"), device=device, dtype=dtype
            ),
            ssm_state_indices=torch.tensor(
                slots, dtype=torch.int32, device=device
            ),
            use_qk_l2norm_in_kernel=True,
        )
    if operator == "select_experts":
        return dict(
            hidden_states=normal(shape["hidden_states"]),
            router_logits=routing(row["tokens"]),
            topk_config=TopKConfig(),
            layer_id=0,
            num_token_non_padded=None,
            expert_location_dispatch_info=None,
        )
    if operator == "_fused_moe_kernel_sequence":
        hidden = normal(shape["hidden_states"])
        ids_tensor = (
            torch.arange(
                row["tokens"] * 8, device=device, dtype=torch.int32
            ).view(-1, 8)
            % 106
        )
        weights = (
            (torch.arange(8, device=device, dtype=torch.float32) / 16)
            .softmax(-1)
            .expand(row["tokens"], -1)
            .contiguous()
        )
        # A host-only deterministic alignment oracle, outside the timed ABI.
        ids = [i % 106 for i in range(row["tokens"] * 8)]
        sorted_ids, experts = [], []
        for expert in range(256):
            matches = [i for i, item in enumerate(ids) if item == expert]
            padding = (-len(matches)) % 16
            sorted_ids.extend(matches + [len(ids)] * padding)
            experts.extend([expert] * ((len(matches) + padding) // 16))
        valid = len(sorted_ids)
        sorted_ids += [len(ids)] * (shape["sorted_token_ids"][0] - valid)
        experts += [-1] * (shape["expert_ids"][0] - len(experts))
        return dict(
            hidden_states=hidden,
            w1=normal(shape["w1"], 2048**-0.5),
            w2=normal(shape["w2"], 128**-0.5),
            topk_weights=weights,
            topk_ids=ids_tensor,
            sorted_token_ids=torch.tensor(
                sorted_ids, device=device, dtype=torch.int32
            ),
            expert_ids=torch.tensor(experts, device=device, dtype=torch.int32),
            num_tokens_post_padded=torch.tensor(
                [valid], device=device, dtype=torch.int32
            ),
            config=dict(
                BLOCK_SIZE_M=16,
                BLOCK_SIZE_N=32,
                BLOCK_SIZE_K=64,
                GROUP_SIZE_M=1,
            ),
            down_config=None,
            down_moe_use_tma=False,
            b1=None,
            b2=None,
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=False,
            w1_scale=None,
            w2_scale=None,
            w1_zp=None,
            w2_zp=None,
            a1_scale=None,
            a2_scale=None,
            block_shape=None,
            activation="silu",
            is_gated=True,
            no_combine=False,
            inplace=True,
            apply_router_weight_on_input=False,
            routed_scaling_factor=None,
            gemm1_alpha=None,
            gemm1_limit=None,
            filter_expert=False,
            hooks=None,
        )
    raise ValueError(operator)


def baseline_key(operator, inputs):
    """Portable exact geometry/ABI key; callable outside the measured region."""
    import dataclasses
    import hashlib
    import json

    import torch

    def encode(value):
        if isinstance(value, torch.Tensor):
            return dict(
                shape=list(value.shape),
                stride=list(value.stride()),
                dtype=str(value.dtype),
                offset=value.storage_offset(),
            )
        if dataclasses.is_dataclass(value):
            return encode(dataclasses.asdict(value))
        if isinstance(value, dict):
            return {k: encode(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [encode(v) for v in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise TypeError(
            f"Unsupported baseline identity: {type(value).__name__}"
        )

    payload = json.dumps(
        dict(
            operator=operator,
            inputs=encode(inputs),
            timing=timing_contract(operator) or "median",
        ),
        sort_keys=True,
        separators=(",", ":"),
    )
    # The operator is already part of the hashed payload and JSON group.
    return (
        "qwen36-v2-" if timing_contract(operator) else "qwen36-v1-"
    ) + hashlib.sha256(payload.encode()).hexdigest()


def timing_contract(operator):
    import json
    from pathlib import Path

    spec = json.loads(
        Path(__file__).with_name("reset_timing.json").read_text()
    )
    return spec["timing"] if operator in spec["read_write"] else None
