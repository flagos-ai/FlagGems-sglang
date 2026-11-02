# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import torch
from mrotary import load

workloads = load("mrotary_embedding_cases")
CORRECTNESS_CASES = [
    workloads.build_inputs(row, torch.bfloat16, "cuda")
    for row in workloads.WORKLOADS
]
routing = workloads.build_inputs(
    workloads.WORKLOADS[4], torch.bfloat16, "cuda"
)
routing["positions"].add_(torch.arange(3, device="cuda")[:, None] * 7)
CORRECTNESS_CASES.append(routing)
