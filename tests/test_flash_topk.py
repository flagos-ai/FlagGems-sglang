# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import pytest
import torch
import triton

import flaggems_sglang
from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference.flash_topk import score_row


@pytest.fixture
def device():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (
        9,
        0,
    ):
        pytest.skip("FlashTopK is a Hopper arch operator")
    cuda = torch.device("cuda", torch.cuda.current_device())
    if not flaggems_sglang.supports_flash_topk(cuda):
        pytest.skip("Hopper FlashTopK requires FlagTree TLE WGMMA")
    assert flaggems_sglang.get_op("flash_topk") is flaggems_sglang.flash_topk
    assert ".hopper." in flaggems_sglang.flash_topk.__module__
    return cuda


def _inputs(device, requests, lengths, context, *, exact=False):
    rows = len(lengths)
    pages = triton.cdiv(context, 64)
    total_pages = 2 * pages
    rng = torch.Generator(device=device).manual_seed(129)
    q = torch.empty((rows + 1, 64, 128), device=device).normal_(generator=rng)
    q = q.to(torch.float8_e4m3fn)[1:]
    gates = torch.empty((rows, 64, 1), device=device).normal_(generator=rng)
    gates.mul_(1 / (64 * 128) ** 0.5)
    cache = torch.empty(
        (total_pages + 1, 8448), dtype=torch.uint8, device=device
    )[1:]
    keys = torch.empty((total_pages, 8192), device=device).normal_(
        generator=rng
    )
    cache[:, :8192].copy_(keys.to(torch.float8_e4m3fn).view(torch.uint8))
    cache[:, 8192:].view(torch.float32).uniform_(0.2, 1.0, generator=rng)
    if exact:
        q.view(torch.uint8).fill_(0x38)
        gates.fill_(1)
        cache[:, :8192].fill_(0x38)
    table = torch.randperm(total_pages, device=device, generator=rng)
    table = table.to(torch.int32).reshape(2, pages)
    expanded_lengths = torch.tensor(lengths, device=device, dtype=torch.int32)
    row_to_request = torch.tensor(requests, device=device, dtype=torch.int32)
    out_storage = torch.full(
        (rows, 2056), -313, device=device, dtype=torch.int32
    )
    out = out_storage[:, 4:-4]
    size = flaggems_sglang.flash_topk_workspace_size(rows, context)
    work = (
        torch.empty(size + 2, device=device, dtype=torch.int32)[2:]
        if size
        else None
    )
    return (
        q,
        gates,
        cache,
        table,
        row_to_request,
        expanded_lengths,
        context,
        out,
        work,
    ), out_storage


def _local_ids(result, length, request, page_table, offset):
    ids = result[: min(length, 2048)].long()
    assert torch.all(ids >= 0)
    assert torch.all(result[min(length, 2048) :] == -1)
    if page_table is not None:
        physical = ids // 64
        matches = physical[:, None] == page_table[request].long()[None, :]
        assert torch.all(matches.any(dim=1))
        ids = matches.int().argmax(dim=1) * 64 + ids % 64
    elif offset is not None:
        ids -= offset
    assert torch.all(ids < length)
    assert ids.unique().numel() == ids.numel()
    return ids


def _check_output(args, *, page_table=None, offsets=None, exact=False):
    q, gates, cache, table, requests, lengths, context, _, work = args
    cpu = [tensor.cpu() for tensor in args[:6]]
    q_cpu, w_cpu, cache_cpu, table_cpu, req_cpu, lens_cpu = cpu
    values = args[7].cpu()
    page_cpu = None if page_table is None else page_table.cpu()
    off_cpu = None if offsets is None else offsets.cpu()
    ref = get_reference("flash_topk")
    expected = ref(
        *cpu,
        context,
        torch.empty_like(values),
        output_page_table=page_cpu,
        output_offsets=off_cpu,
    )
    own_scores = None
    if work is not None:
        pitch = triton.cdiv(context, 256) * 256
        own_scores = work[: len(lengths) * pitch].view(torch.float32)
        own_scores = own_scores.view(len(lengths), pitch).cpu()
    for row, length in enumerate(lens_cpu.tolist()):
        request = int(req_cpu[row])
        offset = None if off_cpu is None else int(off_cpu[row])
        ids = _local_ids(values[row], length, request, page_cpu, offset)
        if length <= 2048:
            assert torch.equal(
                values[row].sort().values, expected[row].sort().values
            )
        else:
            scores = own_scores[row, :length]
            cutoff = scores.topk(2048).values.min()
            assert torch.all(scores[ids] >= cutoff)
            assert (scores[ids] > cutoff).sum() == (scores > cutoff).sum()
            ref_scores = score_row(
                q_cpu[row],
                w_cpu[row],
                cache_cpu,
                table_cpu[request],
                length,
            )
            # SM90 FP8 WGMMA and CPU FP32 reduction round differently.
            # The 1e-3 absolute budget applies to these normalized random
            # fixtures. Dyadic products below are checked bit for bit.
            torch.testing.assert_close(
                scores, ref_scores, atol=0.0 if exact else 1e-3, rtol=0.0
            )
            ref_cutoff = ref_scores.topk(2048).values.min()
            # If each score shifts by at most e, a correctly ranked key
            # can be at most 2e below the independent cutoff.
            error = (scores - ref_scores).abs().max()
            assert ref_cutoff - ref_scores[ids].min() <= 2 * error


@pytest.mark.parametrize("mode", ["local", "paged", "ragged"])
@pytest.mark.parametrize("context", [2048, 4097, 32771])
@pytest.mark.flash_topk
def test_flash_topk(device, mode, context):
    requests = [0, 0, 0, 1, 1, -1, 1]
    lengths = [
        min(2047, context),
        min(2048, context),
        context - 1,
        context,
        1,
        0,
        64,
    ]
    args, storage = _inputs(device, requests, lengths, context)
    table = args[3].roll(1, dims=1) if mode == "paged" else None
    offsets = args[4].clamp_min(0) * context if mode == "ragged" else None
    got = flaggems_sglang.flash_topk(
        *args,
        prefill=mode != "local",
        output_page_table=table,
        output_offsets=offsets,
    )
    assert got is args[7]
    assert torch.all(storage[:, :4] == -313)
    assert torch.all(storage[:, -4:] == -313)
    _check_output(args, page_table=table, offsets=offsets)


@pytest.mark.flash_topk
def test_long_selector_exact_repair_and_ties(device):
    context = 16384
    args, _ = _inputs(device, [0, 1], [context, context], context, exact=True)
    table = args[3]
    scales = args[2][:, 8192:].view(torch.float32)
    scales.fill_(1)
    # All first request scores tie. For the second request, half the sampled
    # positions are above all unsampled scores, so its sampled band cannot
    # supply 2048 candidates and the exact four-byte walk must repair it.
    key_ids = torch.arange(context, device=device)
    p = table[1, key_ids // 64].long()
    scales[p, key_ids % 64] = torch.where(
        key_ids % 16 == 4,
        2.0,
        1.0 - key_ids.float() / (2 * context),
    )
    flaggems_sglang.flash_topk(*args, prefill=True, output_page_table=table)
    _check_output(args, page_table=table, exact=True)


@pytest.mark.parametrize("prefill", [False, True])
@pytest.mark.flash_topk
def test_flash_topk_cuda_graph_recomputes_metadata(device, prefill):
    context = 16390
    args, _ = _inputs(device, [0, 0, 1], [context, context - 13, 30], context)

    def run():
        return flaggems_sglang.flash_topk(
            *args, prefill=prefill, output_page_table=args[3]
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for offset in (25, 2049):
        args[3].copy_(args[3].roll(1, dims=1))
        args[4].copy_(
            torch.tensor([1, 0, -1], dtype=torch.int32, device=device)
        )
        args[5].copy_(
            torch.tensor(
                [offset, context - 4, 0], dtype=torch.int32, device=device
            )
        )
        args[1].neg_()
        args[7].fill_(-314)
        graph.replay()
        _check_output(args, page_table=args[3])


@pytest.mark.flash_topk
def test_flash_topk_rejects_invalid_contract_before_write(device):
    args, _ = _inputs(device, [0, 1], [4096, 4097], 4097)
    with pytest.raises(ValueError, match="workspace"):
        flaggems_sglang.flash_topk(*args[:-1])
    with pytest.raises(ValueError, match="either paged or ragged"):
        flaggems_sglang.flash_topk(
            *args, output_page_table=args[3], output_offsets=args[5]
        )
    assert torch.all(args[7] == -313)


@pytest.mark.flash_topk
def test_flash_topk_empty(device):
    args, _ = _inputs(device, [], [], 0)
    assert flaggems_sglang.flash_topk(*args).shape == (0, 2048)


@pytest.mark.flash_topk
def test_invalid_device_metadata_cannot_access_out_of_bounds(device):
    context = 4097
    args, _ = _inputs(
        device, [0, 2, 1, 1], [context + 1, context, -1, context], context
    )
    # The active fourth row's two invalid K page entries must never be
    # used to form a TMA address or scales pointer.
    args[3][1, :2] = torch.tensor([-1, len(args[2])], device=device)
    output_table = torch.arange(
        args[3].numel(), device=device, dtype=torch.int32
    )
    output_table = output_table.view_as(args[3])
    flaggems_sglang.flash_topk(*args, output_page_table=output_table)
    assert torch.all(args[7][:3] == -1)
    first_valid = (int(output_table[1, 0]) + 2) * 64
    assert torch.all(args[7][3] >= first_valid)
    assert len(args[7][3].unique()) == 2048
