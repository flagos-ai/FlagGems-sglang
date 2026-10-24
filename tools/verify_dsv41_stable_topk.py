# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Independent stable-reference checks of the bounded tie policy and layouts."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(4)
    from sglang.kernels.ops.attention.dsv4.topk import topk_transform_ragged_v2, topk_transform_paged_v2, plan_topk_v2
    records = []
    cases = []
    for count in [1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 1024, 2048]:
        values = torch.cat([torch.full((505,), 2.), torch.full((count,), .5), torch.full((32,), -4.)])
        cases.append(('boundary-pool-' + str(count), values, 512))
    generator = torch.Generator().manual_seed(858183673)
    for n in [0, 1, 511, 512, 513, 8191, 8192, 8193, 16383, 16384, 16385]:
        cases.append(('length-' + str(n), torch.rand(n, generator=generator) * 8 - 4, 512))
    for k in [1, 127, 128, 129, 511, 2048]:
        cases.append(('K-' + str(k), torch.cat([torch.ones(k - 1), torch.full((129,), .25), torch.full((16,), -8.)]), k))
    def canonical(x):
        return x.masked_fill(x < 0, torch.iinfo(torch.int32).max).sort(-1).values
    for name, values, k in cases:
        n = len(values)
        width = ((n + 7) // 4) * 4
        original = torch.full((2, max(width, 4)), -float('inf'))
        starts = torch.tensor([0, 3], dtype=torch.int32)
        lengths = torch.full((2,), n, dtype=torch.int32)
        offsets = torch.tensor([100, 300], dtype=torch.int32)
        expected = torch.full((2, k), -1, dtype=torch.int32)
        order = torch.argsort(values, descending=True, stable=True)[:k]
        for row in range(2):
            original[row, int(starts[row]):int(starts[row]) + n] = values
            expected[row, :len(order)] = order.int() + offsets[row]
        scores, lens, row_starts, out_offsets = [x.cuda() for x in [original, lengths, starts, offsets]]
        output = torch.empty((2, k), dtype=torch.int32, device='cuda')
        for _ in range(10):
            scores.copy_(original)
            topk_transform_ragged_v2(scores, lens, row_starts=row_starts, out_offsets=out_offsets, out_indices=output)
            assert torch.equal(canonical(output.cpu()), canonical(expected)), name
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            topk_transform_ragged_v2(scores, lens, row_starts=row_starts, out_offsets=out_offsets, out_indices=output)
        scores.copy_(original); graph.replay()
        assert torch.equal(canonical(output.cpu()), canonical(expected)), name + '-graph'
        changed_values = values.flip(0)
        changed_starts = torch.tensor([1, 2], dtype=torch.int32)
        changed_offsets = offsets + 5
        changed_lengths = torch.tensor([max(0, n - 3), n], dtype=torch.int32)
        changed_scores = torch.full_like(original, -float('inf'))
        changed_expected = torch.full_like(expected, -1)
        for row in range(2):
            length = int(changed_lengths[row]); begin = int(changed_starts[row])
            changed_scores[row, begin:begin + length] = changed_values[:length]
            order = torch.argsort(changed_values[:length], descending=True, stable=True)[:k]
            changed_expected[row, :len(order)] = order.int() + changed_offsets[row]
        scores.copy_(changed_scores);lens.copy_(changed_lengths);row_starts.copy_(changed_starts);out_offsets.copy_(changed_offsets)
        graph.replay()
        assert torch.equal(canonical(output.cpu()), canonical(changed_expected)), name + '-changed-input-graph'
        records.append({'id': name, 'N': n, 'K': k, 'eager_repeats': 10, 'ragged_nonzero_starts_offsets': True,
                        'graph': True, 'changed_scores_lengths_starts_offsets_graph': True})
    # Paged raw and dual-output mappings: compare semantic IDs before mapping.
    for n in [1024, 8192, 16384, 32768]:
        values = torch.arange(n, dtype=torch.float32).remainder(2048) / 2048
        # Every exact-value tie pool here fits the fixed 2048-candidate capacity.
        scores = values[None].cuda(); lens = torch.tensor([n], dtype=torch.int32, device='cuda')
        pages = (torch.arange((n + 63) // 64, dtype=torch.int32)[None] * 3 + 7).cuda()
        out = torch.empty((1, 511), dtype=torch.int32, device='cuda')
        raw = torch.empty_like(out)
        plan = plan_topk_v2(lens, static_threshold=1)
        expected_raw = torch.argsort(values, descending=True, stable=True)[:511].int()
        expected_pages = pages.cpu()[0, expected_raw.long() // 64] * 64 + expected_raw % 64
        for _ in range(10):
            topk_transform_paged_v2(scores, lens, pages, out, 64, plan, out_raw_indices=raw)
            actual_raw, actual_out = raw.cpu()[0], out.cpu()[0]
            assert torch.equal(actual_raw.sort().values, expected_raw.sort().values)
            assert torch.equal(actual_out.sort().values, expected_pages.sort().values)
            assert torch.equal(actual_out, pages.cpu()[0, actual_raw.long() // 64] * 64 + actual_raw % 64)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            topk_transform_paged_v2(scores, lens, pages, out, 64, plan, out_raw_indices=raw)
        changed_values = values.flip(0); length = n - 7
        scores.copy_(changed_values[None]);lens.fill_(length);pages.add_(11)
        plan.copy_(plan_topk_v2(lens, static_threshold=1));graph.replay()
        expected_raw = torch.argsort(changed_values[:length], descending=True, stable=True)[:511].int()
        actual_raw, actual_out = raw.cpu()[0], out.cpu()[0]
        assert torch.equal(actual_raw.sort().values, expected_raw.sort().values)
        assert torch.equal(actual_out, pages.cpu()[0, actual_raw.long() // 64] * 64 + actual_raw % 64)
        records.append({'id': 'paged-dual-' + str(n), 'N': n, 'K': 511, 'eager_repeats': 10,
                        'raw_and_page_mapping_checked': True, 'cluster_plan_requested': True,
                        'changed_scores_lengths_pages_plan_graph': True})
    result = {'passed_cases': len(records), 'cases': records,
              'reference': 'Independent CPU stable argsort of exact supplied scores, index ascending at equal scores',
              'coarse_pool_overflow_above2048': 'Not qualified by this patch; not counted as passed.',
              'NaN_valid_scores': 'Undefined native semantic domain, not promoted by these tests.'}
    path = args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + '\n');print(json.dumps({'passed_cases': len(records), 'file': str(path)}))


if __name__ == '__main__':main()
