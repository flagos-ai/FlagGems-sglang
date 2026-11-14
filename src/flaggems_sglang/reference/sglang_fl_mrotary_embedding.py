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

"""Reference for attention/sglang_fl_mrotary_embedding."""

import torch


def axis_lanes(
    mrope_section, num_pairs, mrope_interleaved, mrope_interleaved_glm
):
    """Which of the temporal/height/width axes owns each rotary lane.

    This is the same allocation rule the upstream layer uses to build its
    ``axis_map`` buffer, so the three branch behaviours live in exactly one
    place here.
    """
    section = list(mrope_section)

    if mrope_interleaved_glm:
        lanes = []
        spent = [0, 0, 0]
        for lane in range(num_pairs):
            axis = lane % 3
            # Walk forward until an axis with quota left is found. Sections
            # that sum to num_pairs guarantee termination.
            while spent[axis] >= section[axis]:
                axis = (axis + 1) % 3
            spent[axis] += 1
            lanes.append(axis)
        return lanes

    if mrope_interleaved:
        lanes = [0] * num_pairs
        for axis in (1, 2):
            stop = min(3 * section[axis], num_pairs)
            for lane in range(axis, stop, 3):
                lanes[lane] = axis
        return lanes

    lanes = []
    for axis, size in enumerate(section):
        lanes.extend([axis] * size)
    return lanes


def _cos_sin(cos_sin_cache, positions, lanes):
    num_pairs = len(lanes)
    index = torch.as_tensor(lanes, dtype=torch.long, device=positions.device)
    # positions is [3, tokens]; gather the owning axis per rotary lane, then
    # transpose so each row is one token's pair of positions.
    pos = positions.index_select(0, index).transpose(0, 1)
    lane = torch.arange(num_pairs, device=positions.device)
    cos = cos_sin_cache[pos, lane.view(1, -1).expand(pos.shape)]
    sin = cos_sin_cache[pos, (lane + num_pairs).view(1, -1).expand(pos.shape)]
    return cos, sin


def _rotate(x, cos, sin, head_size, rotary_dim, is_neox_style):
    """Rotate the first ``rotary_dim`` channels of every head, in place."""
    tokens = x.shape[0]
    num_pairs = rotary_dim // 2
    heads = x.shape[1] // head_size

    view = x.view(tokens, heads, head_size)
    rot = view[..., :rotary_dim]

    cos = cos[:, None, :]
    sin = sin[:, None, :]

    if is_neox_style:
        first = rot[..., :num_pairs]
        second = rot[..., num_pairs:rotary_dim]
        out_first = first * cos - second * sin
        out_second = second * cos + first * sin
        rotated = torch.cat((out_first, out_second), dim=-1)
    else:
        even = rot[..., 0:rotary_dim:2]
        odd = rot[..., 1:rotary_dim:2]
        out_even = even * cos - odd * sin
        out_odd = odd * cos + even * sin
        rotated = torch.stack((out_even, out_odd), dim=-1).flatten(-2)

    # Only the rotary band is written; the trailing head_size - rotary_dim
    # channels are carried through untouched.
    view[..., :rotary_dim] = rotated
    return x


def reference(
    q,
    k,
    cos_sin_cache,
    positions,
    mrope_section,
    head_size,
    rotary_dim,
    mrope_interleaved,
    mrope_interleaved_glm,
    is_neox_style,
    axis_map=None,
):
    num_pairs = rotary_dim // 2

    lanes = axis_lanes(
        mrope_section, num_pairs, mrope_interleaved, mrope_interleaved_glm
    )

    # The caller supplies a prebuilt map only in the glm branch; assert it
    # agrees rather than trusting it silently.
    if axis_map is not None:
        supplied = axis_map.tolist()
        if supplied != lanes:
            raise AssertionError(
                "axis_map disagrees with the section allocation: "
                f"{supplied} != {lanes}"
            )

    cos, sin = _cos_sin(cos_sin_cache, positions, lanes)

    _rotate(q, cos, sin, head_size, rotary_dim, is_neox_style)
    _rotate(k, cos, sin, head_size, rotary_dim, is_neox_style)

    # The contract is in-place with no outputs.
    return None
