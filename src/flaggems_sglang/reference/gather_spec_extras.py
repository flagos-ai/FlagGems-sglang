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


def reference(
    indices, topk_p_buf, topk_index_buf, output_tokens_buf, hidden_states_buf
):
    idx = indices.contiguous().long()
    hidden = hidden_states_buf[idx] if hidden_states_buf is not None else None
    return (
        topk_p_buf[idx],
        topk_index_buf[idx],
        output_tokens_buf[idx],
        hidden,
    )
