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


def reference(k_view, v_view, cache_k, cache_v, loc, page_size):
    kv, vv = k_view.clone(), v_view.clone()
    page = (loc // page_size).long()
    tok = (loc % page_size).long()
    kv[page, tok] = cache_k.to(kv.dtype)
    vv[page, tok] = cache_v.to(vv.dtype)
    return kv, vv
