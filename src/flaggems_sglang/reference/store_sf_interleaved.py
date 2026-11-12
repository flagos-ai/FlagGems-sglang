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

"""Reference implementation for quantization/store_sf_interleaved."""


def reference(sf_in, sf_out, loc, page_size):
    out = sf_out.clone()
    page = (loc // page_size).long()
    off = (loc % page_size).long()
    out[page, :, off % 32, off // 32, :] = sf_in
    return out
