#!/bin/bash


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

# 接收从 backend-test.yaml 传过来的 vendor 参数 (如 nvidia, ascend)
VENDOR=${1:?"Usage: bash tools/run_tests.sh <vendor>"}
# 如果你的项目需要特定的环境变量，在这里导出。这里参考 FlagGems 的命名：
export GEMS_VENDOR=$VENDOR

echo "===================================================="
echo "🚀 开始运行 FlagGems-sglang 测试 | 硬件平台: $GEMS_VENDOR"
echo "===================================================="


export CUDA_VISIBLE_DEVICES=0

# 激活 Conda 环境
# 注意：请务必把这里的路径和虚拟环境名称换成你 GPU 服务器上的实际路径！
# 如果你的 Runner 已经在正确的环境里了，这两行可以注释掉。
# source "/path/to/your/miniconda3/etc/profile.d/conda.sh"
# conda activate your_env_name

# 引入执行命令的 wrapper (遇到报错会自动停止 CI)
source tools/run_command.sh

echo "----------------------------------------------------"
echo "开始执行 Pytest 单元测试..."

# 执行测试脚本
run_command pytest -s tests/test_relu.py
# run_command pytest -s tests/test_sqrt.py


echo "===================================================="
echo "✅ 所有测试执行完毕，全部通过！"
echo "===================================================="
