---
name: sglang-cross-backend-validation
description: 为 FlagGems-sglang 新算子开发 test、benchmark、definition，并通过 NVIDIA SGLang golden、KS3 和 ops-baseline 验证国产后端的 reference、精度及跨机性能。
---

# SGLang 算子跨后端开发与验证

按 [流程与命令](references/workflow.md) 开发并验证算子。复用 Qwen3.6/mRoPE 数据时，再读 [已验证实例与阈值](references/validated-example.md)；其中的机器、版本和阈值不是新任务的默认配置。

## 开始前

- 读仓库指令及用户指定的机器 handoff，确认算子、文档版本、目标机器、容器、工作目录和设备；不要从 SSH config 或其他实验自行扩大范围。
- 固定 FlagGems-sglang、op-verify、ops-baseline 的实际 commit，核对所需功能是否存在。使用隔离目录与任务环境，保留现有 SDK。
- 上传、实机执行、发布及合并以当前用户授权为准；新算子三件套通过 PR 进入 `kernelgen-dev`，配套功能向各自仓库提 PR。

## 保持三条证据链独立

1. **测试链路**：list cases、逐 case dummy、reference-as-candidate 和 override 验证候选接线。reference 自比较不能独立证明数学正确。
2. **跨机精度**：各后端的 Torch reference 使用固定 NVIDIA 输入，与原始 SGLang golden 比较；candidate 再与已验证的本地 reference 比较。国产后端 replay 复用 vendor reference 分发；NVIDIA 分发若返回 native，必须显式选择 portable Torch 实现做独立验证，见流程第 3 节。
3. **跨机性能**：H800 上用 ops-baseline 采集原始 SGLang 实现，目标 pytest 读取精确匹配的静态记录并实测 candidate。Torch reference 耗时不能代替 NVIDIA 原生基线。

无法连接、skip、未运行、缺失基线或缩放因子分别报告，不计为验证通过。交付记录要求见流程第 6 节。
