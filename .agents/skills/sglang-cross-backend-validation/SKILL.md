---
name: sglang-cross-backend-validation
description: 为 FlagGems-sglang 新算子开发 test、benchmark、definition，并通过 NVIDIA SGLang golden、KS3 和 ops-baseline 验证国产后端的 reference、精度及跨机性能。
---

# SGLang 算子跨后端开发与验证

按 [流程与命令](references/workflow.md) 开发并验证算子。复用已有算子的跨机验证数据时，再读 [已验证实例与阈值](references/validated-example.md)；其中的机器、版本和阈值不是新任务的默认配置。

## 开始前

- 读仓库指令及用户指定的机器 handoff，确认算子、文档版本、目标机器、容器、工作目录和设备；不要从 SSH config 或其他实验自行扩大范围。
- 固定 FlagGems-sglang、op-verify、ops-baseline 的实际 commit，核对所需功能是否存在。使用隔离目录与任务环境，保留现有 SDK。
- 上传、实机执行、发布及合并以当前用户授权为准；新算子三件套通过 PR 进入 `kernelgen-dev`，配套功能向各自仓库提 PR。

## 保持三条证据链独立

1. **测试链路**：list cases、逐 case dummy、reference-as-candidate 和 override 验证候选接线。reference 自比较不能独立证明数学正确。
2. **跨机精度**：各后端的 Torch reference 使用固定 NVIDIA 输入，与原始 SGLang golden 比较；candidate 再与已验证的本地 reference 比较。国产后端 replay 复用 vendor reference 分发；NVIDIA 分发若返回 native，必须显式选择 portable Torch 实现做独立验证，见流程第 3 节。
3. **跨机性能**：H800 上用 ops-baseline 采集原始 SGLang 实现，目标 pytest 读取精确匹配的静态记录并实测 candidate。Torch reference 耗时不能代替 NVIDIA 原生基线。

无法连接、skip、未运行、缺失基线或缩放因子分别报告，不计为验证通过。交付记录要求见流程第 6 节。

## 任务结束后清理机器

- 实机开发、验证、诊断或基线采集结束后，默认清理本任务在各机器创建的中间产物，包括临时源码副本、虚拟环境、构建目录、下载的 wheel、编译缓存和临时 tensor/cache；失败任务也需收尾。明确要求保留复用的内容除外，不把临时目录默认为长期环境。
- 清理时直接删除已确认可清理的中间产物，不额外制作或保留备份、归档、证据包、清理清单、manifest、receipt 或其他清理记录，也不将其转存到本地或其他机器。仅在用户明确要求保留时保存相应内容；完成本次清理核验后删除为清理临时生成的脚本和文件。正式 golden、KS3 数据和已发布 baseline 是交付物，不随中间产物删除。
- 删除前核对目录归属、运行状态及共享依赖，检查宿主机和共享挂载容器的进程引用。仅清理本任务拥有且已不再使用的明确路径；不删除共享仓库、SDK、公共环境或其他任务产物，不为清理而停止其他任务。无法确认归属或仍被引用时保留，并报告具体阻碍。
- 每轮实机工作结束后关闭已不再需要的自有连接，在回复中简要说明完成情况和必要的保留项，不为此另存清理报告。PR 待 review 不能作为默认保留远程中间目录的理由；保留活跃任务必需的环境和 PR review 所需的本地源码，其余无用中间产物直接清理，不以制作可恢复副本为前提。
