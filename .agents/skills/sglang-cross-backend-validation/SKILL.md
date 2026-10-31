---
name: sglang-cross-backend-validation
description: 为 FlagGems-sglang 新算子开发 test、benchmark、definition，并通过 NVIDIA SGLang golden、KS3 和 ops-baseline 验证国产后端的 reference、精度及跨机性能。用于 workload 接入及跨后端验证。
---

# SGLang 算子跨后端开发与验证

交付可复用的三件套、显式 reference、可追溯的 NVIDIA 固定数据与原生性能基线，以及按机器区分的验证结果。新算子的 test、benchmark、definition 通过 PR 进入本仓库 `kernelgen-dev`；配套功能进入各自仓库的 PR。是否上传、实机执行、发布及合并，以当前用户授权为准，使用本 skill 本身不扩大授权范围。

先读 [流程与命令](references/workflow.md)。只有复用 Qwen3.6/mRoPE 实例时，再读 [已验证实例与阈值](references/validated-example.md)。示例中的 case 数、软件版本、机器和阈值不是新任务的默认要求。

## 开始前确定边界

- 读当前仓库指令及用户指定的机器 handoff，确认算子、文档版本、目标机器、容器、工作目录、设备及现有运行时。不要从 SSH config、其他实验或旧日志自行扩展机器范围。
- 固定三个仓库的实际 commit：FlagGems-sglang、op-verify、ops-baseline。检查所需能力是否已合并；不要把其他 PR 的文件当成目标分支已有功能。
- 使用隔离分支/目录与任务环境，保留现有 SDK。记录实际硬件与软件；无法连接或缺少能力时明确记为未验证，不算通过。

## 三条证据链

1. **测试链路**：list cases、逐 case dummy 注入、reference-as-candidate 和实际 override。它们证明测试能列举、执行并触达候选。ref-as-candidate 自比较不能独立证明数学正确。
2. **跨机精度**：在 NVIDIA 上运行固定版本的原始 SGLang 实现，保存调用前输入及返回/变更输出，上传 KS3；各后端用同一批输入运行其 Torch reference，与 NVIDIA golden 比较。后端 candidate 再与已验证的本地 reference 比较。
3. **跨机性能**：在 H800 用 ops-baseline 测量原始 SGLang 实现；目标 pytest 从静态 artifact 读取精确匹配的记录，同时实测本地 candidate。不能用 Torch 组合 reference 的耗时替代 NVIDIA 原生性能。

## 实施要点

- 每个算子包含 `tests/test_<op>.py`、`benchmark/test_<op>.py`、`definitions/<op>.json`。正确性与 benchmark 尽量共享 workload/input builder，benchmark core 覆盖文档 workload，正确性包括这些 workload 和必要边界。
- 保留完整 ABI：shape、dtype、stride、storage offset、标量配置、返回结构、None、mutation、alias、只读输入及未选中状态。记录合成输入与真实 tensor dump 的区别。不能只按 shape 或 case 数声称覆盖一致。
- 显式定义 reference；正确性使用 `get_candidate(op_name, reference=reference)`，benchmark 提供同一语义的 `torch_op`。复用已有 override/dummy/ref-as-candidate 选择机制，缺 reference 时让公共机制明确报错。保持默认测试方式和注入入口兼容。
- reference 首选共享 Torch 组网。只有复现后端能力差异后才增加 `src/flaggems_sglang/reference/vendors/_<vendor>/<op>.py` 的 `reference`，经 `get_reference` 分发；golden replay 必须调用同一分发。不要另写一套验证专用数学实现。
- 不假定 NVIDIA 原始实现均为 Triton：检查实际 CUDA、FlashInfer、SGLang kernel 或 Triton 入口、依赖和调用参数，需要时建立独立 SGLang 环境并固定源码/轮子摘要。原生入口缺失应失败，不能静默换成 Torch reference。
- 精度阈值按算子、dtype、输出制定并说明来源，在 pytest 与 KS3 manifest 中保持语义一致。结构、索引和副作用契约单独检查；不要为了让某台机器通过而临时放宽阈值。
- 全链路结果区分：已列举、dummy 已触达、数值通过、实际计时完成、基线匹配、可折算。skip、未运行、缺少缩放因子均单独报告。

## 交付

PR 写明 workload 来源、case 覆盖、reference 来源、实际机器、阈值理由、KS3 prefix/manifest 摘要、静态 baseline、环境版本、实测结果和限制。保留原始日志与失败诊断，只清理本任务拥有且已结束的进程/连接。仓库内不放凭证、机器密码或私有接入信息。未经明确要求不合并 PR。
