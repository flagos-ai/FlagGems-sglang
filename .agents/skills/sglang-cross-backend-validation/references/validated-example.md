# 已验证实例：Qwen3.6 与 mRoPE

这是本 skill 提取流程时的历史样例，不授权后续任务使用所有机器，也不代表每个后端都已有优化 candidate。复用前检查相关 PR 的当前状态、commit、接口及原始日志。

## 功能来源与依赖

- [FlagGems-sglang #249](https://github.com/flagos-ai/FlagGems-sglang/pull/249)：mRoPE portable reference 与跨机接入，已合并。作为本 skill 分支基点的 `kernelgen-dev` commit 为 `310f079`。
- [FlagGems-sglang #251](https://github.com/flagos-ai/FlagGems-sglang/pull/251)：其余七个 workload 接口、共享 Torch reference、native adapters 和多后端验证，整理本 skill 时仍未合并，验证源码为 `0617e9a60057716e4408c2e6b8a2a473352ca692`。
- [op-verify #9](https://github.com/lukatao/op-verify/pull/9)：v2 返回/变更/alias 契约、逐输出阈值、压缩 bundle、目标平台支持；整理时未合并，验证源码为 `9b4703b896a48aa6792198eb19cd21a0ffb57e51`。
- [ops-baseline #68](https://github.com/zhaojs12345/ops-baseline/pull/68)：独立 `ops/sglang_qwen36` recipes 和 H800 artifact，整理时未合并，验证源码为 `c43874649248de6024e4b492e08a7a48ebed512c`。

不要仅安装目标分支就假定拥有上述未合并能力；以实际版本确认依赖。不要将这些 feature commits 的实现拷入文档 PR。

## Workload 与数据

[主 workload 文档](https://jwolpxeehx.feishu.cn/wiki/VTUbwwCzsiynr5kKe2Zcpd2cnpf) 与 27B/35B 子文档提供几何和调用配置；此次核对版本分别为 107、21、54。tensor 值为确定性合成值，并非实际推理 dump。

| 公共算子名 | benchmark core | 正确性 |
| --- | ---: | ---: |
| `gemma_rmsnorm` | 12 | 13 |
| `gemma_fused_add_rmsnorm` | 4 | 5 |
| `silu_and_mul_out` | 4 | 5 |
| `select_experts` | 2 | 3 |
| `_fused_moe_kernel_sequence` | 2 | 3 |
| `chunk_gated_delta_rule_fwd` | 2 | 6 |
| `fused_recurrent_gated_delta_rule_packed_decode` | 2 | 3 |
| 已有 `mrotary_embedding` | 6 | 7 |

共 34 core、45 correctness。固定 NVIDIA golden 为七个新接口的 28 core 加 mRoPE 的 7 correctness，即 35 cases，不能声称 golden 覆盖所有 45 个 correctness cases。

MoE marker 为 `underscore_fused_moe_kernel_sequence`，problem ID 为 `qwen36/fused_moe_kernel_sequence`，注意与公共算子名的映射。已有 `silu_and_mul` 与新 `silu_and_mul_out` 的 ABI 不同，不互相替换。

- Qwen golden prefix：`op-verify/sglang-qwen36/20261030-native-v1`
- Qwen manifest SHA256：`d37d49f2dab3c14108acdd0ad730cb1d11d32424f7c19f5b03045ae377ca4ece`
- mRoPE prefix：`op-verify/sglang-mrotary/20261029-native-v1`
- mRoPE manifest SHA256：`22d91f78dfc1f702732ebe57747c6a376fdd3a7c405dab3742fc4c384a74efee`
- 七个新接口 H800 artifact：ops-baseline 的 `op_sglang_qwen36_h800_perf_baseline.json`，SHA256 `d27aba1e52f7ac41800b0b3e108c383dd8c044256dc7789ac4574318b39aa09f`，28 latency + 28 有效 NCU 记录。
- 已有 mRoPE artifact：ops-baseline 的 `op_sglang_mrotary_h800_perf_baseline.json`。

七个新接口的原生环境使用 SGLang v0.5.11 commit `612785ffdcaf35552f1ed433a981d596ca9fe900`、Torch 2.11.0+cu130、sglang-kernel 0.4.2、FlashInfer 0.6.8.post1、Triton 3.6.0。mRoPE 最初采集使用另一套环境，见本仓库 `docs/mrotary_cross_machine.md`；不能合写成相同运行环境。

## 精度策略

以下为本次工程选择，workload 文档没有规定这些阈值。不是适用于所有新算子的统一标准。验证时核对 manifest 中实际 `check` 和 `output_tolerances`，也核对 pytest 的显式检查。

| 输出类别 | atol | rtol |
| --- | ---: | ---: |
| 本次常规 BF16 输出和 GDN FP32 状态池 | 0.002 | 0.016 |
| chunk 的 FP32 累积结果（返回 slot 0） | 2e-5 | 2e-5 |
| select_experts FP32 权重及其契约下的 FP32 透传输出 | 2e-5 | 2e-5 |
| select_experts 专家索引（slot 1） | 0 | 0 |

所有 golden 检查拒绝 NaN。索引、未变更区域和结构/alias 等另按精确契约检查。后端间没有按机器放宽阈值；FP32 状态池的容差不能套用到所有 FP32 tensor。

## 后端验证范围与已知差异

本次验证覆盖 Ascend、Iluvatar、Kunlunxin、Hygon、MetaX，以及用户追加的 T-Head PPU 和 MThreads MUSA。Suiyuan 按用户要求跳过；没有验收 Cambricon。共享数学 Torch reference 在上述七类目标上均可使用，无需新增 vendor 数学特化。

六个新目标（除先前单独验证的 Ascend）逐台完成 35 golden、45 ref-as-candidate correctness、45 correctness dummy 触达、34 实际 core 计时与基线匹配，并核对每台下载的 70 个输入/输出 payload 引用摘要。Ascend 证据来自此前分别完成的 Qwen 与 mRoPE 验证。参阅 #251 中的 `docs/qwen36_multibackend_validation.json` 和 `docs/qwen36_workloads.md`；这些结果验证 reference/测试管线，不代表六台均完成优化 candidate 性能评估。

有复用价值的环境差异：

- Kunlunxin 原 SDK 的无关内置候选装饰器不兼容时，显式 `FLAGGEMS_SGLANG_REFERENCE_ONLY=1` 可跳过内置候选注册，保留真实 runtime/reference 分发。此功能来自 #251，默认行为不变；只用于 reference 或明确注入的 candidate。
- Kunlunxin 测试 fixture 的随机排列通过隔离的 CPU generator 构建后搬到目标，数学 reference 留在目标。KS3 replay 直接使用固定输入。
- 基线历史 vendor 标签 `t-head` 与 runtime 的 `thead` 需一致归一化。
- 若 SDK 仅报告 `GPU`，可用实际设备工具确认型号后显式 `FLAGGEMS_SGLANG_CHIP=P800`。它选择缩放型号，不能代替设备可见性配置。
- P800 缺少 bandwidth factor，而这批 34 个基线均为 memory bottleneck，因此报告 `factor=None` 和原始延迟比。不能借用 P900 factor，也不能声称已完成硬件归一化。

具体 IP、容器、接入和运行目录从用户当前 handoff 获取；不要将本次环境差异当成新任务的强制机器配置。
