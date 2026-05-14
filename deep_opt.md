# Deep Optimization: 性能未达标时的二阶段攻关手册

## 0. 文档定位

本文档只在“一阶段实现已经正确，但性能未达标”时启用。它的目标不是重新讲开发流程，而是强制 AI 进入 profile-driven 的性能攻关循环，持续尝试与算子家族匹配的高级策略，直到达到目标、收缩覆盖面或形成可解释的停止结论。

当进入本文档时，AI 不再允许泛泛地说“可以继续优化”。必须维护 Deep Opt Board，逐轮给出假设、证据、改动、验证、benchmark 和 keep/revert。

## 1. 进入条件

必须同时满足：

- 功能测试通过。
- dtype、边界语义、数值行为、unsupported scope 已记录。
- benchmark active set、metric、unit、direction、aggregation 已冻结。
- 初版 host dispatch、kernel、测试、benchmark、导出、注册已完成。
- 生产实现、dispatch、helper、fallback 中没有 torch compute fallback。
- NVIDIA backend 的可调 kernel 已接入 `runtime.get_tuned_config("<op_or_path>")` + `@libtuner(...)`，或 no-autotune 豁免已写明。
- 性能未达标，例如关键目标 shape `SpeedUp < 0.9`，或 active set 中关键 workload 出现不可接受回归。

不满足这些条件时，先回 `workflow.md` / `optimization.md`，不要提前堆技巧。

## 2. 禁止事项

进入 deep optimization 后，以下行为一律禁止：

- 用 `torch.<op>`、`torch.nn.functional.*`、`torch.ops.aten.*` 或 Tensor compute 方法作为生产 fallback。
- 为了达标而放松 dtype 覆盖、边界语义、NaN/Inf、tie-break、deterministic、accumulate precision。
- 修改 benchmark 口径后继续沿用旧 baseline。
- 不看 dispatch/autotune/profile，连续盲调 block size、num_warps 或 unroll。
- 一轮同时改算法、host dispatch、autotune key、候选空间、benchmark harness。
- 用少数非关键 shape 的提升掩盖关键 active workload 的回归。

如果发现 torch compute fallback，应立即停止性能讨论，先修复生产路径。

## 3. 退出条件

满足任一条件即可停止当前 deep optimization 轮次：

- active set 达到性能目标，且功能测试、dtype、边界语义仍通过。
- 新策略收益只集中在非关键 shape，并让关键 workload 回归。
- 连续 3 到 5 轮 profile-driven trial 没有稳定收益，且已尝试该家族主要策略。
- profiler 证明瓶颈来自无法在本算子内修复的上游布局、框架调度、厂商库优势或硬件限制。
- 进一步优化需要改变语义、放松 dtype、修改 benchmark 口径或引入不可接受维护复杂度。
- 自研覆盖面应收缩：该 path 改为 unsupported、external-library path 或只保留 fast path。

停止不等于失败。停止结论必须写清尝试过什么、为什么停止、建议收缩哪些覆盖面，以及为什么不能用 torch fallback。

## 4. Deep Opt Board

进入本文档后必须维护以下表格。每轮 trial 更新一次。

| 字段 | 内容 |
| --- | --- |
| Failed target | 未达标 dtype/shape/layout/path，目标阈值 |
| Baseline evidence | FlagDNN latency、torch latency、SpeedUp、aggregation、运行命令 |
| Production audit | 是否存在 torch compute fallback；可疑调用及处理 |
| Dispatch evidence | 目标 shape 实际走的 path、host gate、workspace、early return |
| Autotune evidence | 配置名、候选集、key、strategy、best config、是否触发目标 path |
| Bottleneck guess | bandwidth / compute / launch / dispatch / atomic / occupancy / cache / compile |
| Profiler evidence | nsys/ncu 或其他 profiler 关键证据；不可用时写替代证据 |
| Operator family | pointwise、reduction、scan、softmax/norm、pooling、GEMM-like、selection、scatter/gather、embedding、backward |
| Candidate strategies | 本轮从本文档选择的 1 到 3 个策略 |
| Trial change | 本轮唯一主要变化，包括代码 path 与 tune config 变化 |
| Validation | 功能测试命令和结果 |
| Benchmark result | active set 结果、improved/regressed/unchanged |
| Keep/revert | 决策和原因 |
| Next action | 继续、换策略、拆 path、收缩覆盖面、停止 |

## 5. 强制攻关顺序

不要一进入 deep optimization 就调 tile。按以下阶段推进。

### Phase 0. 复核验收口径

确认：

- active set 覆盖失败 workload。
- metric、unit、direction 正确。
- SpeedUp 公式未反向。
- benchmark 没有冷启动污染或缓存不一致。
- torch baseline 和 FlagDNN 实现使用相同输入、dtype、layout、同步规则。
- benchmark harness 未在优化过程中被改口径。

若口径不一致，先重建 baseline。

### Phase 1. 审计生产路径和 dispatch

确认：

- 生产路径无 torch compute fallback。
- 失败 shape 走到预期 fast/special/general path。
- tiny/global/contiguous/dtype fast path 没被 general path 吞掉。
- host 判断顺序从 cheap 到 expensive。
- workspace 没有隐藏重复分配。
- 不支持 path 是显式报错，而不是 torch 补洞。

很多性能问题来自错误 path，而不是 kernel 主循环。

### Phase 2. 审计 autotune

确认：

- `runtime.get_tuned_config("x")` 对应 `tune_configs.yaml` 中存在 `x`。
- 候选集非空，META 与 `tl.constexpr` 一致。
- key 没有过粗导致不同形态共用错误配置。
- key 没有过细导致编译/缓存碎片。
- active set 实际触发该 autotuned path。
- best config 与失败 shape 有关，而不是被其他 shape 主导。
- 若变化涉及 block/tile/group/unroll/warps/stages，优先更新 `tune_configs.yaml`，不要只在代码里写死。

### Phase 3. 做系统级 profile

NVIDIA backend 先用 `nsys` 看：

- kernel launch 数。
- kernel 时间占比。
- host gap。
- 同步、拷贝、分配。
- 是否多 kernel pipeline 被 launch overhead 吃掉。

如果 launch 或 host gap 主导，优先合并 kernel、减少中间 tensor、early return、host gate、workspace 复用，而不是调 tile。

### Phase 4. 做 kernel 级 profile

再用 `ncu` 看核心 kernel：

- achieved bandwidth。
- FLOPS/TOPS。
- SM occupancy。
- register / spill。
- shared memory 使用和 bank conflict。
- L2 hit rate。
- warp stall 原因。
- atomic conflict / retry。
- Tensor Core 利用率。

如果 ncu 不可用，必须用替代证据：latency 随 shape 的 scaling、launch 数、occupancy 估算、autotune best config、路径拆分实验。

### Phase 5. 按家族选择策略

每轮只选择一个主变化。优先级：

1. 修正 path / dispatch。
2. 修正访存方向和连续性。
3. 修正并行粒度和 tile。
4. 修正同步、atomic、two-stage、workspace。
5. 修正 dtype / precision / Tensor Core。
6. 收敛 autotune key 和候选空间。
7. 收缩自研覆盖面或停止。

## 6. Profile 信号到动作映射

| 信号 | 常见含义 | 首选动作 | 不应先做 |
| --- | --- | --- | --- |
| launch 数多，kernel 很短 | host/launch overhead 主导 | 合并 kernel、early return、tiny path、减少中间 tensor | 盲调 BLOCK_SIZE |
| host gap 大 | Python/dispatch/workspace 开销 | 缓存 spec、减少分配、简化 dispatch、复用 workspace | 改 num_warps |
| bandwidth 低且 load/store 不连续 | 访存未合并或地址复杂 | contiguous fast path、轴搬移、block_ptr、向量化、简化 mask | 增大 tile |
| bandwidth 接近上限 | memory-bound 正常 | 减少读写、融合、压缩中间数据、提高 cache reuse | 追求更高 FLOPS |
| FLOPS 低且 Tensor Core 未用 | 算力路径错误 | `tl.dot`、dtype/layout 对齐、K 分块、input_precision | 增加 stages |
| occupancy 低且寄存器高 | tile 过大或活跃变量多 | 缩小 tile、拆 kernel、rematerialization、branch-local pointer | 增大 BLOCK |
| spill 明显 | 寄存器压力过高 | 减少 accumulators、拆 path、重算索引、降低 unroll | 增加 ROWS_PER_PROGRAM |
| L2 hit 低 | tile 顺序/复用差 | program id swizzle、grouped order、cache policy | 单纯加 warps |
| stall memory dependency 高 | load 延迟暴露 | 提高并行度、prefetch/双缓冲、num_stages、L2 locality | 减小并发 |
| stall execution dependency 高 | 算术依赖链长 | 多 accumulator、合理 unroll、改归约树 | 只调 cache |
| atomic retry 高 | 写冲突主导 | 局部聚合、分桶、two-stage reduce/finalize | 继续 atomic_add |
| shared memory bank conflict | smem 布局问题 | padding、tile shape、减少 smem、寄存器路径 | 增大 smem |
| 编译/调优很慢 | 变体太多 | key 压缩、候选裁剪、do_not_specialize、拆配置 | 扩大搜索空间 |
| 某些 shape 极慢 | dispatch 阈值错 | 单独 special path、调 crossing point | 全局改参数 |

## 7. 通用策略阶梯

无论算子家族如何，优先按这个阶梯推进。

1. **路径修正**：fast/special/general/unsupported 是否正确。
2. **访存修正**：连续性、轴方向、mask、block_ptr、cache、向量化。
3. **并行度修正**：program 粒度、tile shape、warps、stages、split、persistent。
4. **同步修正**：atomic、本地聚合、two-stage、workspace、locks。
5. **数值/dtype 特化**：fp16/bf16/fp32/fp64/complex/int/bool 的 acc dtype 和 tile 分流。
6. **Autotune 收敛**：配置组拆分、key/strategy 修正、候选空间裁剪。
7. **收缩覆盖面**：若该 path 无法稳定赢，改为 unsupported 或项目批准的 external-library path，不调用 torch fallback。

## 8. Trial 纪律

每一轮 trial 必须满足：

- 从当前 best 开始。
- 只改一个主要变量。
- 改动前写假设。
- 改动后先 validation，再 benchmark。
- benchmark 结束后再决定 keep/revert。
- 若需要重建 baseline，必须先停止比较旧结果。

Trial 结果格式：

```text
Trial <n>:
- Target: <dtype/shape/path>
- Hypothesis: <one sentence>
- Change: <code/config>
- Validation: <pass/fail + command>
- Benchmark: <SpeedUp before -> after, active regressions>
- Decision: <keep/revert/split/stop>
- Evidence: <profiler or alternative>
```

## 9. 家族策略卡

### 9.1 Pointwise / Broadcast

失败信号：

- 小 shape 慢，launch/dispatch 占比高。
- 大 shape bandwidth 低。
- broadcast rank 高时明显慢于 contiguous。
- 融合后寄存器激增或 occupancy 下降。

优先 trial：

1. **contiguous flat path**：连续输入输出展平成 1D，最小地址计算和 1D mask。
2. **no-broadcast 与 broadcast 分离**：避免每个元素做多维 stride 解码。
3. **rank-specialized path**：1D/2D/4D 常见 rank 独立 kernel，general rank 只兜底或 unsupported。
4. **fusion pressure audit**：复杂表达式拆成两个 kernel，比较 launch vs register tradeoff。
5. **dtype-special tile**：fp16/bf16 更大 block，fp64 更保守，bool/int 避免无意义 fp 转换。

Autotune 候选：

- `BLOCK_SIZE`: 128, 256, 512, 1024, 2048
- `num_warps`: 4, 8
- `num_stages`: 2, 3

接受条件：bandwidth 或 latency 在 active set 稳定提升，且 tiny shape 没被更大 launch/compile 成本拖慢。

风险：signed zero、NaN、complex、integer overflow、bool 语义不能改变。

### 9.2 Reduction

失败信号：

- 输出多但每个 K 中等，吞吐低。
- 输出少但 K 极长，单 program 串行化。
- atomic 冲突高。
- 归约轴不连续，load efficiency 低。

优先 trial：

1. **按输出规模 × K 拆 path**：many-output medium-K、few-output huge-K、tiny-output 分离。
2. **loop reduction path**：输出多、中长 K 时，一个 program 处理一个或多个输出。
3. **split/two-stage reduction**：输出少、K 极长时，第一阶段 partial，第二阶段 finalize。
4. **atomic path only when conflicts low**：冲突高则局部聚合或 two-stage。
5. **contiguous axis fast path**：最后一维归约单独优化；非连续轴评估 host 重排或 stride path。
6. **value+index 一体化**：max/min/arg* 不允许 value 后回扫 index。

Autotune 候选：

- `BLOCK_M`: 1, 2, 4, 8, 16, 32, 64, 128
- `BLOCK_K`: 64, 128, 256, 512, 1024, 2048, 4096
- `SPLIT_K`: 2, 4, 8, 16
- `num_warps`: 4, 8, 16

配置建议：`<op>_reduce_loop`、`<op>_reduce_wide`、`<op>_reduce_atomic` 分离。

风险：accumulate dtype、NaN 传播、Inf、empty reduction、keepdim、output dtype、deterministic、workspace 多 stream。

### 9.3 Scan / Prefix-Sum

失败信号：

- 长轴 scan 单 kernel 慢或寄存器高。
- 行/列 scan 性能差异极大。
- 非连续轴 load/store efficiency 低。
- 多阶段 launch 成本压过小 shape。

优先 trial：

1. **short row persistent**：固定短长度，一行一个 program 或一 program 多行。
2. **long row three-stage**：块内 scan、块和 scan、回填块前缀。
3. **row/column 分流**：连续 row scan 与 strided column scan 不混用。
4. **small shape single/fixed path**：避免多阶段 launch 成本。
5. **dtype 分流**：fp16/bf16 用 fp32 中间累加；整数检查溢出语义。

Autotune 候选：

- `BLOCK_SIZE`: 128, 256, 512, 1024
- `ROWS_PER_PROGRAM`: 1, 2, 4, 8
- `num_warps`: 4, 8

风险：inclusive/exclusive、dim、empty、NaN、integer overflow、partial buffer 初始化。

### 9.4 Softmax / Normalization

失败信号：

- 短行 launch overhead 或 occupancy 差。
- 长行 register spill 或多次读取过多。
- norm 在大 hidden 上慢，mean/var 重复读写。
- 数值误差或 NaN 行为不对齐。

优先 trial：

1. **按 N 三段拆 path**：small persistent、medium one-row-program、large online/two-pass。
2. **inner/non-inner 分流**：最后一维 contiguous fast path；非最后一维先评估转置/stride path。
3. **online softmax**：长行减少中间存储，控制寄存器。
4. **Welford / merged statistics**：mean/var/std/layer_norm/group_norm/rms_norm 共享统计。
5. **融合 affine/activation**：只在寄存器和输出语义允许时融合。
6. **all -inf / NaN path audit**：数值边界不允许被 fast path 改坏。

Autotune 候选：

- `BLOCK_N`: 128, 256, 512, 1024, 2048, 4096, 8192
- `ROWS_PER_PROGRAM`: 1, 2, 4
- `num_warps`: 4, 8, 16, 32
- `num_stages`: 2, 3, 4

配置建议：`<op>_softmax_small`、`<op>_softmax_medium`、`<op>_softmax_large`。

风险：exp overflow、全 `-inf`、NaN 传播、epsilon、biased/unbiased variance、accumulate dtype。

### 9.5 Window / Pooling

失败信号：

- global pool 仍走 local window general logic。
- output 很小但 program 数碎片化。
- window 可整除但每次动态算边界。
- 3D pooling 或大窗口地址计算过重。

优先 trial：

1. **global -> reduction**：完全绕开局部窗口地址逻辑。
2. **divisible window path**：窗口、stride、padding 静态化。
3. **small output path**：一个 program 多个输出点。
4. **layout 分流**：NCHW/NHWC/channels-last 独立地址计算。
5. **max pool index path**：value/index 一体搬运，tie-break 对齐。

Autotune 候选：

- `BLOCK_C`: 16, 32, 64, 128
- `BLOCK_O`: 1, 2, 4, 8, 16
- `WINDOW_BLOCK`: 4, 8, 16, 32
- `num_warps`: 4, 8

风险：padding、ceil_mode、count_include_pad、dilation、return_indices、NaN/tie-break、越界读。

### 9.6 MatVec / GEMM-like / Convolution-like

失败信号：

- fp16/bf16 没用 Tensor Core。
- 小 M/N/K 被常规 tile 拖慢。
- `K >> M,N` 时尾波空转。
- 1x1 conv / depthwise 仍走 general conv 映射。
- 大 shape 上强库明显更快。

硬约束：如果自研 path 不适合某些 GEMM/Conv 形态，不得 fallback 到 `torch.mm`、`torch.matmul`、`torch.nn.functional.linear`、`torch.conv*`。只能 unsupported、项目批准外部库或收缩自研覆盖面。

优先 trial：

1. **确认是否应自研**：大而规则 GEMM/Conv 若强库长期更优，自研只保留融合、小 shape、特殊布局或特殊语义 path。
2. **Tensor Core path audit**：`tl.dot`、dtype 对齐、K tile、input_precision、layout 和 stride。
3. **small matrix path**：tiny M/N/batch，一个 program 覆盖更多输出 tile，减少调度开销。
4. **swizzled/grouped tiling**：提高 L2 复用，避免线性 tile 顺序。
5. **wide-K / split-K / Stream-K**：`K >> M,N` 或尾波空转明显时启用，先设计 partial buffer、locks、deterministic。
6. **epilogue fusion**：bias/activation/scale 融合，但不破坏输出 dtype 和语义。
7. **1x1 conv path**：去掉空间窗口循环，本质转 GEMM/batched GEMM。
8. **depthwise path**：通道独立，不保留 general group conv 复杂映射。
9. **packing/cache**：热路径需要特殊布局时，host 侧 pack/cache，key 绑定 data_ptr、shape、stride、dtype、device、版本。

Autotune 候选：

- `BLOCK_M`: 16, 32, 64, 128
- `BLOCK_N`: 16, 32, 64, 128, 256
- `BLOCK_K`: 32, 64, 128
- `GROUP_M`: 4, 8
- `SPLIT_K`: 1, 2, 4, 8
- `num_warps`: 4, 8
- `num_stages`: 3, 4, 5

配置建议：`<op>_gemm_tiny`、`<op>_gemm_regular`、`<op>_gemm_wide_k`、`<op>_gemm_fp64`、`<op>_conv_1x1`、`<op>_conv_depthwise`。

风险：output dtype、accumulate precision、TF32/input_precision、complex、empty K、non-contiguous、alias、split-K atomic/locks、多 stream。

### 9.7 Selection / Sorting / TopK / Arg*

失败信号：

- `topk` 接近全量 sort，随 N 增长过快。
- `argmax/argmin` value 后回扫 index。
- tie-break 不一致。
- 大 K 下寄存器或 smem 压力高。

优先 trial：

1. **k=1/small-K special**：arg-only 和小 K 单独 path。
2. **chunk select -> candidates -> sort small**：避免 full sort。
3. **value+index pair reduction**：比较时携带 index 和 tie-break。
4. **two-stage selection**：block 局部 topk，第二阶段 merge。
5. **radix/select/threshold**：大规模选择先减少候选。

Autotune 候选：

- `BLOCK_N`: 128, 256, 512, 1024, 2048
- `CANDIDATES`: 16, 32, 64, 128
- `K_TILE`: 1, 4, 8, 16, 32
- `num_warps`: 4, 8, 16

风险：stable、largest/smallest、sorted、NaN、tie-break、index dtype、empty dim、候选裁剪正确性。

### 9.8 Scatter / Gather / Index Add / Routing

失败信号：

- atomic 写冲突严重。
- gather 随机访问导致 L2 hit 低。
- routing/bucket 需要多次全局同步或大量临时 tensor。
- deterministic 与 atomic path 冲突。

优先 trial：

1. **index distribution benchmark**：random/local/repeated/hotspot 分开。
2. **local aggregation before global write**：CTA 内聚合重复 index。
3. **bucket/sort/group**：提高 locality，减少 atomic 冲突。
4. **feature 维向量化**：table lookup/embedding dim 连续时沿 feature 维读写。
5. **flatten index rank**：高 rank index 不污染主 kernel。
6. **deterministic 分离**：固定归约顺序 path 与 fast atomic path 分开。

Autotune 候选：

- `BLOCK_INDEX`: 64, 128, 256, 512
- `BLOCK_FEATURE`: 16, 32, 64, 128, 256
- `ROWS_PER_PROGRAM`: 1, 2, 4
- `num_warps`: 4, 8

风险：index 越界、负 index、重复 index、atomic 精度、deterministic、sparse/dense 输出。

### 9.9 Embedding / Table Lookup

失败信号：

- forward gather 带宽低。
- small embedding_dim launch/地址开销高。
- large embedding_dim 单行拆分不足。
- 随机 index 与热点 index 性能差异大但 benchmark 未区分。
- `max_norm`、`padding_idx`、backward 语义混进 forward fast path。

优先 trial：

1. **forward gather 与 renorm/backward 分离**：普通 gather 不隐藏写入逻辑。
2. **按 embedding_dim 分段**：small-D 多行合并，large-D 拆列块。
3. **沿 D 向量化**：weight 行连续时 feature 维连续读写。
4. **index locality active set**：random、local repeated、hotspot 分开记录。
5. **padding 语义 audit**：forward 是否读取真实 `weight[padding_idx]` 按参考 API。
6. **backward 单独设计**：dense grad 是 scatter-add，sparse grad 改变返回形态。

Autotune 候选：

- `BLOCK_INDEX`: 16, 32, 64, 128
- `BLOCK_D`: 16, 32, 64, 128, 256
- `ROWS_PER_PROGRAM`: 1, 2, 4
- `num_warps`: 4, 8

风险：`padding_idx`、`max_norm`、`norm_type`、`scale_grad_by_freq`、`sparse`、index dtype、out shape。

### 9.10 Backward / Training Path

失败信号：

- forward 快但训练整体无收益。
- backward 复用 forward tile 后寄存器/访存不合适。
- grad accumulation atomic 冲突高。
- 保存中间值导致显存/HBM 压力大。

优先 trial：

1. **grad path 分离**：grad_input、grad_weight、grad_bias 独立。
2. **保存 vs 重算实验**：显存、HBM、寄存器、数值误差对比。
3. **atomic accumulation 优化**：局部聚合、分桶、two-stage。
4. **deterministic path 分流**：排序或固定归约顺序，不与 fast path 混合。
5. **dtype 梯度精度**：低精度输入通常 fp32 累加，输出 grad dtype 对齐参考。

Autotune 候选：

- `<op>_bwd_input`、`<op>_bwd_weight`、`<op>_bwd_bias` 分配置名。
- atomic path 调 `BLOCK_*`、local aggregation 粒度。
- recompute path 调 tile 与重算粒度。

风险：requires_grad 组合、非连续 grad、deterministic、mixed precision、NaN/Inf 反传。

## 10. 微观优化工具箱

这些工具只能在 profile 或明确假设支持下使用。

| 技巧 | 适用 | 风险 |
| --- | --- | --- |
| Loop peeling | 主循环规则，只有尾块 mask | 需测试整除/非整除 |
| Rematerialization | 长循环后仍需索引/offset | 过度重算增加整数指令 |
| Branch-local pointer construction | 复杂地址只在分支内需要 | 避免 lane-level divergence |
| `tl.max_contiguous` / `tl.multiple_of` | host 已保证连续/对齐 | 假设不真会错 |
| `make_block_ptr` | 规则 2D/3D tile、复杂 stride | 小 pointwise 可能过重 |
| Program ID swizzle | 相邻 tile 共享输入 | crossing point 需重测 |
| Cache policy | 只读热点、双遍、partial buffer | 架构敏感，需目标设备验证 |
| Base pointer hoisting | batch/group/row base 固定 | 别破坏 alias/stride 语义 |
| Mask simplification | 1D fast path 可替代多维 mask | 需保证边界不越界 |
| Independent accumulators | execution dependency 高 | 寄存器可能上升 |

## 11. Autotune 深度规则

### 11.1 配置拆分

默认拆独立配置名：

- 不同算法族：loop、atomic、two-stage、persistent、online、split-K。
- 不同 dtype 族：fp16/bf16、fp32、fp64、complex。
- 不同 workload 段：tiny、small、medium、large、wide、global。
- 不同 layout：contiguous、channels-last、strided/general。

### 11.2 key 设计

key 只保留改变 best config 的维度：

- M/N/K、输出规模、归约长度、embedding dim、window size。
- 连续性、layout、退化结构标志。
- dtype family，前提是确实改变 best config。

不要放纯语义参数，例如 `eps`、training 标志、异常模式。它们应由 host dispatch 或 `do_not_specialize` 管理。

### 11.3 候选空间收敛

候选空间先覆盖合理范围，再根据 active benchmark 裁剪。不要为了单个 shape 加入大量离群配置。

每次修改候选空间记录：

- 配置名。
- 新增/删除参数。
- 触发的 active shape。
- best config 是否变化。
- 是否引入编译或冷启动成本。

## 12. 收缩覆盖面而不是 torch fallback

若某 path 经过 deep optimization 仍不达标，按以下顺序处理：

1. **拆 path**：把可赢的 fast/special path 保留，失败 shape 退出自研。
2. **标记 unsupported**：显式报错并测试，不调用 torch。
3. **项目批准 external-library path**：非 torch API，单独标注，不算自研 kernel 成绩。
4. **重新定义 active set**：仅当用户/项目确认该 workload 不再是目标，且必须重建 baseline。
5. **停止自研**：说明原因、证据和残余风险。

禁止把失败 path 静默 fallback 到 `torch` 来让功能测试通过。

## 13. Deep Opt 最小努力标准

在宣称“无法继续优化 / 建议停止”之前，至少完成：

- 一次 metric/benchmark 口径复核。
- 一次生产路径 no-torch audit。
- 一次 dispatch path 复核。
- 一次 autotune 接入复核。
- 一次 profiler 证据；不可用时写明限制和替代证据。
- 至少两个与算子家族匹配的策略 trial，除非 profiler 已证明无意义。
- 一次覆盖面收缩评估。

## 14. Deep Opt 交付结论模板

```text
Deep optimization summary:
- Trigger: 哪些 active shape 未达标，原始 SpeedUp 是多少
- Audit: benchmark 口径、dispatch path、autotune、no torch compute fallback 审计结果
- Family: 算子家族与主瓶颈判断
- Profile evidence: nsys/ncu 或替代证据
- Tried: 每轮策略、change、validation、benchmark、keep/revert
- Tuned configs: 修改过的配置名、key、候选参数、best config
- Result: 最终 SpeedUp、active set 回归、是否达到目标
- Coverage decision: 保留哪些自研 path，哪些 unsupported/external-library/停止
- Stop reason: 达标、收益不足、维护复杂度、厂商库优势、profiler 限制等
- Residual risk: 未覆盖 dtype/shape/layout/training/path
```

## 15. 一句话准则

Deep optimization 不是“继续调参数”，而是强制 AI 用证据把失败 shape 拆成可解释的瓶颈，再按算子家族逐轮验证高级策略；如果仍无法稳定赢，就收缩自研覆盖面或显式不支持，绝不把 torch 计算 fallback 当作完成方案。
