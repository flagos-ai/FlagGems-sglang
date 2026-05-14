# Optimization: Triton 算子设计原则与一阶段实现策略

## 0. 文档定位

本文档只回答三类问题：

1. 这个算子是否值得自研。
2. 第一版应该如何分型、拆路径、设计 host dispatch、kernel 家族和 autotune。
3. 如何在不破坏 `torch` 语义对齐的前提下完成基础性能闭环。

本文档不负责仓库路径、命令、当前任务输入或某次实验结果；这些内容只放在 `workflow.md`。当实现已经正确但性能仍未达标时，停止在本文档里继续找技巧，必须进入 `deep_opt.md`。

三份文档的唯一分工如下：

| 文档 | 负责内容 | 不负责内容 |
| --- | --- | --- |
| `optimization.md` | 通用原则、一阶段设计、路径拆分、host dispatch、基础 autotune 与验证思路 | 仓库命令、当前任务、二阶段深度攻关 |
| `workflow.md` | 仓库执行顺序、必须输出的表、文件落点、测试/benchmark/交付 gate | 解释每类算子的高级优化细节 |
| `deep_opt.md` | 性能不达标后的 profile-driven trial loop、高级策略、停止/收缩覆盖面判断 | 新算子的入门设计和仓库执行命令 |

## 1. 不可违背的硬约束

### 1.1 `torch` 是语义规格，不是生产 fallback

对标 `torch` 的自定义算子时，`torch` 对应 API 是语义规格和测试/benchmark 参考，不是生产实现的一部分。

生产路径包括：

- `src/flag_dnn/ops/<op>.py` 中的公开函数、host dispatch、helper、kernel launcher。
- 被该算子生产路径直接或间接调用的 runtime helper。
- fallback、unsupported path、异常路径和 dtype/layout 补洞路径。

生产路径中默认禁止任何 `torch` 计算算子参与目标结果计算，包括但不限于：

- `torch.<target_op>` 或语义等价的 `torch.<other_op>` 组合。
- `torch.nn.functional.*`。
- `torch.ops.aten.*`。
- `Tensor.<compute_method>`，例如 `x.matmul(...)`、`x.sum(...)`、`x.softmax(...)`、`x.contiguous()`、`x.clone()`、`x.to(...)`、`x.copy_(...)` 等会触发真实计算、拷贝或类型转换 kernel 的方法。

默认允许的 `torch` 用法仅限元信息和分配：

- 读取 `shape`、`stride`、`dtype`、`device`、`layout`、`requires_grad` 等元信息。
- `torch.empty`、`torch.empty_like`、`torch.empty_strided` 等未初始化输出分配。
- 纯 view/metadata 操作，前提是确认不会触发数据移动。

若某条路径暂时无法自研，优先选择以下处理，而不是调用 `torch`：

1. 显式 `raise NotImplementedError`，并在测试中覆盖该 unsupported scope。
2. 路由到本项目已有的非 `torch` kernel 或 runtime 实现。
3. 路由到项目明确允许的厂商库/外部库接口，并在交付结论中说明该路径不属于自研 kernel 性能结果。
4. 收缩自研覆盖面，只对已证明可赢的 path 开启自研。

只有任务输入显式写入 `ALLOW_TORCH_COMPUTE_FALLBACK = true` 时，才允许生产路径使用 `torch` 计算 fallback；默认值永远是 false。即便允许，该路径也必须单独标注、单独 benchmark，不能混入自研性能收益。

### 1.2 dtype 覆盖必须按目标 device 上的 `torch` 行为建立矩阵

不能只实现常用 dtype 后宣称完成。编码前必须列出目标 device/backend 上 `torch` 对应算子的实际 dtype 支持矩阵，包括：

- floating：`float16`、`bfloat16`、`float32`、`float64`。
- integer / bool。
- complex。
- dtype promotion、accumulate dtype、输出 dtype。
- 目标 device 不支持某 dtype 时的 skip 或报错语义。

如果本阶段只覆盖子集，交付结论必须写成“未完成 / 限定覆盖”，不能写成“与 torch 对齐”。

### 1.3 性能优化不能改变语义边界

不得为了速度放松以下行为：

- NaN / Inf / signed zero。
- tie-break 和 deterministic。
- empty tensor、非法输入、shape 检查、stride/layout 边界。
- accumulate precision、type promotion、输出 dtype。
- `out=`、alias、in-place、autograd 或训练路径边界。

### 1.4 benchmark 口径冻结后默认只读

一旦建立 baseline，以下内容默认只读：

- benchmark harness。
- validation harness。
- reference 生成逻辑。
- metric、unit、direction、aggregation。
- active set。

除非先重建 baseline，否则不能一边修改 benchmark 统计方式，一边声称实现变快。

## 2. 编码前必须冻结的信息

编码前至少写清以下事实；未知项必须写“未知 + 当前假设”。

| 类别 | 必填信息 |
| --- | --- |
| `torch` 契约 | 对应 API、官方参考、目标版本、合法输入、输出、dtype、promotion、NaN/Inf、异常、autograd |
| workload | 热点 shape、退化 shape、active benchmark set、目标阈值、聚合方式 |
| layout | contiguous、channels-last、strided、packed、是否允许预打包或重排 |
| dtype | 输入 dtype、accumulate dtype、输出 dtype、混合精度策略 |
| 场景 | inference / training、单次调用 / 高频重复调用、是否要求 deterministic |
| 自研边界 | 哪些 path 自研，哪些 path unsupported，哪些 path 可用项目外部库 |
| fallback policy | 默认 `NO_TORCH_COMPUTE_FALLBACK`；若有例外必须显式列出 |
| autotune | 每个 kernel path 的配置名、key、候选参数、配置来源、no-autotune 豁免理由 |
| 验证 | 功能测试矩阵、benchmark 命令、profile 工具可用性 |

## 3. 第一决策：是否值得自研

优先自研的条件：

- 原生实现缺失、明显过慢或无法覆盖目标语义。
- 多个操作可融合，能显著减少 HBM 往返、中间张量或 launch 数。
- workload 稳定，适合长期维护专门 fast path。
- 特殊 layout、窗口、分块或退化结构让通用库难以吃满硬件。
- 该路径是核心热点，收益能覆盖开发、autotune 和维护成本。

优先不自研或收缩覆盖面的条件：

- 本质是成熟 GEMM/Conv 大形状，强库已经长期更优，且没有融合收益。
- 路径低频、结构复杂、训练语义复杂或 dtype/layout 分叉过多。
- 需要每次做昂贵重排，且重排成本吃掉收益。
- 编译变体、autotune、workspace 或缓存成本会显著拖累上线体验。
- 尚未建立 `torch` 语义、dtype 支持和 benchmark 口径。

“不自研”不等于“调用 torch fallback”。默认做法是 unsupported、项目外部库、或只开启已证明有效的自研 fast path。

## 4. Workload 分型方法

编码前先把算子压缩成最小计算模型，再决定路径。

| 家族 | 主导维度 | 高频退化结构 |
| --- | --- | --- |
| pointwise / broadcast | 元素数、broadcast rank、stride、输入输出个数 | contiguous、no-broadcast、scalar、tiny |
| reduction | 输出规模、归约长度、归约轴连续性 | global、very-wide、small-output、last-dim |
| scan | scan 轴长度、行/列方向、是否连续 | short row、long row、strided column |
| softmax / norm | row length、统计量、是否最后一维 | small/medium/large N、inner/non-inner |
| pooling / window | window、stride、padding、输出空间 | global、divisible、small-output |
| matvec / GEMM-like / conv-like | M/N/K、batch、layout、dtype、Tensor Core | tiny M/N、wide K、1x1、depthwise |
| selection / sort / topk | 输入长度、K、是否 sorted/stable | k=1、小 K、arg-only |
| scatter / gather / index_add | index 分布、冲突率、feature dim | contiguous feature、重复 index、hotspot |
| embedding | num_indices、embedding_dim、index locality | small-D、large-D、padding、forward-only |
| backward / training | grad path、累加冲突、保存/重算 | grad_input、grad_weight、grad_bias 分离 |

如果不同 shape/dtype/layout 的最优调度明显不同，默认必须拆 path。不要试图让一个 general kernel 靠 autotune 同时解决算法选择、布局选择和退化结构。

## 5. 路径拆分规则

每个算子至少给出以下路径判断，即便某些路径最终 unsupported：

| 路径 | 含义 | 默认要求 |
| --- | --- | --- |
| early return | empty、identity、非法输入快速报错 | 不发射 kernel，不调用 torch compute |
| fast path | 高频、结构稳定、可赢的主路径 | host gate 清晰，kernel 简洁，benchmark 覆盖 |
| special path | 退化或特殊结构，例如 global、wide、tiny、1x1 | 不被 general path 吞掉 |
| general path | 低频但需要覆盖的复杂情况 | 保守实现或明确 unsupported |
| unsupported path | 本阶段不覆盖 | 显式报错，测试覆盖，不调用 torch compute |
| external-library path | 项目允许的非 torch 强库 | 单独标注，不混入自研 kernel 结论 |

## 6. Host dispatch 是算法的一部分

Host 侧负责便宜判断和算法路由，kernel 只负责已经选定的计算形态。

Host dispatch 至少负责：

- 维度规范化：负维转正、输出 shape、目标轴判断。
- layout gate：contiguous、channels-last、stride、是否需要 pack/cache。
- dtype gate：dtype 支持矩阵、accumulate dtype、输出 dtype。
- cheap early return：empty、identity、标量、小常数 case。
- path 路由：tiny / wide / global / special / general / unsupported。
- workspace：partial buffer、locks、scratch、packed weight。
- autotune：配置名、key、strategy、设备隔离、缓存控制。
- fallback 边界：禁止 torch compute fallback，unsupported 必须显式。

最小结构：

```python
def op(*inputs, **kwargs):
    spec = normalize_and_validate(*inputs, **kwargs)

    if spec.is_empty:
        return allocate_empty_result(spec)

    if spec.is_identity:
        return return_view_or_launch_copy_free_path(spec)

    if spec.unsupported:
        raise NotImplementedError(spec.reason)

    # Do not call torch.<op> or torch compute fallback here.
    if spec.is_tiny:
        return launch_tiny_path(spec)

    if spec.is_special_structure:
        return launch_special_path(spec)

    if spec.is_fast_contiguous:
        return launch_fast_path(spec)

    return launch_general_or_raise(spec)
```

如果性能不稳定，先审查 host dispatch 是否把目标 shape 送到了正确路径，再审查 kernel 参数。

## 7. Autotune 一阶段规则

### 7.1 Autotune 只在算法族已选定后使用

Autotune 不是算法选择器。先由 host dispatch 决定路径，再在路径内部搜索 tile、block、warps、stages、group、unroll 等参数。

### 7.2 每个可调 kernel path 必须有决策

每条会发射 Triton kernel 的路径都要写清：

- 配置名，例如 `<op>_contiguous`、`<op>_wide_reduce`、`<op>_tiny_fp64`。
- 可调参数：`BLOCK_*`、`TILE_*`、`GROUP_*`、`SPLIT_*`、`UNROLL`、`num_warps`、`num_stages`、`num_ctas`。
- key：只保留影响 best config 的性能维度。
- strategy：分箱、align、dtype family、layout flag。
- 配置来源：NVIDIA backend 默认从 `src/flag_dnn/runtime/backend/_nvidia/tune_configs.yaml` 读取。
- no-autotune 豁免理由。

### 7.3 允许 no-autotune 的情况

- 不发射 kernel 的 early return。
- 显式 unsupported path。
- 项目批准的外部库 path。
- 无可调参数的固定小 kernel。
- 已有 benchmark 证明固定参数足够稳定，且写入交付风险。

“不想先调”、“后续再补”、“只有一个 shape”不是合格理由。

### 7.4 key 设计

key 只放会改变性能最优配置的字段，例如：

- M/N/K、输出元素数、归约长度、embedding dim、window size。
- dtype family。
- layout / stride 方向 / contiguous flag。
- tiny / wide / global 等退化结构标志。

不要把纯语义参数打进 key，例如 `eps`、异常模式、只影响输出检查的 flag。它们应由 host dispatch、heuristics 或 `do_not_specialize` 处理。

## 8. Triton specialization 规则

### `tl.constexpr`

适合 constexpr 的参数：

- tile、block、window、unroll、算法开关。
- 改变循环层数、mask 形状、地址模式或控制流的参数。

不适合 constexpr 的参数：

- 高频变化但收益很小的标量。
- 只影响少量计算的语义参数。
- 会造成编译缓存爆炸的动态长度。

### `@triton.heuristics`

适合轻量结构判断，例如 `HAS_BIAS`、`IS_INNER`、`RETURN_INDICES`、`ONE_TILE_PER_CTA`。它不能替代 host dispatch。

### `do_not_specialize`

适合 `eps`、correction、动态 iteration、纯语义标量等高频变化但不值得编译成变体的参数。

### `make_block_ptr` / `boundary_check`

适合规则 2D/3D tile、复杂 stride、多布局、varlen 和需要清晰边界检查的路径。小型 contiguous pointwise 不要为了形式统一引入过重 block pointer。

### `tl.static_range`

适合固定窗口、固定小 K、固定 KH/KW。若循环边界稳定，优先静态展开，避免主循环保留动态分支。

## 9. 算子家族的一阶段选择

本节只用于第一版路径选择。性能不达标时进入 `deep_opt.md`。

### 9.1 Pointwise / Broadcast

- 必须优先做 contiguous flat fast path。
- no-broadcast 与 broadcast general path 分开。
- 高 rank broadcast 优先 rank-specialized generator，不要一个超泛化 kernel 吃全部。
- 多输入/多输出融合前评估寄存器和写回压力。

### 9.2 Reduction

- 输出多 + 中等 K：loop reduction。
- 输出少 + huge K：split reduction、two-stage 或 atomic，但必须评估冲突。
- value+index：比较时携带 index，不能 value 后回扫。
- mean/var/std/norm：优先合并统计，例如 Welford。
- 非连续归约轴：先判断 host 侧重排或特殊 stride path 是否值回成本。

### 9.3 Scan / Prefix-Sum

- 短行：row persistent 或一 program 多行。
- 长轴：多阶段 scan，避免单 kernel 硬吃。
- 行/列方向分流。
- 多阶段 path 要明确 partial buffer 和边界块。

### 9.4 Softmax / Norm

- 按 row length 拆 small / medium / large。
- last-dim contiguous 与 non-inner 分流。
- 长行 softmax 优先 online 或双遍。
- norm 类优先高精度中间统计和可合并统计。

### 9.5 Pooling / Window

- global pool 退化成 reduction。
- divisible window 静态化窗口与边界。
- small output 一个 program 处理多个输出点。
- NCHW / NHWC / channels-last 不混用同一地址计算。

### 9.6 MatVec / GEMM-like / Conv-like

- 成熟大 GEMM/Conv 先判断是否应自研；若不自研，不得调用 `torch.mm` / `torch.matmul` 作为 fallback。
- 自研优先覆盖融合、小 shape、特殊 layout、tiny M/N、wide K、1x1、depthwise、特殊 dtype。
- fp16/bf16 优先 Tensor Core 路径；fp32 明确 TF32/input_precision；fp64 保守或 unsupported。
- K 远大于 M/N 时考虑 split-K / Stream-K，但 workspace、锁和 deterministic 先设计。

### 9.7 Selection / TopK / Arg*

- `topk` 不等于 full sort。
- `k=1`、小 K、arg-only 单独 path。
- value/index 一体搬运，tie-break 与参考对齐。
- 大规模选择先局部候选，再二阶段 merge。

### 9.8 Scatter / Gather / Index Add / Routing

- 先评估 index locality 和冲突率。
- 冲突高时优先局部聚合、分桶或两阶段，而不是直接 atomic。
- gather 沿连续 feature 维向量化。
- deterministic path 与 fast atomic path 分离。

### 9.9 Embedding / Table Lookup

- forward gather、renorm、backward scatter/index-add 分离。
- 按 embedding_dim 和 index locality 分型。
- `padding_idx` 不等于 forward 自动置零；按参考语义处理。
- `max_norm` 是写入路径，不能藏在普通 gather kernel 的运行时分支里。
- backward dense/sparse 是独立问题，不能作为 forward 的附属开关。

### 9.10 Backward / Training

- backward 不默认复用 forward tile。
- grad_input、grad_weight、grad_bias 分开设计。
- 保存 vs 重算必须比较显存、HBM、寄存器和数值稳定。
- atomic grad accumulation 必须评估冲突和 deterministic。

## 10. 数值稳定与 dtype 规则

所有含归约、指数、范数、除法、方差、排序比较的算子都要检查：

- 是否需要提升到 fp32/fp64 累加。
- 是否需要 max-shift、Welford、Kahan 或其他稳定策略。
- mask lane 的哨兵值是否正确。
- NaN/Inf 是否与 `torch` 对齐。
- tie-break 是否明确。
- integer overflow、bool、complex 的语义是否与参考一致。
- 输出 dtype 和 type promotion 是否正确。

性能优化不得降低这些规则。

## 11. 验证矩阵

每个新算子默认覆盖：

- empty、single element、非 2 的幂、极短/极长轴、退化维度。
- contiguous fast path、非连续输入、特殊 layout path。
- 目标 device 上 `torch` 支持的全部 dtype。
- 输出 dtype / promotion / accumulate precision。
- NaN / Inf / signed zero / tie-break / deterministic。
- out / alias / in-place 边界，若接口暴露。
- forward / backward，若训练路径承诺支持。
- unsupported path 的明确报错。
- 生产实现中无 torch compute fallback 的静态审计。

## 12. Benchmark 与基础 trial loop

基础 benchmark 至少记录：

- torch baseline latency。
- 自研实现 latency。
- SpeedUp = baseline / implementation。
- cold-start 与 warm latency。
- kernel launch 数。
- workspace 成本。
- active set 与 aggregation。
- shape crossing point。

基础 trial loop：

1. 从未修改版本或当前 best 建立 baseline。
2. 一轮只验证一个主要假设。
3. build -> validate -> benchmark。
4. 记录结果再决定 keep/revert。
5. 功能失败立即回滚。
6. 连续 3 到 5 轮没有稳定收益，必须 profile。
7. 关键目标仍未达标，进入 `deep_opt.md`。

## 13. 交付契约

完成一个算子时，默认必须交付：

1. `torch` 对齐表。
2. dtype 支持矩阵。
3. workload 分型。
4. path 拆分表。
5. host dispatch 设计。
6. kernel 组织和数值策略。
7. autotune 配置或 no-autotune 豁免。
8. 测试矩阵和结果。
9. benchmark active set、aggregation、结果。
10. fallback / unsupported 边界，明确生产路径无 torch compute fallback。
11. 若进入 deep optimization，补充 deep opt summary。

## 14. 一句话准则

高性能算子的核心不是把一个 kernel 写得更复杂，而是先冻结 `torch` 语义和 dtype 契约，再用正确的 workload 分型、host dispatch、kernel specialization、autotune 和证据闭环决定哪些路径值得自研，哪些路径必须显式收缩或不支持。
