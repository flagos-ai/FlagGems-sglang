# 流程与命令

以下为分阶段操作模板。变量由当前任务设置，路径指向相应仓库与已验证的运行环境。示例以已经接入的 `mrotary_embedding` 为入口；换算子时同时替换文件、marker、公共 op_name、problem ID 和 baseline recipe，名称不一定相同。

## 1. 对齐 workload 与 ABI

从主文档及子 workload 文档提取逐调用表，记录文档版本、模型、输入布局、dtype、标量、输出和副作用。区分文档 core 与新增边界案例。映射到 Definition、正确性参数、benchmark 内层 case 和 baseline key，逐条核对配置，不能只比较总数或各参数边际集合。

每个算子提供 `tests/test_<op>.py`、`benchmark/test_<op>.py` 和 `definitions/<op>.json`，尽量共享 workload/input builder，正确性包括 benchmark core 与必要边界。保留 shape、dtype、stride、storage offset、None、mutation、alias、只读输入及未选中状态的实际 ABI。区分合成输入与真实 dump；随机种子相同不保证跨设备 tensor 相同。

## 2. 开发 reference 与测试三件套

reference 优先使用共享 Torch 组网。正确性显式调用 `get_candidate(op_name, reference=reference)`；benchmark 通过 `torch_op` 声明同语义 reference，由公共 runner 选择 candidate。保持默认、override 和 dummy 行为兼容，避免在解析 override 前访问可能不存在的导出。

精度阈值按算子、dtype、输出制定，在 pytest 与 KS3 manifest 中保持语义一致；结构、索引与副作用单独检查，不按机器临时放宽阈值。

参考仓库已有 `tests/test_mrotary_embedding.py`、`benchmark/test_mrotary_embedding.py` 与 `definitions/mrotary_embedding.json`，以及当前 `src/flaggems_sglang/reference_candidate.py` 的接口。

依次单独运行这些检查（从 SGLang 仓库根目录；已配置其真实设备运行时）：

```sh
# 默认行为、case 列举；保存收集结果并核对完整参数配置。
pytest tests/test_mrotary_embedding.py -m mrotary_embedding --collect-only -q
pytest benchmark/test_mrotary_embedding.py -m mrotary_embedding --level core --list-cases
pytest tests/test_mrotary_embedding.py -m mrotary_embedding

# dummy 只验证触达，不产生有效的数值/性能结果。
pytest tests/test_mrotary_embedding.py -m mrotary_embedding \
  --dummy-candidate mrotary_embedding --dummy-report "$RUN_DIR/dummy-correctness.json"
pytest benchmark/test_mrotary_embedding.py -m mrotary_embedding --level core \
  --dummy-candidate mrotary_embedding --dummy-report "$RUN_DIR/dummy-benchmark.json"

# 显式 reference 替换候选，correctness 和 benchmark 使用公共机制。
pytest tests/test_mrotary_embedding.py -m mrotary_embedding --ref-as-candidate
pytest benchmark/test_mrotary_embedding.py -m mrotary_embedding --level core \
  --ref-as-candidate --metrics latency --base-data "$BASELINE_JSON"

# 有实际候选后，另跑真正的 candidate 检查与性能。
pytest tests/test_mrotary_embedding.py -m mrotary_embedding \
  --override "mrotary_embedding:$CANDIDATE_FILE:$CANDIDATE_FUNCTION"
pytest benchmark/test_mrotary_embedding.py -m mrotary_embedding --level core \
  --override "mrotary_embedding:$CANDIDATE_FILE:$CANDIDATE_FUNCTION" \
  --metrics latency --base-data "$BASELINE_JSON"
```

这些模式分别执行；不要将 `--ref-as-candidate` 与 dummy、override 或 list/collect 选项混用。仅在测试支持时用 `--ref cpu` 检查 CPU reference 路径，并检查实际输出 device/alias/变更语义；本例 mRoPE 明确不支持该模式。默认没有公开 candidate 可能按既有行为 skip；这不算数值通过。review 接线时同时确认缺少显式 reference 会明确失败。

检查 dummy 报告中的每个 pytest case 和每个 benchmark 内层 case 是否实际触达。marker 选择的是 pytest 节点，一个 benchmark 节点可包含多个计时配置；整个会话调用总数大于零不能证明每个配置都覆盖。计时验收需要实际完成，case listing 不能代替计时。

## 3. NVIDIA 原生入口及固定 golden

确认原始 SGLang 入口实际属于 CUDA、FlashInfer、SGLang kernel 或 Triton，固定源码、依赖版本、轮子摘要和实际 GPU；需要时建立独立 SGLang 环境。native 入口缺失应失败，不得静默回退 Torch oracle。

在 `verification/problems/<group>/<problem>/` 提供 cases、native 和目标 reference adapter，并按当前 op-verify schema 定义验证契约。按第 1 节 ABI 定义返回及变更契约，mutation 的输入在调用前保存。对没有顺序保证的 TopK，可同时按 ID 重排权重和索引后比较，不能分别排序破坏配对。

在 H800 上先验证 Torch reference 与原始 SGLang 的一致性。mRoPE 的 NVIDIA `get_reference()` 返回原生 Triton，普通 `--ref-as-candidate` 是自比较；下面显式注入 portable Torch candidate，而 expected 保持原生 Triton。其他算子也先确认两端实现的来源，再选择独立验证路径。

若数据集只导出 core，保留全面 correctness/边界检查作为独立证据，报告两者不同的覆盖范围。

```sh
# SGLANG_SOURCE 是相应版本的 FlagGems-sglang checkout。
export PYTHONPATH="$SGLANG_SOURCE/src:$SGLANG_SOURCE/verification${PYTHONPATH:+:$PYTHONPATH}"
export OP_VERIFY_PROBLEMS_DIR="$SGLANG_SOURCE/verification/problems"
# 使用新任务的唯一 prefix；所有参与机随后使用同一个值。
export OP_VERIFY_KS3_PREFIX="$GOLDEN_PREFIX"

# 在已确认的 H800 原生环境、SGLang 仓库根目录运行。
# expected = 原生 Triton；candidate = 显式 portable Torch（不叠加 --ref-as-candidate）。
pytest tests/test_mrotary_embedding.py -m mrotary_embedding \
  --override "mrotary_embedding:$SGLANG_SOURCE/src/flaggems_sglang/reference/mrotary_embedding_torch.py:reference"

# 独立精度验证通过后，从 native 入口离线导出 golden。
op-verify export -p rope/mrotary_embedding --reference-source native \
  --compress --output-dir "$GOLDEN_BUNDLE"

# 原样传输 bundle 后，在可访问 KS3 的机器上传。
op-verify upload --input-dir "$GOLDEN_BUNDLE"
```

核对 op-verify 是否支持 `--compress` 和所需契约版本。凭证通过私有配置或安全环境注入，不进入代码、命令参数或日志。记录 manifest/payload 摘要、prefix 和原生入口 provenance；上传已有 bundle 不重新生成 golden，不覆盖历史 prefix。

## 4. 各后端重放并验证 reference

在指定目标上检查实际设备、Python、厂商 Torch 插件与设备可见性，按现有运行时加载环境。

国产后端 adapter 复用 `get_reference`。NVIDIA replay 则遵循第 3 节的独立性要求，检查 `verification/problems/rope/mrotary_embedding/reference_torch.py` 实际选择的实现。

设置与导出相同的 problems/source/KS3 prefix，以及任务独立的缓存目录，再运行：

```sh
export OP_VERIFY_CACHE_DIR="$RUN_DIR/ks3-cache"
op-verify check -p rope/mrotary_embedding --device "$TORCH_DEVICE"
```

`TORCH_DEVICE` 必须是该 SDK 实际支持的设备字符串，例如 Ascend 的 `npu`、MUSA 的 `musa`；某些 SDK 兼容 `cuda`。确认 op-verify 的平台检测支持目标，不能仅修改字符串冒充已接入。

核对下载 manifest 与输入/输出 payload 摘要、精度策略、每 case 比较结果、worker 错误及 skip，不只看 shell 退出码。固定 replay 不重新随机生成输入。

共享 Torch reference 失败时，先区分输入/设备搬运/布局/比较错误与数学 API 缺失；确需特化时，在 `src/flaggems_sglang/reference/vendors/_<vendor>/<op>.py` 定义 `reference`，通过既有分发接入并在同一批 NVIDIA 输入上重验，不另写 replay 专用数学实现。fixture 可针对 SDK 缺少随机排列等能力在 CPU 构造后搬到目标，但不能把数学 reference 悄悄搬到 CPU，或把 fixture 开销计入 GPU kernel 时间。

若需要显式 reference-only 导入模式，先确认对应实现已存在，仅将其用于 reference 或明确注入 candidate 的验证；它不证明内置优化候选可用。

## 5. H800 性能基线与本地 candidate 计时

在 ops-baseline 添加/复用 `ops/<repo>/<op>.py`：调用原始 SGLang，实现 dtype、输入、配置及唯一 key。source 与 target 使用相同 case contract，key 包括完整几何布局和配置，避免 shape 相同但 stride/offset 或标量不同的记录混用。

```sh
# 从 ops-baseline checkout、H800 原生 SGLang 环境执行。
export FLAGGEMS_SGLANG_SOURCE="$SGLANG_SOURCE"
python tools/collect_baseline_nvidia.py --repo sglang --ops mrotary_embedding \
  --device "$GPU_INDEX" --reference-chip H800 --report-dir "$NCU_REPORT_DIR" \
  --output "$BASELINE_JSON"
```

新 Qwen3.6 recipes 使用独立的 `--repo sglang_qwen36`，不要误用旧同名 `sglang` profiles。`--reference-chip H800` 只是缩放参考配置，仍需检查实际 collection GPU 是 H800。

需要资源折算时保留 NCU，不能只用 `--no-ncu` 延迟采集声称完成。检查每 case 延迟、有效 NCU 结果、瓶颈、单位和真实 `_environment`，保留 raw reports/软件与源码版本。不要将合并后的静态 snapshot 当作 resume 输入。

将 artifact 保存到 ops-baseline，经目标 benchmark 的 `--base-data` 消费并实测本地 candidate；对精确 key miss 明确报告，不能靠 shape-only fallback 填补。

现有模型中 `vs Base = (H800 latency / target latency) / factor`，factor 为 target/reference 的瓶颈资源能力比例。记录实际芯片、资源、factor 来源。缺少对应资源的 factor 时仅报告 raw ratio/factor=None，不能借邻近型号、假造 factor 或称为已归一化。共享机器受干扰的时间不作为正式性能发布依据。

## 6. PR 与可复核结果

按算子×机器汇总 collected、golden passed/failed/skipped、默认/CPU reference、dummy 逐 case 触达、reference-as-candidate、实际 override、实际 core 计时、精确 baseline 命中与缩放状态。PR 写明 workload 来源、阈值理由、环境与源码版本、KS3 prefix/摘要、静态 baseline 和限制。保留命令、原始日志与失败诊断，公开材料不含凭证或私有接入信息；只清理本任务拥有且已结束的进程/连接。

说明依赖 PR 和未验证项；引用历史结果时标明其源码与覆盖范围，未经用户要求不合并 PR。
