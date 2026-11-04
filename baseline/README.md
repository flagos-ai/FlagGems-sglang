# ops/sglang

存放 sglang 算子基准模块（方案 B）。目录结构与契约同 `ops/vllm`（见
`ops/vllm/__init__.py`）：每个算子一个 `ops/sglang/<OP_NAME>.py`，导出
`OP_NAME / DTYPES / IS_INPLACE / native() / grid() / build_inputs() / key_shape()`
（复杂算子再加 `config()`）。

采集时用 `--repo sglang` 指向本目录：

    python3 tools/collect_baseline_nvidia.py --repo sglang [--ops ...] [--device N]

缺省 `--repo vllm` 走 `ops/vllm`。本目录当前为空，待 sglang 算子落地后补充。
