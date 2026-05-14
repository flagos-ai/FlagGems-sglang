import os
import glob
import subprocess
import time
import json
import re
from datetime import datetime


# ================= 配置区 =================

# 目标算子列表 (白名单)
# 例如: ["relu", "add"]。如果留空 []，则自动测试目录下所有的 test_*_perf.py
TARGET_OPERATORS = [
    "relu",
    "gelu",
    "silu",
    "leaky_relu",
    "prelu",
    "softmax",
    "batch_norm",
    "layer_norm",
    "rms_norm",
    "group_norm",
    "max_pool2d",
    "avg_pool2d",
    "adaptive_avg_pool2d",
    "adaptive_max_pool2d",
    "add",
    "sub",
    "mul",
    "div",
    "pow",
    "sqrt",
    "abs",
    "neg",
    "clamp",
    "sum",
    "mean",
    "prod",
    "cumsum",
    "cumprod",
    "eq",
    "ne",
    "max_pool1d",
    "max_pool3d",
    "avg_pool1d",
    "avg_pool3d",
    "adaptive_avg_pool1d",
    "adaptive_avg_pool3d",
    "adaptive_max_pool1d",
    "adaptive_max_pool3d",
    "threshold",
    "threshold_",
    "hardtanh",
    "hardtanh_",
    "hardswish",
    "relu6",
    "elu",
    "elu_",
    "selu",
    "celu",
    "leaky_relu_",
    "rrelu",
    "rrelu_",
    "glu",
    "logsigmoid",
    "tanh",
    "mish",
    "softsign",
    "softplus",
    "softmin",
    "softshrink",
    "mv",
    "mm",
    "dot",
    "embedding",
    "conv1d",
    "conv2d",
]

TEST_DIR = "benchmark"  # 性能测试文件所在目录
LOG_DIR = "perf_logs"  # 单个测试日志的存放目录

# 运行状态汇总，例如 total / passed / failed / details
REPORT_FILE = "perf_summary.json"

# 核心性能数据，例如 operator / dtype / shape / speedup
DATA_FILE = "perf_data.json"

# ==========================================


def get_operator_name(filename):
    """从文件名中提取算子名，例如 test_relu_perf.py -> relu"""
    basename = os.path.basename(filename)
    if basename.startswith("test_") and basename.endswith("_perf.py"):
        return basename[5:-8]
    return basename


def parse_float(value):
    """
    安全解析 float。
    支持普通小数、科学计数法、nan、inf。
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_meta_args(meta_text):
    """
    解析 Operator 行括号里的参数。

    输入示例：
        dtype=torch.float16, mode=kernel, level=comprehensive

    输出：
        {
            "dtype": "torch.float16",
            "mode": "kernel",
            "level": "comprehensive"
        }
    """
    meta = {}

    if not meta_text:
        return meta

    for item in meta_text.split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue

        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()

        if key:
            meta[key] = value

    return meta


def short_dtype(dtype):
    """
    torch.float16 -> float16
    torch.bfloat16 -> bfloat16
    float16 -> float16
    """
    if not dtype:
        return "unknown"

    dtype = str(dtype).strip()

    if dtype.startswith("torch."):
        return dtype[len("torch.") :]

    return dtype


def parse_legacy_operator_dtype(operator_name):
    """
    兼容老格式：
        Operator: relu_fp16 Performance Test

    如果日志里没有 dtype=(...)，但 operator 名字带了后缀，
    可以尽量拆成：
        operator = relu
        dtype = torch.float16
    """
    legacy_map = {
        "fp16": "torch.float16",
        "float16": "torch.float16",
        "bf16": "torch.bfloat16",
        "bfloat16": "torch.bfloat16",
        "fp32": "torch.float32",
        "float32": "torch.float32",
        "fp64": "torch.float64",
        "float64": "torch.float64",
        "int32": "torch.int32",
        "int64": "torch.int64",
        "bool": "torch.bool",
    }

    for suffix, dtype in legacy_map.items():
        marker = "_" + suffix
        if operator_name.endswith(marker):
            return operator_name[: -len(marker)], dtype

    return operator_name, None


def parse_perf_output(stdout_text):
    """
    从 pytest 的输出中解析出性能数据。

    重点改动：
    1. 从 Operator 行解析 dtype / mode / level。
    2. 每一条 SUCCESS 记录都带上当前 dtype。
    3. 这样 perf_data.json 后续就可以按 dtype 分组统计平均 speedup。
    """

    records = []

    current_context = {
        "operator": None,
        "dtype": "unknown",
        "dtype_short": "unknown",
        "mode": None,
        "level": None,
    }

    for line in stdout_text.splitlines():
        line = line.rstrip("\n")

        # 例子：
        # Operator: conv2d  Performance Test
        # (dtype=torch.float16, mode=kernel, level=comprehensive)
        op_match = re.search(
            r"Operator:\s*(?P<operator>[A-Za-z0-9_]+)\s*"
            r"Performance Test"
            r"(?:\s*\((?P<meta>[^)]*)\))?",
            line,
        )

        if op_match:
            raw_operator = op_match.group("operator")
            meta_text = op_match.group("meta") or ""
            meta = parse_meta_args(meta_text)

            operator, legacy_dtype = parse_legacy_operator_dtype(raw_operator)

            dtype = meta.get("dtype") or legacy_dtype or "unknown"
            mode = meta.get("mode")
            level = meta.get("level")

            current_context = {
                "operator": operator,
                "dtype": dtype,
                "dtype_short": short_dtype(dtype),
                "mode": mode,
                "level": level,
            }
            continue

        # 只解析 SUCCESS 行
        if not line.startswith("SUCCESS"):
            continue

        current_op = current_context.get("operator")
        if not current_op:
            continue

        parts = line.split()

        # SUCCESS TorchLatency GemsLatency Speedup TorchGBPS GemsGBPS ...
        if len(parts) < 4:
            continue

        torch_latency = parse_float(parts[1])
        gems_latency = parse_float(parts[2])
        speedup = parse_float(parts[3])

        if torch_latency is None or gems_latency is None or speedup is None:
            continue

        record = {
            "operator": current_context["operator"],
            "dtype": current_context["dtype"],
            "dtype_short": current_context["dtype_short"],
            "torch_latency": torch_latency,
            "gems_latency": gems_latency,
            "speedup": speedup,
        }

        # mode / level 不是所有日志都有，有就写入，没有就不写
        if current_context.get("mode") is not None:
            record["mode"] = current_context["mode"]

        if current_context.get("level") is not None:
            record["level"] = current_context["level"]

        # 如果日志中有 GBPS 数据，也一并提取
        if len(parts) >= 6:
            torch_gbps = parse_float(parts[4])
            gems_gbps = parse_float(parts[5])

            if torch_gbps is not None and gems_gbps is not None:
                record["torch_gbps"] = torch_gbps
                record["gems_gbps"] = gems_gbps

        # 提取 shape 信息
        # 例子：
        # [torch.Size([32, 3, 224, 224]), torch.Size([64, 3, 7, 7]),
        # None, 2, 3, 1, 1]
        size_match = re.search(r"(\[torch\.Size.*\])", line)
        if size_match:
            record["size_detail"] = size_match.group(1)

        records.append(record)

    return records


def main():
    os.makedirs(LOG_DIR, exist_ok=True)

    # 收集并过滤测试文件
    all_test_files = sorted(
        glob.glob(os.path.join(TEST_DIR, "test_*_perf.py"))
    )

    if not all_test_files:
        print(f"未在 {TEST_DIR} 目录下找到任何 test_*_perf.py 文件。")
        return

    test_files = []

    if TARGET_OPERATORS:
        for f in all_test_files:
            op_name = get_operator_name(f)
            if op_name in TARGET_OPERATORS:
                test_files.append(f)

        print(f"🔍 已启用算子过滤，目标算子数量: {len(TARGET_OPERATORS)}")
    else:
        test_files = all_test_files
        print("🔍 未设置过滤，将执行所有性能测试。")

    if not test_files:
        print(
            "过滤后没有需要执行的测试文件，请检查 TARGET_OPERATORS 是否拼写正确。"
        )
        return

    print(f"🚀 共发现 {len(test_files)} 个待测性能文件，开始提交测试任务...\n")
    print("-" * 60)

    summary = {
        "total": len(test_files),
        "passed": 0,
        "failed": 0,
        "errored_or_interrupted": 0,
        "details": [],
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # 存储所有算子的所有性能测试记录
    all_perf_data = []

    start_time_total = time.time()

    for idx, file_path in enumerate(test_files, 1):
        file_name = os.path.basename(file_path)
        log_file = os.path.join(LOG_DIR, f"{file_name}.log")

        print(
            f"[{idx}/{len(test_files)}] 正在测速: {file_name:<35}",
            end="",
            flush=True,
        )

        # 构建命令
        # 如果你需要 yhrun，把下面注释打开即可。
        cmd = [
            # "yhrun",
            # "-p", "h100x",
            # "-G", "1",
            "python3",
            "-m",
            "pytest",
            "-v",
            "-s",
            file_path,
        ]

        start_time = time.time()

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        duration = time.time() - start_time

        # 分析退出状态码
        if result.returncode == 0:
            status = "✅ PASS"
            summary["passed"] += 1
        elif result.returncode == 1:
            status = "❌ FAIL"
            summary["failed"] += 1
        elif result.returncode == 5:
            status = "⚠️ NO TESTS"
            summary["errored_or_interrupted"] += 1
        else:
            status = f"💥 ERROR (Code: {result.returncode})"
            summary["errored_or_interrupted"] += 1

        print(f" -> {status} ({duration:.2f}s)")

        # ==== 核心解析步骤 ====
        extracted_data = parse_perf_output(result.stdout)
        all_perf_data.extend(extracted_data)

        # 将标准输出和错误写入独立日志文件
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"=== Command: {' '.join(cmd)} ===\n")
            f.write(f"=== Status: {status} ===\n")
            f.write(f"=== Duration: {duration:.2f}s ===\n\n")
            f.write("--- STDOUT ---\n")
            f.write(result.stdout)
            f.write("\n")

            if result.stderr:
                f.write("--- STDERR ---\n")
                f.write(result.stderr)
                f.write("\n")

        # 当前测试文件中解析到的 dtype 集合
        dtypes_collected = sorted(
            {item.get("dtype", "unknown") for item in extracted_data}
        )

        # 当前测试文件中解析到的算子集合
        operators_collected = sorted(
            {item.get("operator", "unknown") for item in extracted_data}
        )

        summary["details"].append(
            {
                "file": file_name,
                "operator": get_operator_name(file_name),
                "status": status.strip("✅❌⚠️💥 "),
                "return_code": result.returncode,
                "duration_seconds": round(duration, 2),
                "log_path": log_file,
                "data_points_collected": len(extracted_data),
                "dtypes_collected": dtypes_collected,
                "operators_collected": operators_collected,
            }
        )

    summary["total_duration_seconds"] = round(
        time.time() - start_time_total,
        2,
    )

    # 写运行状态汇总
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    # 写性能明细数据
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(all_perf_data, f, indent=4, ensure_ascii=False)

    print("-" * 60)
    print("📊 性能测试执行完毕！")
    print(
        f"总计脚本: {summary['total']} | "
        f"通过: {summary['passed']} | "
        f"异常: {summary['failed'] + summary['errored_or_interrupted']}"
    )
    print(
        f"共收集到 {len(all_perf_data)} 条性能数据记录，已保存至 {DATA_FILE}"
    )
    print(f"运行状态汇总已保存至 {REPORT_FILE}")
    print(f"总耗时: {summary['total_duration_seconds']} 秒")


if __name__ == "__main__":
    main()
