import os
import glob
import subprocess
import time
import json
from datetime import datetime


# ================= 配置区 =================

# 目标算子列表 (白名单)
# 如果列表为空 []，则默认执行目录下所有的 test_*.py
# 如果填入算子名（如 "batch_norm", "relu"），则只执行这些算子对应的测试
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

TEST_DIR = "tests"  # 测试文件所在目录
LOG_DIR = "test_logs"  # 单个测试日志的存放目录
REPORT_FILE = "test_summary.json"  # 最终汇总报告的文件名

# ==========================================


def get_operator_name(filename):
    """从文件名中提取算子名，例如 test_batch_norm.py -> batch_norm"""
    basename = os.path.basename(filename)
    if basename.startswith("test_") and basename.endswith(".py"):
        return basename[5:-3]
    return basename


def main():
    os.makedirs(LOG_DIR, exist_ok=True)

    # 收集并过滤测试文件
    all_test_files = sorted(glob.glob(os.path.join(TEST_DIR, "test_*.py")))
    if not all_test_files:
        print("未找到任何 test_*.py 文件。")
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
        print("🔍 未设置过滤，将执行所有测试。")

    if not test_files:
        print(
            "过滤后没有需要执行的测试文件，请检查 TARGET_OPERATORS 是否拼写正确。"
        )
        return

    print(f"🚀 共发现 {len(test_files)} 个待测文件，开始提交 yhrun 任务...\n")
    print("-" * 60)

    summary = {
        "total": len(test_files),
        "passed": 0,
        "failed": 0,
        "errored_or_interrupted": 0,
        "details": [],
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    start_time_total = time.time()

    # 逐个执行测试
    for idx, file_path in enumerate(test_files, 1):
        file_name = os.path.basename(file_path)
        log_file = os.path.join(LOG_DIR, f"{file_name}.log")

        print(
            f"[{idx}/{len(test_files)}] 正在测试: {file_name:<30}",
            end="",
            flush=True,
        )

        # 构建 yhrun 命令
        cmd = [
            # "yhrun",
            # "-p", "h100x",
            # "-G", "1",
            "python3",
            "-m",
            "pytest",
            "-v",
            "-s",
            "--ref=cpu",
            file_path,
        ]

        start_time = time.time()

        # 启动子进程并捕获输出
        result = subprocess.run(cmd, capture_output=True, text=True)
        duration = time.time() - start_time

        # 分析退出状态码 (Pytest return codes)
        if result.returncode == 0:
            status = "✅ PASS"
            summary["passed"] += 1
        elif result.returncode == 1:
            status = "❌ FAIL"
            summary["failed"] += 1
        elif result.returncode == 5:
            status = "⚠️ NO TESTS"
        else:
            status = f"💥 ERROR (Code: {result.returncode})"
            summary["errored_or_interrupted"] += 1

        print(f" -> {status} ({duration:.2f}s)")

        # 将标准输出和错误写入独立的日志文件
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"=== Command: {' '.join(cmd)} ===\n")
            f.write(f"=== Status: {status} ===\n")
            f.write(f"=== Duration: {duration:.2f}s ===\n\n")
            f.write("--- STDOUT ---\n" + result.stdout + "\n")
            if result.stderr:
                f.write("--- STDERR ---\n" + result.stderr + "\n")

        # 记录汇总信息
        summary["details"].append(
            {
                "file": file_name,
                "status": status.strip("✅❌⚠️💥 "),
                "return_code": result.returncode,
                "duration_seconds": round(duration, 2),
                "log_path": log_file,
            }
        )

    # 生成报告与控制台汇总
    summary["total_duration_seconds"] = round(
        time.time() - start_time_total, 2
    )

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print("-" * 60)
    print("📊 任务执行完毕！")
    print(
        f"总计: {summary['total']} | "
        f"通过: {summary['passed']} | "
        f"失败: {summary['failed']} | "
        f"异常中断: {summary['errored_or_interrupted']}"
    )
    print(f"总耗时: {summary['total_duration_seconds']} 秒")
    print(f"详细日志已保存至 '{LOG_DIR}' 目录。")


if __name__ == "__main__":
    main()
