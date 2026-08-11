import ast
import csv
import glob
import json
import re
import statistics
from pathlib import Path

records = Path("/root/autodl-tmp/actionpiece_records")
results_dir = records / "results"
results_dir.mkdir(parents=True, exist_ok=True)

eval_instances = 35598


def parse_duration(value):
    if not value:
        return None
    parts = [int(x) for x in value.split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    raise ValueError(f"Unknown duration: {value}")


def read_meta(path):
    data = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            data[key.strip()] = value.strip()
    return data


rows = []

for seed in range(2024, 2029):
    run_id = f"sports_full_seed{seed}_eval128"
    log_path = records / "logs" / f"{run_id}.log"
    meta_path = results_dir / f"{run_id}_meta.txt"
    config_path = results_dir / f"{run_id}_effective_config.txt"

    if not log_path.is_file():
        raise FileNotFoundError(log_path)
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    text = log_path.read_text(encoding="utf-8", errors="replace")
    text = text.replace("\r", "\n")
    meta = read_meta(meta_path)

    best_matches = re.findall(
        r"Best epoch:\s*(\d+),\s*Best val score:\s*([0-9.eE+-]+)",
        text,
    )
    if len(best_matches) != 1:
        raise RuntimeError(
            f"{run_id}: expected one Best epoch line, got {len(best_matches)}"
        )

    test_matches = re.findall(
        r"Test Results:\s*OrderedDict\((\[\(.*?\)\])\)",
        text,
    )
    if len(test_matches) != 1:
        raise RuntimeError(
            f"{run_id}: expected one Test Results line, got {len(test_matches)}"
        )

    metric_pairs = ast.literal_eval(test_matches[0])
    metrics = dict(metric_pairs)

    train_matches = re.findall(
        r"Training - \[Epoch\s+(\d+)\]:\s*100%[^\n]*?"
        r"\[([0-9:]+)<00:00",
        text,
    )
    train_by_epoch = {}
    for epoch, duration in train_matches:
        train_by_epoch[int(epoch)] = parse_duration(duration)

    val_by_epoch = {}
    for result_match in re.finditer(
        r"INFO:root:\[Epoch\s+(\d+)\]\s+Val Results:",
        text,
    ):
        epoch = int(result_match.group(1))
        train_loss_marker = f"INFO:root:[Epoch {epoch}] Train Loss:"
        segment_start = text.rfind(
            train_loss_marker, 0, result_match.start()
        )
        if segment_start < 0:
            raise RuntimeError(
                f"{run_id}: Train Loss marker missing for epoch {epoch}"
            )

        segment = text[segment_start:result_match.start()]
        completed_val = re.findall(
            r"Eval - val:\s*100%[^\n]*?"
            r"\[([0-9:]+)<00:00",
            segment,
        )
        if not completed_val:
            raise RuntimeError(
                f"{run_id}: validation duration missing for epoch {epoch}"
            )

        val_by_epoch[epoch] = parse_duration(completed_val[-1])

    val_seconds = sum(val_by_epoch.values())

    test_time_matches = re.findall(
        r"Eval - test:\s*100%[^\n]*?\[([0-9:]+)<00:00",
        text,
    )
    if not test_time_matches:
        raise RuntimeError(f"{run_id}: test evaluation duration not found")

    test_eval_seconds = parse_duration(test_time_matches[-1])
    train_seconds = sum(train_by_epoch.values())

    checkpoint_matches = sorted(
        glob.glob(
            str(
                records
                / "checkpoints"
                / f"{run_id}_best_epoch*.pth"
            )
        )
    )
    if len(checkpoint_matches) != 1:
        raise RuntimeError(
            f"{run_id}: expected one best checkpoint, got {checkpoint_matches}"
        )

    checksum_path = results_dir / f"{run_id}_checkpoint_sha256.txt"
    checksum = checksum_path.read_text(
        encoding="utf-8", errors="replace"
    ).split()[0]

    duration_seconds = int(meta["duration_seconds"])
    best_epoch = int(best_matches[0][0])
    best_val = float(best_matches[0][1])

    row = {
        "seed": seed,
        "run_id": run_id,
        "best_epoch": best_epoch,
        "best_val_ndcg10": best_val,
        "test_ndcg5": float(metrics["ndcg@5"]),
        "test_ndcg10": float(metrics["ndcg@10"]),
        "test_recall5": float(metrics["recall@5"]),
        "test_recall10": float(metrics["recall@10"]),
        "total_run_seconds": duration_seconds,
        "total_run_hms": meta["duration_hms"],
        "optimizer_train_seconds": train_seconds,
        "validation_seconds": val_seconds,
        "test_eval_seconds": test_eval_seconds,
        "test_eval_instances": eval_instances,
        "inference_ms_per_user": (
            test_eval_seconds * 1000.0 / eval_instances
        ),
        "test_users_per_second": (
            eval_instances / test_eval_seconds
        ),
        "gpu": meta["gpu"],
        "peak_gpu_memory_mib": int(meta["peak_gpu_memory_mib"]),
        "exit_code": int(meta["exit_code"]),
        "checkpoint": checkpoint_matches[0],
        "checkpoint_sha256": checksum,
        "config": str(config_path),
        "log": str(log_path),
    }
    rows.append(row)

csv_path = results_dir / "actionpiece_sports_five_seed_results.csv"
json_path = results_dir / "actionpiece_sports_five_seed_results.json"
stats_path = results_dir / "actionpiece_sports_five_seed_mean_std.csv"
md_path = results_dir / "actionpiece_sports_five_seed_report.md"

with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

json_path.write_text(
    json.dumps(rows, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

stat_fields = [
    "best_epoch",
    "best_val_ndcg10",
    "test_ndcg5",
    "test_ndcg10",
    "test_recall5",
    "test_recall10",
    "total_run_seconds",
    "optimizer_train_seconds",
    "validation_seconds",
    "test_eval_seconds",
    "inference_ms_per_user",
    "test_users_per_second",
    "peak_gpu_memory_mib",
]

stats_rows = []
for field in stat_fields:
    values = [float(row[field]) for row in rows]
    stats_rows.append(
        {
            "metric": field,
            "n": len(values),
            "mean": statistics.mean(values),
            "sample_std": statistics.stdev(values),
            "min": min(values),
            "max": max(values),
        }
    )

with stats_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["metric", "n", "mean", "sample_std", "min", "max"],
    )
    writer.writeheader()
    writer.writerows(stats_rows)

lines = [
    "# ActionPiece Sports_and_Outdoors 五随机种子实验",
    "",
    "## 实验协议",
    "",
    "- 数据集：Amazon Reviews 2014 / Sports_and_Outdoors",
    "- 评估：全物品评估",
    "- 随机种子：2024、2025、2026、2027、2028",
    "- 验证选择指标：NDCG@10",
    "- 最大轮数：200",
    "- 早停耐心：20",
    "- GPU：NVIDIA A800 80GB PCIe",
    "- 推理时间：最终测试总墙钟时间 / 35,598 个测试用户",
    "- 标准差：五个随机种子的样本标准差",
    "",
    "## 单种子结果",
    "",
    "| Seed | Best epoch | Val NDCG@10 | Test NDCG@5 | "
    "Test NDCG@10 | Test Recall@5 | Test Recall@10 | "
    "总运行时间 | 测试时间(s) | 推理(ms/user) |",
    "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]

for row in rows:
    lines.append(
        f"| {row['seed']} | {row['best_epoch']} | "
        f"{row['best_val_ndcg10']:.8f} | "
        f"{row['test_ndcg5']:.8f} | "
        f"{row['test_ndcg10']:.8f} | "
        f"{row['test_recall5']:.8f} | "
        f"{row['test_recall10']:.8f} | "
        f"{row['total_run_hms']} | "
        f"{row['test_eval_seconds']} | "
        f"{row['inference_ms_per_user']:.4f} |"
    )

lines.extend(
    [
        "",
        "## 均值与样本标准差",
        "",
        "| 指标 | 均值 | 样本标准差 |",
        "|---|---:|---:|",
    ]
)

for item in stats_rows:
    lines.append(
        f"| {item['metric']} | {item['mean']:.10f} | "
        f"{item['sample_std']:.10f} |"
    )

md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

print("GENERATED:", csv_path)
print("GENERATED:", json_path)
print("GENERATED:", stats_path)
print("GENERATED:", md_path)
print()
print("SEED RESULTS")
for row in rows:
    print(
        row["seed"],
        "best_epoch=", row["best_epoch"],
        "val_ndcg10=", row["best_val_ndcg10"],
        "test_ndcg10=", row["test_ndcg10"],
        "test_recall10=", row["test_recall10"],
        "duration=", row["total_run_hms"],
        "test_seconds=", row["test_eval_seconds"],
        "ms/user=", round(row["inference_ms_per_user"], 4),
    )

print()
print("MEAN ± SAMPLE STD")
for item in stats_rows:
    if item["metric"] in {
        "test_ndcg5",
        "test_ndcg10",
        "test_recall5",
        "test_recall10",
        "total_run_seconds",
        "inference_ms_per_user",
    }:
        print(
            item["metric"],
            "=",
            item["mean"],
            "±",
            item["sample_std"],
        )
