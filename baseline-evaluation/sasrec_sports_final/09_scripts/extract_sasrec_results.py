#!/usr/bin/env python3
"""Extract per-seed SASRec results from the archived JSON/JSONL files."""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "06_results" / "sasrec_per_seed_results.csv"
FIELDS = [
    "method",
    "dataset",
    "seed",
    "best_epoch",
    "stopped_epoch",
    "ndcg_at_5",
    "recall_hr_at_5",
    "ndcg_at_10",
    "recall_hr_at_10",
    "selection_wall_seconds",
    "pure_training_seconds",
    "test_wall_seconds",
    "wall_ms_per_user",
    "ranking_ms_per_user",
    "evaluated_users",
    "catalog_item_count",
    "gpu",
    "source_result_file",
]


def last_epoch(path: Path) -> int:
    final_record = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                final_record = json.loads(line)
    if final_record is None:
        raise ValueError(f"Empty training history: {path}")
    return int(final_record["epoch"])


def main() -> None:
    rows = []
    for seed in (2026, 2027, 2028):
        run_dir = ROOT / "05_runs" / f"seed_{seed}"
        result_path = run_dir / "test_results.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        metrics = result["test_metrics"]
        rows.append(
            {
                "method": "SASRec-PyTorch (pmixer)",
                "dataset": "Amazon Sports",
                "seed": seed,
                "best_epoch": result["best_epoch"],
                "stopped_epoch": last_epoch(run_dir / "training_history.jsonl"),
                "ndcg_at_5": metrics["NDCG@5"],
                "recall_hr_at_5": metrics["Recall@5"],
                "ndcg_at_10": metrics["NDCG@10"],
                "recall_hr_at_10": metrics["Recall@10"],
                "selection_wall_seconds": result["selection_wall_seconds"],
                "pure_training_seconds": result["pure_training_seconds"],
                "test_wall_seconds": metrics["wall_seconds"],
                "wall_ms_per_user": metrics["wall_ms_per_user"],
                "ranking_ms_per_user": metrics["ranking_ms_per_user"],
                "evaluated_users": metrics["evaluated_users"],
                "catalog_item_count": metrics["candidate_item_count"],
                "gpu": result["environment"]["gpu"],
                "source_result_file": f"05_runs/seed_{seed}/test_results.json",
            }
        )

    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()

