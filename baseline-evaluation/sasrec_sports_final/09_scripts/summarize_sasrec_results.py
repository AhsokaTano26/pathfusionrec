#!/usr/bin/env python3
"""Calculate mean, sample standard deviation, minimum and maximum."""

import csv
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "06_results" / "sasrec_per_seed_results.csv"
OUTPUT = ROOT / "06_results" / "sasrec_summary_results.csv"
METRICS = [
    "ndcg_at_5",
    "recall_hr_at_5",
    "ndcg_at_10",
    "recall_hr_at_10",
    "selection_wall_seconds",
    "pure_training_seconds",
    "test_wall_seconds",
    "wall_ms_per_user",
    "ranking_ms_per_user",
]


def main() -> None:
    with INPUT.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    summary = []
    for metric in METRICS:
        values = [float(row[metric]) for row in rows]
        summary.append(
            {
                "metric": metric,
                "n": len(values),
                "mean": statistics.mean(values),
                "sample_std": statistics.stdev(values),
                "min": min(values),
                "max": max(values),
            }
        )

    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["metric", "n", "mean", "sample_std", "min", "max"]
        )
        writer.writeheader()
        writer.writerows(summary)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()

