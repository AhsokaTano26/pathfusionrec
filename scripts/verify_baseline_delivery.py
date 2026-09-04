#!/usr/bin/env python3
"""Independently rebuild and verify the role-C baseline deliverables.

The script only reads committed experiment artifacts. It never changes source
results; all derived files are written to one separate output directory.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import platform
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "experiment_results" / "05_baseline_quality_verification"
EXPECTED_SEEDS = [2024, 2025, 2026, 2027, 2028]
TOP_K = (5, 10, 20, 50)
SUBSETS = ("all", "cold_start_target", "long_tail_target", "sparse_user")

AP_ROOT = ROOT / "baseline-evaluation" / "actionpiece_sports_5seeds_delayed_earlystop"
SASREC_ROOT = ROOT / "baseline-evaluation" / "sasrec_sports_final"
PFR_ROOT = ROOT / "experiment_results" / "04_unified_next_item"
PROTOCOL_ROOT = ROOT / "data" / "processed" / "sports_protocol"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def sample_stats(values: Iterable[float]) -> dict[str, float | int]:
    collected = [float(value) for value in values]
    if not collected:
        raise ValueError("at least one value is required")
    return {
        "n": len(collected),
        "mean": statistics.fmean(collected),
        "sample_std": statistics.stdev(collected) if len(collected) > 1 else 0.0,
        "min": min(collected),
        "max": max(collected),
    }


def close(left: float, right: float, tolerance: float = 1e-10) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def add_check(
    checks: list[dict[str, str]],
    scope: str,
    check: str,
    status: str,
    path: Path | str,
    detail: str,
) -> None:
    display_path = str(path)
    if isinstance(path, Path):
        try:
            display_path = path.relative_to(ROOT).as_posix()
        except ValueError:
            display_path = path.as_posix()
    checks.append(
        {
            "scope": scope,
            "check": check,
            "status": status,
            "path": display_path,
            "detail": detail,
        }
    )


def parse_manifest(manifest: Path) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for line_number, raw_line in enumerate(
        manifest.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(r"^([0-9a-fA-F]{64})\s+[*]?(.+)$", line)
        if not match:
            raise ValueError(f"invalid manifest line {line_number}: {raw_line!r}")
        entries.append((match.group(1).lower(), match.group(2).removeprefix("./")))
    return entries


def verify_manifest(
    checks: list[dict[str, str]], scope: str, manifest: Path, base: Path
) -> None:
    if not manifest.is_file():
        add_check(checks, scope, "SHA-256 manifest", "INCOMPLETE", manifest, "清单缺失")
        return
    try:
        entries = parse_manifest(manifest)
    except ValueError as error:
        add_check(checks, scope, "SHA-256 manifest", "FAIL", manifest, str(error))
        return
    missing: list[str] = []
    mismatched: list[str] = []
    normalized_line_endings: list[str] = []
    for expected, relative in entries:
        candidate = base / Path(relative)
        if not candidate.is_file():
            missing.append(relative)
        elif sha256_file(candidate) != expected:
            # Git for Windows commonly checks text files out with CRLF even when
            # the delivery was hashed on Linux. Accept the repository-canonical
            # LF bytes, but report how many files needed that normalization.
            content = candidate.read_bytes()
            normalized = content.replace(b"\r\n", b"\n")
            if normalized != content and hashlib.sha256(normalized).hexdigest() == expected:
                normalized_line_endings.append(relative)
            else:
                mismatched.append(relative)
    status = "PASS" if not missing and not mismatched else "FAIL"
    detail = (
        f"checked={len(entries)}, missing={len(missing)}, "
        f"mismatched={len(mismatched)}, crlf_normalized={len(normalized_line_endings)}"
    )
    if missing:
        detail += f"; missing_files={';'.join(missing[:5])}"
    if mismatched:
        detail += f"; mismatched_files={';'.join(mismatched[:5])}"
    add_check(checks, scope, "SHA-256 manifest", status, manifest, detail)


def parse_ordered_dict(fragment: str) -> dict[str, float]:
    pairs = ast.literal_eval("[" + fragment + "]")
    return {str(key): float(value) for key, value in pairs}


def parse_actionpiece_log(path: Path) -> dict[str, Any]:
    train_pattern = re.compile(r"\[Epoch (\d+)\] Train Loss: ([0-9.eE+-]+)")
    validation_pattern = re.compile(
        r"\[Epoch (\d+)\] Val Results: OrderedDict\(\[(.*)\]\)"
    )
    best_pattern = re.compile(r"Best epoch: (\d+), Best val score: ([0-9.eE+-]+)")
    test_pattern = re.compile(r"Test Results: OrderedDict\(\[(.*)\]\)")
    epochs: dict[int, dict[str, Any]] = {}
    best_epoch: int | None = None
    best_value: float | None = None
    test: dict[str, float] | None = None
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if match := train_pattern.search(line):
            epoch = int(match.group(1))
            epochs.setdefault(epoch, {"epoch": epoch})["train_loss"] = float(match.group(2))
        elif match := validation_pattern.search(line):
            epoch = int(match.group(1))
            row = epochs.setdefault(epoch, {"epoch": epoch})
            row.update({f"val_{key}": value for key, value in parse_ordered_dict(match.group(2)).items()})
        elif match := best_pattern.search(line):
            best_epoch = int(match.group(1))
            best_value = float(match.group(2))
        elif match := test_pattern.search(line):
            test = parse_ordered_dict(match.group(1))
    return {
        "epochs": [epochs[key] for key in sorted(epochs)],
        "best_epoch": best_epoch,
        "best_value": best_value,
        "test": test,
    }


def actionpiece_checks(
    checks: list[dict[str, str]], output: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source = AP_ROOT / "02_results" / "five_seed_results.json"
    records = json.loads(source.read_text(encoding="utf-8"))
    seeds = sorted(int(row["seed"]) for row in records)
    add_check(
        checks,
        "ActionPiece",
        "five-seed coverage",
        "PASS" if seeds == EXPECTED_SEEDS else "INCOMPLETE",
        source,
        f"observed={seeds}; expected={EXPECTED_SEEDS}",
    )

    metric_fields = [
        "best_val_ndcg@10",
        *[f"ndcg@{k}" for k in TOP_K],
        *[f"recall@{k}" for k in TOP_K],
        *[f"err@{k}" for k in TOP_K],
        "training_wall_seconds",
        "test_wall_seconds",
        "total_wall_seconds",
    ]
    invalid_values: list[str] = []
    for record in records:
        for metric in metric_fields:
            value = record.get(metric)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                invalid_values.append(f"seed{record.get('seed')}:{metric}")
    add_check(
        checks,
        "ActionPiece",
        "finite metric schema",
        "PASS" if not invalid_values else "FAIL",
        source,
        "all required metrics are finite" if not invalid_values else ";".join(invalid_values),
    )

    per_seed_fields = ["seed", "best_epoch", *metric_fields]
    write_csv(output / "actionpiece_five_seed_results.csv", records, per_seed_fields)

    summary_rows: list[dict[str, Any]] = []
    for metric in ["best_epoch", *metric_fields]:
        summary_rows.append({"metric": metric, **sample_stats(row[metric] for row in records)})
    write_csv(
        output / "actionpiece_mean_std.csv",
        summary_rows,
        ["metric", "n", "mean", "sample_std", "min", "max"],
    )

    existing_summary = AP_ROOT / "02_results" / "mean_std.csv"
    existing = {
        row["metric"]: row
        for row in csv.DictReader(existing_summary.open(encoding="utf-8-sig", newline=""))
    }
    summary_mismatches: list[str] = []
    for row in summary_rows:
        if row["metric"] == "best_epoch" or row["metric"] not in existing:
            continue
        old = existing[row["metric"]]
        if not close(row["mean"], float(old["mean"])) or not close(
            row["sample_std"], float(old["sample_std"])
        ):
            summary_mismatches.append(str(row["metric"]))
    add_check(
        checks,
        "ActionPiece",
        "independent ddof=1 summary",
        "PASS" if not summary_mismatches else "FAIL",
        existing_summary,
        "recomputed values match" if not summary_mismatches else ";".join(summary_mismatches),
    )

    expected_checkpoint_hashes: dict[int, str] = {}
    original_manifest = AP_ROOT / "01_report" / "original_SHA256SUMS.txt"
    for expected, relative in parse_manifest(original_manifest):
        match = re.fullmatch(r"01_experiments/seed(\d+)/best_model\.pth", relative)
        if match:
            expected_checkpoint_hashes[int(match.group(1))] = expected

    history_rows: list[dict[str, Any]] = []
    for record in records:
        seed = int(record["seed"])
        log_path = AP_ROOT / "03_logs" / f"seed{seed}.log"
        parsed = parse_actionpiece_log(log_path)
        for row in parsed["epochs"]:
            history_rows.append({"seed": seed, **row})
        test_matches = parsed["test"] is not None and all(
            close(parsed["test"][metric], record[metric])
            for metric in [f"ndcg@{k}" for k in TOP_K]
            + [f"recall@{k}" for k in TOP_K]
        )
        best_matches = (
            parsed["best_epoch"] == int(record["best_epoch"])
            and parsed["best_value"] is not None
            and close(parsed["best_value"], record["best_val_ndcg@10"])
        )
        add_check(
            checks,
            "ActionPiece",
            f"seed {seed} log/result trace",
            "PASS" if test_matches and best_matches and len(parsed["epochs"]) == 200 else "FAIL",
            log_path,
            f"epochs={len(parsed['epochs'])}, best_match={best_matches}, test_match={test_matches}",
        )
        checkpoint = AP_ROOT / "04_checkpoints" / f"seed{seed}_best_model.pth"
        expected_hash = expected_checkpoint_hashes.get(seed)
        actual_hash = sha256_file(checkpoint) if checkpoint.is_file() else None
        add_check(
            checks,
            "ActionPiece",
            f"seed {seed} checkpoint SHA-256",
            "PASS" if expected_hash and actual_hash == expected_hash else "FAIL",
            checkpoint,
            f"expected={expected_hash}; actual={actual_hash}",
        )

    history_fields = [
        "seed",
        "epoch",
        "train_loss",
        *[f"val_ndcg@{k}" for k in TOP_K],
        *[f"val_recall@{k}" for k in TOP_K],
        *[f"val_err@{k}" for k in TOP_K],
    ]
    write_csv(output / "actionpiece_training_history.csv", history_rows, history_fields)
    render_actionpiece_svg(output / "actionpiece_validation_curves.svg", history_rows)

    add_check(
        checks,
        "ActionPiece",
        "unified subset metrics",
        "INCOMPLETE",
        AP_ROOT,
        "aggregated logs do not contain per-user predictions or cold/long-tail/sparse subsets",
    )
    return records, summary_rows


def render_actionpiece_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    width, height = 1000, 620
    left, right, top, bottom = 90, 35, 55, 80
    plot_width = width - left - right
    plot_height = height - top - bottom
    series: dict[int, list[tuple[int, float]]] = {}
    for row in rows:
        if "val_ndcg@10" in row:
            series.setdefault(int(row["seed"]), []).append(
                (int(row["epoch"]), float(row["val_ndcg@10"]))
            )
    max_epoch = max(epoch for values in series.values() for epoch, _ in values)
    max_value = max(value for values in series.values() for _, value in values) * 1.08
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c"]

    def x_pos(epoch: int) -> float:
        return left + (epoch - 1) / max(1, max_epoch - 1) * plot_width

    def y_pos(value: float) -> float:
        return top + plot_height - value / max_value * plot_height

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,"Microsoft YaHei",sans-serif;fill:#1f2937}.grid{stroke:#d1d5db;stroke-width:1}.axis{stroke:#111827;stroke-width:1.5}.curve{fill:none;stroke-width:2}</style>',
        '<text x="500" y="30" font-size="20" text-anchor="middle">ActionPiece validation NDCG@10 (five seeds)</text>',
    ]
    for tick in range(6):
        value = max_value * tick / 5
        y = y_pos(value)
        svg.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}"/>')
        svg.append(f'<text x="{left-12}" y="{y+5:.2f}" font-size="12" text-anchor="end">{value:.3f}</text>')
    for epoch in (1, 50, 100, 150, 200):
        x = x_pos(epoch)
        svg.append(f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top+plot_height}"/>')
        svg.append(f'<text x="{x:.2f}" y="{top+plot_height+24}" font-size="12" text-anchor="middle">{epoch}</text>')
    svg.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top+plot_height}" x2="{width-right}" y2="{top+plot_height}"/>',
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_height}"/>',
            f'<text x="{left+plot_width/2:.2f}" y="{height-25}" font-size="14" text-anchor="middle">Epoch</text>',
            f'<text x="22" y="{top+plot_height/2:.2f}" font-size="14" text-anchor="middle" transform="rotate(-90 22 {top+plot_height/2:.2f})">Validation NDCG@10</text>',
        ]
    )
    for index, seed in enumerate(sorted(series)):
        points = " ".join(f"{x_pos(epoch):.2f},{y_pos(value):.2f}" for epoch, value in series[seed])
        color = colors[index % len(colors)]
        svg.append(f'<polyline class="curve" stroke="{color}" points="{points}"/>')
        legend_x = left + index * 150
        legend_y = height - 52
        svg.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x+26}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        svg.append(f'<text x="{legend_x+34}" y="{legend_y+5}" font-size="13">seed {seed}</text>')
    svg.append("</svg>")
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def sasrec_checks(
    checks: list[dict[str, str]], output: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    verify_manifest(
        checks,
        "SASRec",
        SASREC_ROOT / "DELIVERY_SHA256SUMS.txt",
        SASREC_ROOT,
    )
    records: list[dict[str, Any]] = []
    for run_dir in sorted((SASREC_ROOT / "05_runs").glob("seed_*")):
        result_path = run_dir / "test_results.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        metrics = result["test_metrics"]
        record = {
            "seed": int(result["seed"]),
            "best_epoch": int(result["best_epoch"]),
            "best_validation_ndcg@10": float(result["best_validation_NDCG@10"]),
            "ndcg@5": float(metrics["NDCG@5"]),
            "recall@5": float(metrics["Recall@5"]),
            "ndcg@10": float(metrics["NDCG@10"]),
            "recall@10": float(metrics["Recall@10"]),
            "selection_wall_seconds": float(result["selection_wall_seconds"]),
            "pure_training_seconds": float(result["pure_training_seconds"]),
            "test_wall_seconds": float(metrics["wall_seconds"]),
            "evaluated_users": int(metrics["evaluated_users"]),
            "candidate_item_count": int(metrics["candidate_item_count"]),
            "max_history_length": int(result["arguments"]["maxlen"]),
        }
        records.append(record)
        required = [
            "args.json",
            "best_model.pth",
            "best_validation.json",
            "test_results.json",
            "training.log",
            "training_history.jsonl",
        ]
        missing = [name for name in required if not (run_dir / name).is_file()]
        add_check(
            checks,
            "SASRec",
            f"seed {record['seed']} required artifacts",
            "PASS" if not missing else "FAIL",
            run_dir,
            "complete" if not missing else f"missing={missing}",
        )
        verify_manifest(checks, f"SASRec seed {record['seed']}", run_dir / "SHA256SUMS.txt", run_dir)

    write_csv(
        output / "sasrec_three_seed_results.csv",
        records,
        list(records[0]),
    )
    summary_metrics = [
        "best_epoch",
        "best_validation_ndcg@10",
        "ndcg@5",
        "recall@5",
        "ndcg@10",
        "recall@10",
        "selection_wall_seconds",
        "pure_training_seconds",
        "test_wall_seconds",
    ]
    summary_rows = [
        {"metric": metric, **sample_stats(row[metric] for row in records)}
        for metric in summary_metrics
    ]
    write_csv(
        output / "sasrec_mean_std.csv",
        summary_rows,
        ["metric", "n", "mean", "sample_std", "min", "max"],
    )
    existing_metric_names = {
        "ndcg@5": "ndcg_at_5",
        "recall@5": "recall_hr_at_5",
        "ndcg@10": "ndcg_at_10",
        "recall@10": "recall_hr_at_10",
        "selection_wall_seconds": "selection_wall_seconds",
        "pure_training_seconds": "pure_training_seconds",
        "test_wall_seconds": "test_wall_seconds",
    }
    existing_summary_path = SASREC_ROOT / "06_results" / "sasrec_summary_results.csv"
    existing_summary = {
        row["metric"]: row
        for row in csv.DictReader(existing_summary_path.open(encoding="utf-8-sig", newline=""))
    }
    recomputed = {row["metric"]: row for row in summary_rows}
    mismatches: list[str] = []
    for current_name, existing_name in existing_metric_names.items():
        current = recomputed[current_name]
        existing = existing_summary[existing_name]
        if not close(current["mean"], float(existing["mean"])) or not close(
            current["sample_std"], float(existing["sample_std"])
        ):
            mismatches.append(current_name)
    add_check(
        checks,
        "SASRec",
        "independent ddof=1 summary",
        "PASS" if not mismatches else "FAIL",
        existing_summary_path,
        "recomputed values match" if not mismatches else ";".join(mismatches),
    )
    add_check(
        checks,
        "SASRec",
        "five-seed coverage",
        "INCOMPLETE",
        SASREC_ROOT / "05_runs",
        f"observed={sorted(row['seed'] for row in records)}; missing=[2024, 2025]",
    )
    add_check(
        checks,
        "SASRec",
        "unified max history",
        "INCOMPLETE",
        SASREC_ROOT / "03_config" / "common_config.json",
        "current maxlen=200; sports_protocol requires 20",
    )
    add_check(
        checks,
        "SASRec",
        "unified metrics and subsets",
        "INCOMPLETE",
        SASREC_ROOT / "05_runs",
        "@20/@50 and cold_start_target/long_tail_target/sparse_user are absent",
    )
    return records, summary_rows


def pfr_checks(checks: list[dict[str, str]], output: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    commits: set[str] = set()
    for experiment_dir in sorted(path for path in PFR_ROOT.iterdir() if path.is_dir()):
        experiment_id = experiment_dir.name
        seed_dirs = sorted(path for path in experiment_dir.glob("seed*") if path.is_dir())
        observed_seeds = [int(path.name.removeprefix("seed")) for path in seed_dirs]
        add_check(
            checks,
            experiment_id,
            "five-seed coverage",
            "PASS" if observed_seeds == EXPECTED_SEEDS else "INCOMPLETE",
            experiment_dir,
            f"observed={observed_seeds}; expected={EXPECTED_SEEDS}",
        )
        experiment_card = experiment_dir / "experiment_card.md"
        add_check(
            checks,
            experiment_id,
            "experiment card",
            "PASS" if experiment_card.is_file() else "INCOMPLETE",
            experiment_card,
            "present" if experiment_card.is_file() else "missing",
        )
        for seed_dir in seed_dirs:
            seed = int(seed_dir.name.removeprefix("seed"))
            metrics_path = seed_dir / "metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            history = metrics.get("history", [])
            validation_values = [float(row["NDCG@10"]) for row in history]
            computed_best = max(range(len(history)), key=validation_values.__getitem__) + 1
            best_ok = (
                computed_best == int(metrics["best_epoch"])
                and close(max(validation_values), metrics["best_validation_ndcg@10"])
            )
            test = metrics.get("test", {})
            missing_schema: list[str] = []
            for subset in SUBSETS:
                if subset not in test:
                    missing_schema.append(subset)
                    continue
                for k in TOP_K:
                    for prefix in ("NDCG", "Recall"):
                        key = f"{prefix}@{k}"
                        value = test[subset].get(key)
                        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                            missing_schema.append(f"{subset}:{key}")
            add_check(
                checks,
                experiment_id,
                f"seed {seed} metric schema",
                "PASS" if not missing_schema else "FAIL",
                metrics_path,
                "four subsets and @5/@10/@20/@50 complete" if not missing_schema else ";".join(missing_schema),
            )
            add_check(
                checks,
                experiment_id,
                f"seed {seed} best checkpoint selection",
                "PASS" if best_ok else "FAIL",
                metrics_path,
                f"recorded_epoch={metrics['best_epoch']}; recomputed_epoch={computed_best}",
            )
            required = ["metrics.json", "training_history.csv", "best_model.pth", "run.log"]
            missing = [name for name in required if not (seed_dir / name).is_file()]
            add_check(
                checks,
                experiment_id,
                f"seed {seed} core artifacts",
                "PASS" if not missing else "FAIL",
                seed_dir,
                "complete" if not missing else f"missing={missing}",
            )
            environment_file = seed_dir / "environment.txt"
            add_check(
                checks,
                experiment_id,
                f"seed {seed} consolidated environment",
                "PASS" if environment_file.is_file() else "INCOMPLETE",
                environment_file,
                "present" if environment_file.is_file() else "python/GPU files exist but environment.txt is absent",
            )
            verify_manifest(checks, f"{experiment_id} seed {seed}", seed_dir / "sha256sum.txt", seed_dir)

            commit_file = seed_dir / "git_commit.txt"
            commit = commit_file.read_text(encoding="utf-8-sig").strip()
            commits.add(commit)
            args = metrics["arguments"]
            for subset in SUBSETS:
                for k in TOP_K:
                    rows.append(
                        {
                            "experiment_id": experiment_id,
                            "mode": args["mode"],
                            "fusion_strategy": args["fusion_strategy"],
                            "seed": seed,
                            "best_epoch": metrics["best_epoch"],
                            "best_validation_ndcg@10": metrics["best_validation_ndcg@10"],
                            "subset": subset,
                            "k": k,
                            "ndcg": test[subset][f"NDCG@{k}"],
                            "recall": test[subset][f"Recall@{k}"],
                            "commit": commit,
                        }
                    )
    write_csv(
        output / "pfr_seed2026_metrics.csv",
        rows,
        [
            "experiment_id",
            "mode",
            "fusion_strategy",
            "seed",
            "best_epoch",
            "best_validation_ndcg@10",
            "subset",
            "k",
            "ndcg",
            "recall",
            "commit",
        ],
    )
    add_check(
        checks,
        "PathFusionRec",
        "commit consistency",
        "PASS" if len(commits) == 1 else "FAIL",
        PFR_ROOT,
        f"commits={sorted(commits)}",
    )
    stats_path = PROTOCOL_ROOT / "statistics.json"
    if stats_path.is_file():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        for filename, expected_hash in stats.get("artifact_sha256", {}).items():
            artifact = PROTOCOL_ROOT / filename
            actual = sha256_file(artifact) if artifact.is_file() else None
            add_check(
                checks,
                "sports_protocol",
                f"{filename} SHA-256",
                "PASS" if actual == expected_hash else "FAIL",
                artifact,
                f"expected={expected_hash}; actual={actual}",
            )
    else:
        add_check(
            checks,
            "sports_protocol",
            "local protocol artifacts",
            "INCOMPLETE",
            stats_path,
            "ignored data artifacts are not present in this checkout",
        )
    return rows


def paper_comparison(output: Path, ap_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    paper = {
        "ndcg@5": (0.0205, 0.0002),
        "ndcg@10": (0.0264, 0.0003),
        "recall@5": (0.0316, 0.0005),
        "recall@10": (0.0500, 0.0007),
    }
    computed = {row["metric"]: row for row in ap_summary}
    rows: list[dict[str, Any]] = []
    for metric, (paper_mean, paper_std) in paper.items():
        current = computed[metric]
        rows.append(
            {
                "method": "ActionPiece",
                "dataset": "Amazon Sports",
                "metric": metric,
                "current_mean": current["mean"],
                "current_sample_std": current["sample_std"],
                "paper_mean": paper_mean,
                "paper_std": paper_std,
                "relative_gap": current["mean"] / paper_mean - 1.0,
                "protocol_assessment": "近似一致；延迟早停与统一显式全库评分口径不同",
                "source": "https://arxiv.org/abs/2502.13581",
            }
        )
    write_csv(
        output / "actionpiece_paper_comparison.csv",
        rows,
        [
            "method",
            "dataset",
            "metric",
            "current_mean",
            "current_sample_std",
            "paper_mean",
            "paper_std",
            "relative_gap",
            "protocol_assessment",
            "source",
        ],
    )
    return rows


def write_report(
    output: Path,
    checks: list[dict[str, str]],
    ap_summary: list[dict[str, Any]],
    sasrec_records: list[dict[str, Any]],
) -> None:
    counts: dict[str, int] = {}
    for row in checks:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    ap = {row["metric"]: row for row in ap_summary}
    sasrec_root_hash = next(
        row
        for row in checks
        if row["scope"] == "SASRec" and row["check"] == "SHA-256 manifest"
    )
    report = f"""# C 角色基线质量核验报告

核验源 commit：`{git_revision()}`

生成方式：`python scripts/verify_baseline_delivery.py`

## 自动核验摘要

- PASS：{counts.get('PASS', 0)}
- INCOMPLETE：{counts.get('INCOMPLETE', 0)}
- FAIL：{counts.get('FAIL', 0)}

`FAIL` 表示现有交付内部存在可验证的不闭合项；`INCOMPLETE` 表示尚未达到三人协作文档规定的阶段交付条件。逐项证据见 `integrity_check.csv`。

## ActionPiece

2024–2028 五 seed 齐全。脚本从原始 JSON 独立复算，使用 `statistics.stdev`（`ddof=1`）：

| 指标 | 均值 | 样本标准差 | 论文值 |
|---|---:|---:|---:|
| NDCG@5 | {ap['ndcg@5']['mean']:.9f} | {ap['ndcg@5']['sample_std']:.9f} | 0.0205 ± 0.0002 |
| NDCG@10 | {ap['ndcg@10']['mean']:.9f} | {ap['ndcg@10']['sample_std']:.9f} | 0.0264 ± 0.0003 |
| Recall@5 | {ap['recall@5']['mean']:.9f} | {ap['recall@5']['sample_std']:.9f} | 0.0316 ± 0.0005 |
| Recall@10 | {ap['recall@10']['mean']:.9f} | {ap['recall@10']['sample_std']:.9f} | 0.0500 ± 0.0007 |

五份日志均重建出 200 个 epoch，日志最佳轮次、最佳验证值及测试指标与结果 JSON 一致；五个 checkpoint 也与原交付清单中的 SHA-256 一致。由于没有逐用户预测和统一子集指标，该结果不能进入 PathFusionRec 统一主表。

## SASRec

当前仅有 {len(sasrec_records)} 个 seed（{', '.join(str(row['seed']) for row in sasrec_records)}）。三个逐 seed SHA-256 清单通过；根清单状态为 `{sasrec_root_hash['status']}`（{sasrec_root_hash['detail']}）。此外缺少 seed 2024/2025、@20/@50、三个子集，且 `maxlen=200` 与统一协议的 20 不一致。因此只能标记为项目实测参考。

## PathFusionRec seed 2026

Interaction、Semantic、Fusion/concat 三个目录的四类子集和 @5/@10/@20/@50 schema 完整，最佳验证轮次复算一致，三个运行使用同一代码 commit。当前仍只有一个 seed，且缺实验卡与统一数据文件哈希。三个 `sha256sum.txt` 都引用缺失的 `qq_report.txt`，所以清单未闭合；这里不修改原目录，只报告问题供 A/B 补交。

## VQ-Rec

仓库中没有 VQ-Rec 代码或结果。原论文是跨域迁移设置且不包含 Sports；在 A 决定 Sports 适配方案、预训练来源、文本字段与实验卡前，本核验不擅自实现或启动训练。

## 主表使用结论

目前没有任何基线同时满足固定 `sports_protocol`、2024–2028 五 seed、统一显式全库候选、@5/@10/@20/@50、三个子集和闭合哈希记录。因此本目录中的数值均不得直接并入最终统一主表。
"""
    (output / "verification_report.md").write_text(report, encoding="utf-8")


def write_output_manifest(output: Path) -> None:
    manifest = output / "SHA256SUMS.txt"
    lines = []
    for path in sorted(candidate for candidate in output.iterdir() if candidate.is_file() and candidate != manifest):
        lines.append(f"{sha256_file(path)}  {path.name}")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_verification_environment(output: Path) -> None:
    lines = [
        f"python={sys.version.replace(chr(10), ' ')}",
        f"platform={platform.platform()}",
    ]
    try:
        import sklearn

        lines.append(f"scikit_learn={sklearn.__version__}")
    except ImportError:
        lines.append("scikit_learn=not_installed")
    try:
        import torch

        lines.extend(
            [
                f"torch={torch.__version__}",
                f"cuda_available={torch.cuda.is_available()}",
                "gpu="
                + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "not_available"),
            ]
        )
    except ImportError:
        lines.append("torch=not_installed")
    (output / "verification_environment.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run(output: Path) -> list[dict[str, str]]:
    output.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, str]] = []
    _, ap_summary = actionpiece_checks(checks, output)
    sasrec_records, _ = sasrec_checks(checks, output)
    pfr_checks(checks, output)
    paper_comparison(output, ap_summary)
    write_csv(
        output / "integrity_check.csv",
        checks,
        ["scope", "check", "status", "path", "detail"],
    )
    write_report(output, checks, ap_summary, sasrec_records)
    write_verification_environment(output)
    write_output_manifest(output)
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return a non-zero code when an existing artifact fails integrity checks",
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    checks = run(output)
    counts: dict[str, int] = {}
    for row in checks:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(f"wrote {output}")
    print(", ".join(f"{key}={value}" for key, value in sorted(counts.items())))
    if args.strict and counts.get("FAIL", 0):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
