#!/usr/bin/env python3
"""Generate the human-readable delivery manifest.

The manifest excludes itself and the root checksum list to avoid circular
dependencies. Both files are still covered by the release workflow: the
manifest is included in DELIVERY_SHA256SUMS.txt, while that checksum list
cannot include its own hash.
"""

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "DELIVERY_MANIFEST.csv"
EXCLUDED = {"DELIVERY_MANIFEST.csv", "DELIVERY_SHA256SUMS.txt"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def classify(relative: str) -> str:
    if relative == "README.md":
        return "入口说明"
    prefix = relative.split("/", 1)[0]
    return {
        "01_protocol": "实验协议",
        "02_code": "代码",
        "03_config": "配置",
        "04_data_provenance": "数据与溯源",
        "05_runs": "三种子实验产物",
        "06_results": "结果汇总",
        "07_environment": "环境",
        "08_verification": "核验",
        "09_scripts": "辅助脚本",
        "10_references": "参考文献",
    }.get(prefix, "其他")


def source(relative: str) -> str:
    if relative.endswith("/SHA256SUMS.txt"):
        return "本次整理生成"
    original_markers = (
        "02_code/original_pmixer/",
        "02_code/sports_experiment/",
        "02_code/code_provenance/base_git_commit.txt",
        "02_code/code_provenance/git_status_at_archive.txt",
        "03_config/seed_",
        "04_data_provenance/processed_data/",
        "05_runs/",
        "07_environment/original_environment_records/",
        "10_references/SASRec_paper.pdf",
    )
    return "原实验文件原样复制" if relative.startswith(original_markers) else "本次整理生成"


def seed_value(relative: str) -> str:
    for seed in ("2026", "2027", "2028"):
        if f"seed_{seed}" in relative:
            return seed
    return ""


def main() -> None:
    files = sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file() and path.relative_to(ROOT).as_posix() not in EXCLUDED
    )
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "path",
                "category",
                "seed",
                "source",
                "status",
                "description",
                "sha256",
            ],
        )
        writer.writeheader()
        for path in files:
            relative = path.relative_to(ROOT).as_posix()
            writer.writerow(
                {
                    "path": relative,
                    "category": classify(relative),
                    "seed": seed_value(relative),
                    "source": source(relative),
                    "status": "已固化",
                    "description": path.name,
                    "sha256": digest(path),
                }
            )
    print(f"Wrote {OUTPUT} with {len(files)} entries")


if __name__ == "__main__":
    main()
