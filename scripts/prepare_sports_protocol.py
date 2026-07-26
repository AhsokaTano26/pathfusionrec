#!/usr/bin/env python3
"""Build the unified Sports_and_Outdoors data and evaluation protocol.

The protocol artifacts are derived only from ActionPiece-compatible
``all_item_seqs.json`` and ``id_mapping.json`` interaction sources.  When a
fresh checkout does not contain those sources, ``--bootstrap-actionpiece``
downloads the exact Amazon Reviews 2014 files used by ActionPiece and
materializes its processed interaction files before building the protocol.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Iterator, Mapping, Sequence
from urllib.request import Request, urlopen


CATEGORY = "Sports_and_Outdoors"
DATASET = "AmazonReviews2014"
REVIEWS_URL = (
    "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/"
    "reviews_Sports_and_Outdoors_5.json.gz"
)
METADATA_URL = (
    "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/"
    "meta_Sports_and_Outdoors.json.gz"
)
MAX_HISTORY_LENGTH = 20
MAX_BUNDLE_LENGTH = 8


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=default_root,
        help=f"Project root (default: {default_root})",
    )
    parser.add_argument(
        "--bootstrap-actionpiece",
        action="store_true",
        help=(
            "Download the official Amazon Reviews 2014 5-core reviews and "
            "metadata and create ActionPiece-compatible source JSON if absent."
        ),
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download raw files (requires --bootstrap-actionpiece).",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any, *, pretty: bool = False) -> None:
    """Atomically write deterministic UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            value,
            handle,
            ensure_ascii=False,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
            allow_nan=False,
        )
        handle.write("\n")
    temporary.replace(path)


def download(url: str, destination: Path, *, force: bool = False) -> None:
    if destination.exists() and not force:
        print(f"[download] reuse {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    request = Request(url, headers={"User-Agent": "pathfusionrec-data-preparation/1.0"})
    print(f"[download] {url}")
    with urlopen(request, timeout=120) as response, temporary.open("wb") as output:
        total_header = response.headers.get("Content-Length")
        total = int(total_header) if total_header else None
        copied = 0
        next_report = 32 * 1024 * 1024
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            copied += len(chunk)
            if copied >= next_report:
                if total:
                    print(f"[download] {copied / total:.0%} ({copied:,}/{total:,} bytes)")
                else:
                    print(f"[download] {copied:,} bytes")
                next_report += 32 * 1024 * 1024
    temporary.replace(destination)
    print(f"[download] saved {destination} ({destination.stat().st_size:,} bytes)")


def parse_amazon_gzip(path: Path) -> Iterator[dict[str, Any]]:
    """Parse strict JSON or the legacy Python-literal Amazon line format."""
    with gzip.open(path, "rt", encoding="utf-8", errors="strict") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                try:
                    value = ast.literal_eval(line)
                except (SyntaxError, ValueError) as exc:
                    raise ValueError(
                        f"Cannot parse {path} line {line_number}"
                    ) from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path} line {line_number} is not an object")
            yield value


def build_actionpiece_sources(
    reviews_path: Path, all_item_seqs_path: Path, id_mapping_path: Path
) -> None:
    """Reproduce ActionPiece's stable timestamp sort and first-seen ID mapping."""
    print(f"[bootstrap] parsing interactions from {reviews_path}")
    grouped: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    for input_order, interaction in enumerate(parse_amazon_gzip(reviews_path)):
        try:
            user = str(interaction["reviewerID"])
            item = str(interaction["asin"])
            timestamp = int(interaction["unixReviewTime"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid review record at zero-based index {input_order}"
            ) from exc
        grouped[user].append((item, timestamp, input_order))

    all_item_seqs: dict[str, list[str]] = {}
    user2id: dict[str, int] = {"[PAD]": 0}
    item2id: dict[str, int] = {"[PAD]": 0}
    id2user = ["[PAD]"]
    id2item = ["[PAD]"]

    for user, item_time in grouped.items():
        # Python's sort is stable. input_order makes the tie behavior explicit.
        item_time.sort(key=lambda value: (value[1], value[2]))
        items = [item for item, _, _ in item_time]
        user2id[user] = len(id2user)
        id2user.append(user)
        for item in items:
            if item not in item2id:
                item2id[item] = len(id2item)
                id2item.append(item)
        # ActionPiece stores raw ASINs here and keeps numeric IDs in id_mapping.
        all_item_seqs[user] = items

    id_mapping = {
        "user2id": user2id,
        "item2id": item2id,
        "id2user": id2user,
        "id2item": id2item,
    }
    write_json(all_item_seqs_path, all_item_seqs)
    write_json(id_mapping_path, id_mapping)
    print(
        "[bootstrap] wrote ActionPiece-compatible sources: "
        f"{len(grouped):,} users, {len(id2item) - 1:,} items, "
        f"{sum(map(len, all_item_seqs.values())):,} interactions"
    )


def bootstrap_if_requested(
    *,
    raw_dir: Path,
    processed_dir: Path,
    enabled: bool,
    force_download: bool,
) -> tuple[Path, Path, Path]:
    reviews_path = raw_dir / "reviews_Sports_and_Outdoors_5.json.gz"
    metadata_path = raw_dir / "meta_Sports_and_Outdoors.json.gz"
    all_item_seqs_path = processed_dir / "all_item_seqs.json"
    id_mapping_path = processed_dir / "id_mapping.json"

    if enabled:
        download(REVIEWS_URL, reviews_path, force=force_download)
        download(METADATA_URL, metadata_path, force=force_download)
        if not all_item_seqs_path.exists() or not id_mapping_path.exists():
            processed_dir.mkdir(parents=True, exist_ok=True)
            build_actionpiece_sources(
                reviews_path, all_item_seqs_path, id_mapping_path
            )

    missing = [
        path
        for path in (all_item_seqs_path, id_mapping_path, metadata_path)
        if not path.exists()
    ]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Required source files are missing:\n"
            f"{formatted}\n"
            "Run again with --bootstrap-actionpiece to download and create them."
        )
    return all_item_seqs_path, id_mapping_path, metadata_path


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_sources(
    all_item_seqs: Mapping[str, Sequence[str]], id_mapping: Mapping[str, Any]
) -> None:
    expected_keys = {"user2id", "item2id", "id2user", "id2item"}
    if set(id_mapping) != expected_keys:
        raise ValueError(
            f"id_mapping keys must be {sorted(expected_keys)}, got {sorted(id_mapping)}"
        )
    user2id = id_mapping["user2id"]
    item2id = id_mapping["item2id"]
    id2user = id_mapping["id2user"]
    id2item = id_mapping["id2item"]
    if (
        user2id.get("[PAD]") != 0
        or item2id.get("[PAD]") != 0
        or not id2user
        or not id2item
        or id2user[0] != "[PAD]"
        or id2item[0] != "[PAD]"
    ):
        raise ValueError("ActionPiece padding ID 0 is missing or inconsistent")
    if len(user2id) != len(id2user) or len(item2id) != len(id2item):
        raise ValueError("Forward and reverse mapping lengths differ")
    for raw_user, numeric_id in user2id.items():
        if id2user[numeric_id] != raw_user:
            raise ValueError(f"User mapping is not round-trippable: {raw_user}")
    for asin, numeric_id in item2id.items():
        if id2item[numeric_id] != asin:
            raise ValueError(f"Item mapping is not round-trippable: {asin}")
    for raw_user, sequence in all_item_seqs.items():
        if raw_user not in user2id:
            raise ValueError(f"Sequence user is absent from id_mapping: {raw_user}")
        unknown = next((asin for asin in sequence if asin not in item2id), None)
        if unknown is not None:
            raise ValueError(f"Sequence item is absent from id_mapping: {unknown}")


def trim_history(history: Sequence[int]) -> list[int]:
    return list(history[-MAX_HISTORY_LENGTH:])


def build_split_and_training_counts(
    all_item_seqs: Mapping[str, Sequence[str]], id_mapping: Mapping[str, Any]
) -> tuple[
    dict[str, list[dict[str, Any]]],
    Counter[int],
    dict[int, int],
    dict[str, list[int]],
    int,
]:
    """Create the standard leave-one-out split.

    For i1...iL, validation predicts i(L-1) from i1...i(L-2), and test
    predicts iL from i1...i(L-1).  Training contains every next-item prefix
    ending before the validation target.
    """
    user2id: Mapping[str, int] = id_mapping["user2id"]
    item2id: Mapping[str, int] = id_mapping["item2id"]
    split: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    training_item_counts: Counter[int] = Counter()
    training_history_lengths: dict[int, int] = {}
    mapped_sequences: dict[str, list[int]] = {}
    skipped_short_users = 0

    for raw_user, raw_sequence in all_item_seqs.items():
        if len(raw_sequence) < 3:
            skipped_short_users += 1
            continue
        user_id = int(user2id[raw_user])
        sequence = [int(item2id[asin]) for asin in raw_sequence]
        mapped_sequences[raw_user] = sequence

        training_interactions = sequence[:-2]
        training_item_counts.update(training_interactions)
        training_history_lengths[user_id] = len(training_interactions)

        # t is the prefix length: 1 <= t <= L - 3.
        for t in range(1, len(sequence) - 2):
            split["train"].append(
                {
                    "user_id": user_id,
                    "history": trim_history(sequence[:t]),
                    "target": sequence[t],
                }
            )
        split["validation"].append(
            {
                "user_id": user_id,
                "history": trim_history(sequence[:-2]),
                "target": sequence[-2],
            }
        )
        split["test"].append(
            {
                "user_id": user_id,
                "history": trim_history(sequence[:-1]),
                "target": sequence[-1],
            }
        )

    return (
        split,
        training_item_counts,
        training_history_lengths,
        mapped_sequences,
        skipped_short_users,
    )


def text_field(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return " ".join(part for part in (text_field(v) for v in value) if part)
    return str(value)


def flatten_categories(value: Any) -> list[str]:
    flattened: list[str] = []
    seen: set[str] = set()

    def visit(node: Any) -> None:
        if isinstance(node, str):
            if node and node not in seen:
                seen.add(node)
                flattened.append(node)
        elif isinstance(node, (list, tuple)):
            for child in node:
                visit(child)

    visit(value)
    return flattened


def load_relevant_metadata(
    metadata_path: Path, mapped_asins: set[str]
) -> tuple[dict[str, dict[str, Any]], int]:
    print(f"[metadata] scanning {metadata_path}")
    relevant: dict[str, dict[str, Any]] = {}
    duplicate_records = 0
    for record in parse_amazon_gzip(metadata_path):
        asin = str(record.get("asin", ""))
        if asin not in mapped_asins:
            continue
        if asin in relevant:
            duplicate_records += 1
            continue
        relevant[asin] = record
    return relevant, duplicate_records


def build_bundles(
    metadata: Mapping[str, Mapping[str, Any]],
    id_mapping: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    item2id: Mapping[str, int] = id_mapping["item2id"]
    id2item: Sequence[str] = id_mapping["id2item"]
    bundles: dict[str, dict[str, Any]] = {}

    # id2item order is the ActionPiece numeric-ID order.
    for asin in id2item[1:]:
        meta = metadata.get(asin)
        if meta is None:
            continue
        item_ids = [int(item2id[asin])]
        seen = {asin}
        related = meta.get("related")
        bought_together: Iterable[Any] = ()
        if isinstance(related, Mapping):
            value = related.get("bought_together", ())
            if isinstance(value, (list, tuple)):
                bought_together = value
        for related_asin_value in bought_together:
            related_asin = str(related_asin_value)
            if related_asin in seen or related_asin not in item2id:
                continue
            seen.add(related_asin)
            item_ids.append(int(item2id[related_asin]))
            if len(item_ids) == MAX_BUNDLE_LENGTH:
                break
        bundles[asin] = {
            "item_ids": item_ids,
            "title": text_field(meta.get("title")),
            "description": text_field(meta.get("description")),
            "categories": flatten_categories(meta.get("categories")),
            "source": "bought_together",
        }
    return bundles


def build_subset_masks(
    test_samples: Sequence[Mapping[str, Any]],
    training_item_counts: Mapping[int, int],
    training_history_lengths: Mapping[int, int],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    counts = {
        "all": 0,
        "cold_start_target": 0,
        "long_tail_target": 0,
        "sparse_user": 0,
    }
    for sample_id, sample in enumerate(test_samples):
        user_id = int(sample["user_id"])
        target = int(sample["target"])
        target_count = int(training_item_counts.get(target, 0))
        row = {
            "sample_id": sample_id,
            "user_id": user_id,
            "target": target,
            "all": True,
            "cold_start_target": target_count == 0,
            "long_tail_target": target_count <= 5,
            "sparse_user": int(training_history_lengths[user_id]) <= 5,
        }
        rows.append(row)
        for key in counts:
            counts[key] += int(row[key])
    return {
        "definitions": {
            "basis": "training_interactions_only",
            "all": "all test samples",
            "cold_start_target": "target training interaction count == 0",
            "long_tail_target": "target training interaction count <= 5",
            "sparse_user": "user training history length <= 5",
        },
        "counts": counts,
        "test": rows,
    }


def linear_percentile(sorted_values: Sequence[int], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def rounded(value: float) -> float:
    return round(value, 6)


def validate_artifacts(
    *,
    split: Mapping[str, Sequence[Mapping[str, Any]]],
    bundles: Mapping[str, Mapping[str, Any]],
    subset_masks: Mapping[str, Any],
    mapped_sequences: Mapping[str, Sequence[int]],
    id_mapping: Mapping[str, Any],
) -> dict[str, bool]:
    user2id: Mapping[str, int] = id_mapping["user2id"]
    validation_by_user = {
        int(sample["user_id"]): sample for sample in split["validation"]
    }
    test_by_user = {int(sample["user_id"]): sample for sample in split["test"]}
    leave_one_out_ok = True
    max_history_ok = True
    for raw_user, sequence in mapped_sequences.items():
        user_id = int(user2id[raw_user])
        validation = validation_by_user[user_id]
        test = test_by_user[user_id]
        if (
            int(validation["target"]) != sequence[-2]
            or int(test["target"]) != sequence[-1]
        ):
            leave_one_out_ok = False
            break
        if (
            list(validation["history"]) != trim_history(sequence[:-2])
            or list(test["history"]) != trim_history(sequence[:-1])
        ):
            leave_one_out_ok = False
            break
    for part in split.values():
        if any(len(sample["history"]) > MAX_HISTORY_LENGTH for sample in part):
            max_history_ok = False
            break

    item2id: Mapping[str, int] = id_mapping["item2id"]
    bundle_subject_first_ok = all(
        bool(bundle["item_ids"])
        and int(bundle["item_ids"][0]) == int(item2id[asin])
        and len(bundle["item_ids"]) <= MAX_BUNDLE_LENGTH
        and len(bundle["item_ids"]) == len(set(bundle["item_ids"]))
        for asin, bundle in bundles.items()
    )
    mask_alignment_ok = (
        len(subset_masks["test"]) == len(split["test"])
        and all(
            int(mask["sample_id"]) == index
            and int(mask["user_id"]) == int(split["test"][index]["user_id"])
            and int(mask["target"]) == int(split["test"][index]["target"])
            for index, mask in enumerate(subset_masks["test"])
        )
    )
    checks = {
        "source_mapping_round_trip": True,
        "leave_one_out_targets_and_histories": leave_one_out_ok,
        "history_length_at_most_20": max_history_ok,
        "bundle_subject_first_unique_and_at_most_8": bundle_subject_first_ok,
        "subset_masks_align_with_test_samples": mask_alignment_ok,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise AssertionError(f"Artifact validation failed: {', '.join(failures)}")
    return checks


def main() -> int:
    args = parse_args()
    if args.force_download and not args.bootstrap_actionpiece:
        raise ValueError("--force-download requires --bootstrap-actionpiece")

    project_root = args.project_root.resolve()
    dataset_root = (
        project_root / "data" / DATASET / CATEGORY
    )
    raw_dir = dataset_root / "raw"
    source_processed_dir = dataset_root / "processed"
    output_dir = project_root / "data" / "processed" / "sports_protocol"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_item_seqs_path, id_mapping_path, metadata_path = bootstrap_if_requested(
        raw_dir=raw_dir,
        processed_dir=source_processed_dir,
        enabled=args.bootstrap_actionpiece,
        force_download=args.force_download,
    )

    all_item_seqs = load_json(all_item_seqs_path)
    id_mapping = load_json(id_mapping_path)
    if not isinstance(all_item_seqs, dict) or not isinstance(id_mapping, dict):
        raise ValueError("ActionPiece source JSON roots must be objects")
    validate_sources(all_item_seqs, id_mapping)

    (
        split,
        training_item_counts,
        training_history_lengths,
        mapped_sequences,
        skipped_short_users,
    ) = build_split_and_training_counts(all_item_seqs, id_mapping)

    mapped_asins = set(id_mapping["item2id"])
    mapped_asins.discard("[PAD]")
    metadata, duplicate_metadata_records = load_relevant_metadata(
        metadata_path, mapped_asins
    )
    bundles = build_bundles(metadata, id_mapping)
    subset_masks = build_subset_masks(
        split["test"], training_item_counts, training_history_lengths
    )
    checks = validate_artifacts(
        split=split,
        bundles=bundles,
        subset_masks=subset_masks,
        mapped_sequences=mapped_sequences,
        id_mapping=id_mapping,
    )

    split_path = output_dir / "split.json"
    bundles_path = output_dir / "bundles.json"
    subset_masks_path = output_dir / "subset_masks.json"
    statistics_path = output_dir / "statistics.json"
    write_json(split_path, split)
    write_json(bundles_path, bundles)
    write_json(subset_masks_path, subset_masks)

    bundle_lengths = sorted(len(bundle["item_ids"]) for bundle in bundles.values())
    singleton_count = sum(length == 1 for length in bundle_lengths)
    full_interaction_count = sum(len(seq) for seq in all_item_seqs.values())
    eligible_interaction_count = sum(
        len(seq) for seq in mapped_sequences.values()
    )
    target_without_metadata = {
        part: sum(
            id_mapping["id2item"][int(sample["target"])] not in metadata
            for sample in split[part]
        )
        for part in ("validation", "test")
    }
    statistics = {
        "protocol": {
            "name": "sports_protocol",
            "version": 1,
            "dataset": DATASET,
            "category": CATEGORY,
            "split": "leave_one_out",
            "max_history_length": MAX_HISTORY_LENGTH,
            "bundle_definition": "subject plus Amazon related.bought_together",
            "max_bundle_length": MAX_BUNDLE_LENGTH,
            "candidate_catalog": "all mapped items with metadata",
        },
        "counts": {
            "users": len(id_mapping["id2user"]) - 1,
            "eligible_users_length_at_least_3": len(mapped_sequences),
            "skipped_users_length_below_3": skipped_short_users,
            "items": len(id_mapping["id2item"]) - 1,
            "interactions": full_interaction_count,
            "eligible_user_interactions": eligible_interaction_count,
            "training_interactions": sum(training_item_counts.values()),
            "metadata_items": len(metadata),
            "candidate_items": len(bundles),
            "items_without_metadata": len(mapped_asins - set(metadata)),
            "duplicate_metadata_records_ignored": duplicate_metadata_records,
            "train_samples": len(split["train"]),
            "validation_samples": len(split["validation"]),
            "test_samples": len(split["test"]),
            "validation_targets_without_metadata": target_without_metadata[
                "validation"
            ],
            "test_targets_without_metadata": target_without_metadata["test"],
        },
        "subsets": subset_masks["counts"],
        "bundles": {
            "count": len(bundle_lengths),
            "singleton_count": singleton_count,
            "singleton_ratio": rounded(
                singleton_count / len(bundle_lengths) if bundle_lengths else 0.0
            ),
            "length_mean": rounded(
                sum(bundle_lengths) / len(bundle_lengths)
                if bundle_lengths
                else 0.0
            ),
            "length_percentiles_linear": {
                f"p{percentile}": rounded(
                    linear_percentile(bundle_lengths, percentile)
                )
                for percentile in (0, 25, 50, 75, 90, 95, 99, 100)
            },
        },
        "sources": {
            "reviews_url": REVIEWS_URL,
            "metadata_url": METADATA_URL,
            "all_item_seqs_sha256": sha256_file(all_item_seqs_path),
            "id_mapping_sha256": sha256_file(id_mapping_path),
            "metadata_gzip_sha256": sha256_file(metadata_path),
        },
        "artifact_sha256": {
            "split.json": sha256_file(split_path),
            "bundles.json": sha256_file(bundles_path),
            "subset_masks.json": sha256_file(subset_masks_path),
        },
        "validation": checks,
    }
    write_json(statistics_path, statistics, pretty=True)

    print(json.dumps(statistics["counts"], ensure_ascii=False, indent=2))
    print(json.dumps(statistics["subsets"], ensure_ascii=False, indent=2))
    print(f"[done] artifacts written to {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # concise CLI failure while preserving exception type
        print(f"[error] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise

