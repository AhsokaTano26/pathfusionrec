"""Unified full-catalog ranking utilities for single-target recommendation.

The main protocol evaluates every mapped item that has metadata.  A model may
score candidates however it likes, but history filtering and metric
calculation should pass through the functions in this module.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import json
import math
import random
from typing import TypeVar


ItemId = TypeVar("ItemId", bound=int)


def filter_history_candidates(
    candidate_ids: Iterable[ItemId],
    history: Iterable[ItemId],
    target: ItemId,
) -> list[ItemId]:
    """Remove seen items while retaining the true target.

    Candidate IDs must be unique and the target must already belong to the
    full-catalog candidate set.  Requiring membership catches metadata/catalog
    mismatches instead of silently changing the evaluation universe.
    """
    candidates = list(candidate_ids)
    if len(candidates) != len(set(candidates)):
        raise ValueError("candidate_ids must be unique")
    if target not in candidates:
        raise ValueError("target is absent from the candidate catalog")
    seen = set(history)
    seen.discard(target)
    return [candidate for candidate in candidates if candidate not in seen]


def rank_candidates(
    candidate_ids: Sequence[ItemId],
    scores: Sequence[float] | Mapping[ItemId, float],
    history: Iterable[ItemId],
    target: ItemId,
    *,
    top_k: int = 10,
) -> list[ItemId]:
    """History-filter and rank candidates by descending score.

    Numeric item ID is the deterministic secondary key for score ties.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    candidates = list(candidate_ids)
    if isinstance(scores, Mapping):
        missing = [candidate for candidate in candidates if candidate not in scores]
        if missing:
            raise ValueError(f"scores missing {len(missing)} candidate(s)")
        score_by_item = {
            candidate: float(scores[candidate]) for candidate in candidates
        }
    else:
        if len(scores) != len(candidates):
            raise ValueError("scores and candidate_ids must have equal lengths")
        score_by_item = {
            candidate: float(score)
            for candidate, score in zip(candidates, scores, strict=True)
        }
    if any(math.isnan(score) for score in score_by_item.values()):
        raise ValueError("scores must not contain NaN")

    filtered = filter_history_candidates(candidates, history, target)
    filtered.sort(key=lambda item: (-score_by_item[item], item))
    return filtered[:top_k]


def single_target_metrics(
    ranked_item_ids: Sequence[ItemId],
    target: ItemId,
    *,
    ks: Sequence[int] = (5, 10),
) -> dict[str, float]:
    """Calculate NDCG@K and Recall@K for one relevant item."""
    if not ks or any(k <= 0 for k in ks):
        raise ValueError("ks must contain positive integers")
    try:
        rank = ranked_item_ids.index(target) + 1
    except ValueError:
        rank = None
    metrics: dict[str, float] = {}
    for k in ks:
        hit = rank is not None and rank <= k
        metrics[f"NDCG@{k}"] = 1.0 / math.log2(rank + 1) if hit else 0.0
        metrics[f"Recall@{k}"] = 1.0 if hit else 0.0
    return metrics


def evaluate_ranked_lists(
    ranked_item_ids: Sequence[Sequence[ItemId]],
    targets: Sequence[ItemId],
    *,
    ks: Sequence[int] = (5, 10),
) -> dict[str, float]:
    """Macro-average single-target metrics over aligned ranked lists."""
    if len(ranked_item_ids) != len(targets):
        raise ValueError("ranked_item_ids and targets must have equal lengths")
    if not targets:
        return {
            name: 0.0
            for k in ks
            for name in (f"NDCG@{k}", f"Recall@{k}")
        }
    totals = {
        name: 0.0
        for k in ks
        for name in (f"NDCG@{k}", f"Recall@{k}")
    }
    for ranking, target in zip(ranked_item_ids, targets, strict=True):
        sample_metrics = single_target_metrics(ranking, target, ks=ks)
        for name, value in sample_metrics.items():
            totals[name] += value
    return {name: value / len(targets) for name, value in totals.items()}


def _self_test() -> dict[str, object]:
    candidates = list(range(1, 101))
    target = 50
    history = [1, 2, 3, target]
    perfect_scores = {item: (1.0 if item == target else 0.0) for item in candidates}
    perfect_ranking = rank_candidates(
        candidates, perfect_scores, history, target, top_k=10
    )
    perfect = single_target_metrics(perfect_ranking, target)
    assert all(value == 1.0 for value in perfect.values())
    assert target in filter_history_candidates(candidates, history, target)
    assert all(item not in perfect_ranking for item in (1, 2, 3))

    generator = random.Random(2024)
    random_scores = [generator.random() for _ in candidates]
    random_ranking = rank_candidates(
        candidates, random_scores, history, target, top_k=10
    )
    random_metrics = single_target_metrics(random_ranking, target)
    assert all(0.0 <= value <= 1.0 for value in random_metrics.values())
    return {
        "perfect_scores": perfect,
        "seeded_random_scores": random_metrics,
        "history_filter_target_retained": target
        in filter_history_candidates(candidates, history, target),
        "status": "passed",
    }


if __name__ == "__main__":
    print(json.dumps(_self_test(), indent=2))

