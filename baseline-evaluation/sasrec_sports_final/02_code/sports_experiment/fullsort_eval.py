import time

import numpy as np
import torch


def _build_sequence(user, user_train, user_valid, maxlen, split):
    sequence = np.zeros(maxlen, dtype=np.int64)
    index = maxlen - 1

    if split == "test":
        sequence[index] = user_valid[user][0]
        index -= 1
    elif split != "valid":
        raise ValueError("split必须为valid或test")

    for item in reversed(user_train[user]):
        sequence[index] = item
        index -= 1
        if index == -1:
            break

    return sequence


def _seen_items(user, user_train, user_valid, target, split):
    seen = set(user_train[user])

    if split == "test":
        seen.update(user_valid[user])

    # 若目标物品曾在历史中重复出现，仍必须保留当前目标。
    seen.discard(target)
    return seen


@torch.inference_mode()
def evaluate_fullsort(
    model,
    dataset,
    maxlen,
    split,
    batch_size=256,
    ks=(5, 10),
    max_users=None,
):
    user_train, user_valid, user_test, usernum, itemnum = dataset

    if split not in {"valid", "test"}:
        raise ValueError("split必须为valid或test")

    if not ks:
        raise ValueError("ks不能为空")

    max_k = max(ks)

    if max_k > itemnum:
        raise ValueError("最大的K不能超过物品总数")

    users = list(range(1, usernum + 1))

    if max_users is not None:
        users = users[:max_users]

    device = next(model.parameters()).device
    model.eval()

    hits = {k: 0.0 for k in ks}
    ndcgs = {k: 0.0 for k in ks}

    ranking_seconds = 0.0
    wall_start = time.perf_counter()

    for start in range(0, len(users), batch_size):
        batch_users = users[start:start + batch_size]

        sequences = np.stack([
            _build_sequence(
                user,
                user_train,
                user_valid,
                maxlen,
                split,
            )
            for user in batch_users
        ])

        if split == "valid":
            targets = np.asarray(
                [user_valid[user][0] for user in batch_users],
                dtype=np.int64,
            )
        else:
            targets = np.asarray(
                [user_test[user][0] for user in batch_users],
                dtype=np.int64,
            )

        seen_per_user = [
            _seen_items(
                user,
                user_train,
                user_valid,
                int(target),
                split,
            )
            for user, target in zip(batch_users, targets)
        ]

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        ranking_start = time.perf_counter()

        # log2feats()与固定版本model.py保持一致，输入使用NumPy数组。
        final_features = model.log2feats(sequences)[:, -1, :]

        # 只评估真实物品1～itemnum，明确排除编号0的[PAD]。
        item_embeddings = model.item_emb.weight[1:itemnum + 1]
        scores = torch.matmul(final_features, item_embeddings.transpose(0, 1))

        # 排除已经交互过的历史物品，但保留当前目标物品。
        for row, seen_items in enumerate(seen_per_user):
            if seen_items:
                columns = torch.as_tensor(
                    [item - 1 for item in seen_items],
                    dtype=torch.long,
                    device=device,
                )
                scores[row, columns] = -torch.inf

        top_items = torch.topk(
            scores,
            k=max_k,
            dim=1,
            largest=True,
            sorted=True,
        ).indices + 1

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        ranking_seconds += time.perf_counter() - ranking_start

        targets_tensor = torch.as_tensor(
            targets,
            dtype=torch.long,
            device=device,
        ).unsqueeze(1)

        matches = top_items.eq(targets_tensor)

        for k in ks:
            matches_k = matches[:, :k]
            hit_mask = matches_k.any(dim=1)

            hits[k] += hit_mask.float().sum().item()

            if hit_mask.any():
                hit_rows, hit_columns = torch.where(matches_k)
                ndcgs[k] += (
                    1.0 / torch.log2(hit_columns.float() + 2.0)
                ).sum().item()

    wall_seconds = time.perf_counter() - wall_start
    evaluated_users = len(users)

    metrics = {
        "split": split,
        "evaluated_users": evaluated_users,
        "candidate_item_count": itemnum,
        "history_filtering": (
            "train"
            if split == "valid"
            else "train+valid"
        ),
        "wall_seconds": wall_seconds,
        "ranking_seconds": ranking_seconds,
        "wall_ms_per_user": (
            wall_seconds * 1000.0 / evaluated_users
        ),
        "ranking_ms_per_user": (
            ranking_seconds * 1000.0 / evaluated_users
        ),
    }

    for k in ks:
        # 单正样本留一法下，Recall@K与HR@K数值相同。
        hr = hits[k] / evaluated_users
        ndcg = ndcgs[k] / evaluated_users

        metrics[f"HR@{k}"] = hr
        metrics[f"Recall@{k}"] = hr
        metrics[f"NDCG@{k}"] = ndcg

    return metrics
