import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from fullsort_eval import evaluate_fullsort
from model import SASRec
from utils import WarpSampler, data_partition


def parse_args():
    parser = argparse.ArgumentParser(
        description="SASRec Sports训练与全物品排序评估"
    )

    parser.add_argument("--dataset", default="sports")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, required=True)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--maxlen", type=int, default=200)
    parser.add_argument("--hidden_units", type=int, default=50)
    parser.add_argument("--num_blocks", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--l2_emb", type=float, default=0.0)
    parser.add_argument("--norm_first", action="store_true", default=False)

    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min_delta", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=3)
    parser.add_argument("--device", default="cuda")

    return parser.parse_args()


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def initialize_model(model):
    for _, parameter in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(parameter.data)
        except (ValueError, RuntimeError):
            pass

    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了CUDA，但当前CUDA不可用")

    save_json(vars(args), output_dir / "args.json")

    dataset = data_partition(args.dataset)
    user_train, user_valid, user_test, usernum, itemnum = dataset

    train_interactions = sum(len(items) for items in user_train.values())
    valid_interactions = sum(len(items) for items in user_valid.values())
    test_interactions = sum(len(items) for items in user_test.values())

    dataset_summary = {
        "dataset": args.dataset,
        "user_count": usernum,
        "item_count": itemnum,
        "train_interactions": train_interactions,
        "valid_interactions": valid_interactions,
        "test_interactions": test_interactions,
        "total_interactions": (
            train_interactions
            + valid_interactions
            + test_interactions
        ),
        "average_train_sequence_length": (
            train_interactions / usernum
        ),
    }
    save_json(dataset_summary, output_dir / "dataset_summary.json")

    print("=== 数据统计 ===")
    for key, value in dataset_summary.items():
        print(f"{key}: {value}")

    assert usernum == 35598
    assert itemnum == 18357
    assert train_interactions == 225141
    assert valid_interactions == 35598
    assert test_interactions == 35598

    num_batches = (
        len(user_train) - 1
    ) // args.batch_size + 1

    sampler = WarpSampler(
        user_train,
        usernum,
        itemnum,
        batch_size=args.batch_size,
        maxlen=args.maxlen,
        n_workers=args.num_workers,
    )

    model = SASRec(
        usernum,
        itemnum,
        args,
    ).to(device)

    initialize_model(model)
    model.train()

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
    )

    best_checkpoint_path = (
        output_dir / "best_model_state_dict.pth"
    )
    history_path = output_dir / "training_history.jsonl"

    best_val_ndcg10 = -1.0
    best_epoch = None
    epochs_without_improvement = 0

    pure_training_seconds = 0.0
    experiment_start = time.perf_counter()

    try:
        with open(
            history_path,
            "w",
            encoding="utf-8",
        ) as history_file:

            for epoch in range(1, args.num_epochs + 1):
                model.train()
                epoch_loss = 0.0

                synchronize(device)
                epoch_train_start = time.perf_counter()

                for _ in range(num_batches):
                    users, sequences, positives, negatives = (
                        sampler.next_batch()
                    )

                    users = np.asarray(users)
                    sequences = np.asarray(sequences)
                    positives = np.asarray(positives)
                    negatives = np.asarray(negatives)

                    positive_logits, negative_logits = model(
                        users,
                        sequences,
                        positives,
                        negatives,
                    )

                    positive_labels = torch.ones(
                        positive_logits.shape,
                        device=device,
                    )
                    negative_labels = torch.zeros(
                        negative_logits.shape,
                        device=device,
                    )

                    non_padding = np.where(positives != 0)

                    loss = criterion(
                        positive_logits[non_padding],
                        positive_labels[non_padding],
                    )
                    loss += criterion(
                        negative_logits[non_padding],
                        negative_labels[non_padding],
                    )

                    for parameter in model.item_emb.parameters():
                        loss += (
                            args.l2_emb
                            * torch.sum(parameter ** 2)
                        )

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                synchronize(device)
                epoch_training_seconds = (
                    time.perf_counter()
                    - epoch_train_start
                )
                pure_training_seconds += epoch_training_seconds

                record = {
                    "epoch": epoch,
                    "mean_training_loss": (
                        epoch_loss / num_batches
                    ),
                    "epoch_training_seconds": (
                        epoch_training_seconds
                    ),
                    "pure_training_seconds": (
                        pure_training_seconds
                    ),
                }

                should_evaluate = (
                    epoch % args.eval_every == 0
                    or epoch == args.num_epochs
                )

                if should_evaluate:
                    validation_metrics = evaluate_fullsort(
                        model=model,
                        dataset=dataset,
                        maxlen=args.maxlen,
                        split="valid",
                        batch_size=args.eval_batch_size,
                        ks=(5, 10),
                    )

                    record["validation"] = validation_metrics
                    current_val_ndcg10 = (
                        validation_metrics["NDCG@10"]
                    )

                    improved = (
                        current_val_ndcg10
                        > best_val_ndcg10 + args.min_delta
                    )

                    if improved:
                        best_val_ndcg10 = current_val_ndcg10
                        best_epoch = epoch
                        epochs_without_improvement = 0

                        torch.save(
                            model.state_dict(),
                            best_checkpoint_path,
                        )

                        best_validation = {
                            "best_epoch": best_epoch,
                            "selection_metric": (
                                "validation_NDCG@10"
                            ),
                            "best_validation_metrics": (
                                validation_metrics
                            ),
                            "checkpoint": str(
                                best_checkpoint_path
                            ),
                        }
                        save_json(
                            best_validation,
                            output_dir
                            / "best_validation.json",
                        )
                    else:
                        epochs_without_improvement += 1

                    record["improved"] = improved
                    record["epochs_without_improvement"] = (
                        epochs_without_improvement
                    )

                    print(
                        f"epoch={epoch} "
                        f"loss={record['mean_training_loss']:.6f} "
                        f"train={epoch_training_seconds:.2f}s "
                        f"val_NDCG@5="
                        f"{validation_metrics['NDCG@5']:.6f} "
                        f"val_HR@5="
                        f"{validation_metrics['HR@5']:.6f} "
                        f"val_NDCG@10="
                        f"{validation_metrics['NDCG@10']:.6f} "
                        f"val_HR@10="
                        f"{validation_metrics['HR@10']:.6f} "
                        f"best_epoch={best_epoch} "
                        f"patience="
                        f"{epochs_without_improvement}/"
                        f"{args.patience}",
                        flush=True,
                    )
                else:
                    print(
                        f"epoch={epoch} "
                        f"loss={record['mean_training_loss']:.6f} "
                        f"train={epoch_training_seconds:.2f}s",
                        flush=True,
                    )

                history_file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                history_file.flush()

                if (
                    should_evaluate
                    and epochs_without_improvement
                    >= args.patience
                ):
                    print(
                        f"验证集连续"
                        f"{args.patience}次未提升，"
                        f"在epoch={epoch}早停。",
                        flush=True,
                    )
                    break

    finally:
        sampler.close()

    selection_wall_seconds = (
        time.perf_counter() - experiment_start
    )

    if best_epoch is None:
        raise RuntimeError("训练结束但没有保存最佳模型")

    print(
        f"加载验证集最佳模型：epoch={best_epoch}",
        flush=True,
    )

    state_dict = torch.load(
        best_checkpoint_path,
        map_location=device,
    )
    model.load_state_dict(state_dict)
    model.eval()

    # 测试集只在最佳验证模型确定后评估一次。
    test_metrics = evaluate_fullsort(
        model=model,
        dataset=dataset,
        maxlen=args.maxlen,
        split="test",
        batch_size=args.eval_batch_size,
        ks=(5, 10),
    )

    environment = {
        "python_hash_seed": os.environ["PYTHONHASHSEED"],
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "cudnn_benchmark": (
            torch.backends.cudnn.benchmark
        ),
        "cudnn_deterministic": (
            torch.backends.cudnn.deterministic
        ),
    }

    final_results = {
        "dataset": args.dataset,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "selection_rule": (
            "最高验证集NDCG@10；测试集不参与选模"
        ),
        "best_validation_NDCG@10": best_val_ndcg10,
        "test_metrics": test_metrics,
        "pure_training_seconds": pure_training_seconds,
        "selection_wall_seconds": selection_wall_seconds,
        "checkpoint": str(best_checkpoint_path),
        "environment": environment,
        "arguments": vars(args),
    }

    save_json(
        final_results,
        output_dir / "final_results.json",
    )

    print("\n=== 最终测试结果 ===")
    print(f"seed: {args.seed}")
    print(f"best_epoch: {best_epoch}")
    print(
        f"NDCG@5: {test_metrics['NDCG@5']:.6f}"
    )
    print(
        f"HR@5: {test_metrics['HR@5']:.6f}"
    )
    print(
        f"Recall@5: {test_metrics['Recall@5']:.6f}"
    )
    print(
        f"NDCG@10: {test_metrics['NDCG@10']:.6f}"
    )
    print(
        f"HR@10: {test_metrics['HR@10']:.6f}"
    )
    print(
        f"Recall@10: {test_metrics['Recall@10']:.6f}"
    )
    print(
        f"纯训练时间: "
        f"{pure_training_seconds:.2f}秒"
    )
    print(
        f"训练及验证选模总时间: "
        f"{selection_wall_seconds:.2f}秒"
    )
    print(
        f"测试排序延迟: "
        f"{test_metrics['ranking_ms_per_user']:.6f}"
        f"毫秒/用户"
    )
    print(
        f"结果目录: {output_dir}"
    )
    print("Done")


if __name__ == "__main__":
    main()
