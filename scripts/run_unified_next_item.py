"""Train Semantic Only, Interaction Only, or Fusion on the fixed Sports protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as functional
from torch.utils.data import DataLoader, Dataset

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / 'src'))

from pathfusionrec import PathFusionNextItemModel
from pathfusionrec.evaluation import evaluate_ranked_lists, rank_candidates


FIELD_NAMES = ('title', 'description', 'category')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=PathFusionNextItemModel.MODES, required=True)
    parser.add_argument('--fusion-strategy', choices=PathFusionNextItemModel.FUSION_STRATEGIES, default='concat')
    parser.add_argument('--protocol-dir', type=Path, default=REPOSITORY_ROOT / 'data' / 'processed' / 'sports_protocol')
    parser.add_argument('--data-dir', type=Path, default=REPOSITORY_ROOT / 'data' / 'AmazonReviews2014' / 'Sports_and_Outdoors' / 'processed')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--eval-batch-size', type=int, default=128)
    parser.add_argument('--negative-count', type=int, default=127)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--max-history-length', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max-train-samples', type=int)
    parser.add_argument('--max-validation-samples', type=int)
    parser.add_argument('--max-test-samples', type=int)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open(encoding='utf-8') as file:
        return json.load(file)


def stable_text_vector(value: object, dimension: int) -> Tensor:
    """Create a deterministic field vector without fitting on validation/test text."""
    if isinstance(value, list):
        value = ' '.join(str(part) for part in value)
    tokens = str(value or '[missing]').lower().split()
    vector = torch.zeros(dimension, dtype=torch.float32)
    for token in tokens:
        digest = hashlib.blake2b(token.encode('utf-8'), digest_size=8).digest()
        integer = int.from_bytes(digest, 'little')
        vector[integer % dimension] += 1.0 if integer & 1 else -1.0
    return functional.normalize(vector, dim=0) if tokens else vector


class ProtocolData:
    """In-memory fixed protocol artifacts and model-ready candidate features."""

    def __init__(self, protocol_dir: Path, data_dir: Path) -> None:
        self.split = load_json(protocol_dir / 'split.json')
        self.subset_masks = load_json(protocol_dir / 'subset_masks.json')['test']
        bundles = load_json(protocol_dir / 'bundles.json')
        item_mapping = load_json(data_dir / 'id_mapping.json')['item2id']
        self.num_items = max(int(item_id) for item_id in item_mapping.values()) + 1
        self.candidate_ids = torch.arange(1, self.num_items, dtype=torch.long)
        self.semantic_dim = 128
        semantic_values = np.fromfile(data_dir / 'sentence-t5-base.sent_emb', dtype=np.float32)
        expected_values = (self.num_items - 1) * self.semantic_dim
        if semantic_values.size != expected_values:
            raise ValueError('sentence embedding size does not match the item mapping.')
        self.semantic_lookup = torch.zeros(self.num_items, self.semantic_dim)
        self.semantic_lookup[1:] = torch.from_numpy(semantic_values.reshape(-1, self.semantic_dim))

        item_to_bundle = {int(item_mapping[asin]): value for asin, value in bundles.items()}
        if set(self.candidate_ids.tolist()) != set(item_to_bundle):
            raise ValueError('Candidate catalog and bundle catalog do not match.')
        max_bundle_length = max(len(bundle['item_ids']) for bundle in item_to_bundle.values())
        self.bundle_item_ids = torch.zeros(self.num_items, max_bundle_length, dtype=torch.long)
        self.bundle_mask = torch.zeros(self.num_items, max_bundle_length, dtype=torch.bool)
        self.global_fields = {
            name: torch.zeros(self.num_items, self.semantic_dim) for name in FIELD_NAMES
        }
        for item_id, bundle in item_to_bundle.items():
            item_ids = torch.tensor(bundle['item_ids'], dtype=torch.long)
            self.bundle_item_ids[item_id, : item_ids.numel()] = item_ids
            self.bundle_mask[item_id, : item_ids.numel()] = True
            self.global_fields['title'][item_id] = stable_text_vector(bundle['title'], self.semantic_dim)
            self.global_fields['description'][item_id] = stable_text_vector(bundle['description'], self.semantic_dim)
            self.global_fields['category'][item_id] = stable_text_vector(bundle['categories'], self.semantic_dim)

    def candidate_features(self, item_ids: Tensor, device: torch.device) -> tuple[dict[str, Tensor], Tensor, Tensor]:
        item_ids = item_ids.cpu()
        bundle_ids = self.bundle_item_ids[item_ids]
        return (
            {name: values[item_ids].to(device) for name, values in self.global_fields.items()},
            self.semantic_lookup[bundle_ids].to(device),
            self.bundle_mask[item_ids].to(device),
        )

    def history_semantics(self, history_ids: Tensor) -> Tensor:
        return self.semantic_lookup[history_ids]


class PrefixDataset(Dataset[dict[str, Any]]):
    def __init__(self, samples: list[dict[str, Any]], max_history_length: int) -> None:
        self.samples = samples
        self.max_history_length = max_history_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        history = sample['history'][-self.max_history_length :]
        return {'history': history, 'target': int(sample['target']), 'sample_id': index}


def collate_samples(samples: list[dict[str, Any]]) -> dict[str, Tensor]:
    max_history = max(len(sample['history']) for sample in samples)
    histories = torch.zeros(len(samples), max_history, dtype=torch.long)
    for index, sample in enumerate(samples):
        histories[index, : len(sample['history'])] = torch.tensor(sample['history'], dtype=torch.long)
    return {
        'history_ids': histories,
        'history_mask': histories.ne(0),
        'targets': torch.tensor([sample['target'] for sample in samples], dtype=torch.long),
        'sample_ids': torch.tensor([sample['sample_id'] for sample in samples], dtype=torch.long),
    }


def sample_candidate_ids(targets: Tensor, num_items: int, negative_count: int, generator: random.Random) -> tuple[Tensor, Tensor]:
    candidate_ids = list(dict.fromkeys(targets.tolist()))
    while len(candidate_ids) < len(set(targets.tolist())) + negative_count:
        candidate = generator.randrange(1, num_items)
        if candidate not in candidate_ids:
            candidate_ids.append(candidate)
    labels = torch.tensor([candidate_ids.index(target.item()) for target in targets], dtype=torch.long)
    return torch.tensor(candidate_ids, dtype=torch.long), labels


def move_history(batch: Mapping[str, Tensor], data: ProtocolData, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    history_ids = batch['history_ids'].to(device)
    return history_ids, data.history_semantics(batch['history_ids']).to(device), batch['history_mask'].to(device)


def train_epoch(
    model: PathFusionNextItemModel,
    loader: DataLoader[dict[str, Tensor]],
    data: ProtocolData,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    generator: random.Random,
) -> float:
    model.train()
    losses: list[float] = []
    device = torch.device(args.device)
    for batch in loader:
        candidate_ids, labels = sample_candidate_ids(
            batch['targets'], data.num_items, args.negative_count, generator
        )
        fields, bundle_embeddings, bundle_mask = data.candidate_features(candidate_ids, device)
        output = model(
            *move_history(batch, data, device),
            candidate_ids.to(device),
            fields,
            bundle_embeddings,
            bundle_mask,
        )
        loss = functional.cross_entropy(PathFusionNextItemModel.score(output), labels.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


@torch.no_grad()
def encode_catalog(model: PathFusionNextItemModel, data: ProtocolData, batch_size: int, device: torch.device) -> Tensor:
    model.eval()
    embeddings: list[Tensor] = []
    for candidate_ids in data.candidate_ids.split(batch_size):
        fields, bundle_embeddings, bundle_mask = data.candidate_features(candidate_ids, device)
        vectors, _ = model.encode_candidates(candidate_ids.to(device), fields, bundle_embeddings, bundle_mask)
        embeddings.append(functional.normalize(vectors, dim=-1).cpu())
    return torch.cat(embeddings)


@torch.no_grad()
def evaluate_split(
    model: PathFusionNextItemModel,
    samples: list[dict[str, Any]],
    data: ProtocolData,
    masks: list[dict[str, Any]] | None,
    args: argparse.Namespace,
) -> dict[str, dict[str, float]]:
    device = torch.device(args.device)
    catalog = encode_catalog(model, data, args.eval_batch_size, device)
    loader = DataLoader(PrefixDataset(samples, args.max_history_length), batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_samples)
    rankings: list[list[int]] = []
    targets: list[int] = []
    sample_indices: list[int] = []
    for batch in loader:
        query, _ = model.encode_query(*move_history(batch, data, device))
        scores = functional.normalize(query, dim=-1).cpu() @ catalog.T
        for row, target, history, sample_id in zip(scores, batch['targets'], batch['history_ids'], batch['sample_ids'], strict=True):
            history_ids = history[history.ne(0)].tolist()
            rankings.append(rank_candidates(data.candidate_ids.tolist(), row.tolist(), history_ids, int(target), top_k=50))
            targets.append(int(target))
            sample_indices.append(int(sample_id))

    results = {'all': evaluate_ranked_lists(rankings, targets, ks=(5, 10, 20, 50))}
    if masks is not None:
        for subset in ('cold_start_target', 'long_tail_target', 'sparse_user'):
            selected = [index for index in sample_indices if masks[index][subset]]
            results[subset] = evaluate_ranked_lists(
                [rankings[index] for index in selected], [targets[index] for index in selected], ks=(5, 10, 20, 50)
            )
    return results


def limit_samples(samples: list[dict[str, Any]], maximum: int | None) -> list[dict[str, Any]]:
    return samples if maximum is None else samples[:maximum]


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.negative_count <= 0:
        raise ValueError('--epochs and --negative-count must be positive.')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data = ProtocolData(args.protocol_dir, args.data_dir)
    train_samples = limit_samples(data.split['train'], args.max_train_samples)
    validation_samples = limit_samples(data.split['validation'], args.max_validation_samples)
    test_samples = limit_samples(data.split['test'], args.max_test_samples)
    train_loader = DataLoader(PrefixDataset(train_samples, args.max_history_length), batch_size=args.batch_size, shuffle=True, collate_fn=collate_samples)
    model = PathFusionNextItemModel(num_items=data.num_items, semantic_dim=data.semantic_dim, hidden_dim=args.hidden_dim, mode=args.mode, fusion_strategy=args.fusion_strategy).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    generator = random.Random(args.seed)
    history: list[dict[str, Any]] = []
    best_epoch = 0
    best_validation_ndcg = float('-inf')
    best_state: dict[str, Tensor] | None = None
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, data, optimizer, args, generator)
        validation = evaluate_split(model, validation_samples, data, None, args)['all']
        history.append({'epoch': epoch, 'train_loss': loss, **validation})
        if validation['NDCG@10'] > best_validation_ndcg:
            best_epoch = epoch
            best_validation_ndcg = validation['NDCG@10']
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        print(json.dumps(history[-1], sort_keys=True))

    if best_state is None:
        raise RuntimeError('No validation checkpoint was produced.')
    model.load_state_dict(best_state)
    test_metrics = evaluate_split(model, test_samples, data, data.subset_masks[: len(test_samples)], args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'metrics.json').write_text(json.dumps({
        'arguments': vars(args),
        'best_epoch': best_epoch,
        'best_validation_ndcg@10': best_validation_ndcg,
        'history': history,
        'test': test_metrics,
    }, indent=2, default=str) + '\n')
    with (args.output_dir / 'training_history.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    torch.save(best_state, args.output_dir / 'best_model.pth')


if __name__ == '__main__':
    main()
