"""Train BundleEncoder on real AmazonReviews2014 bought-together bundles.

This is a small, reproducible validation run—not the final recommendation
benchmark. Each raw Amazon ``bought_together`` relation forms a bundle: the
anchor product provides title/description/category semantics and the anchor
plus its co-bought products form the local set. The model learns to retrieve
the anchor's cached sentence-T5 embedding from the fused bundle representation.
"""

from __future__ import annotations

import argparse
import ast
import gzip
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import Tensor
from torch.nn import functional as functional
from torch.utils.data import DataLoader, Dataset

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / 'src'))

from pathfusionrec import BundleEncoder


@dataclass(frozen=True)
class BundleRecord:
    """One real bundle induced by an Amazon bought-together relation."""

    anchor_asin: str
    item_asins: tuple[str, ...]
    title: str
    description: str
    category: str


class BundleDataset(Dataset[dict[str, Tensor]]):
    def __init__(
        self,
        records: list[BundleRecord],
        field_embeddings: dict[str, np.ndarray],
        item_embeddings: dict[str, np.ndarray],
    ) -> None:
        self.records = records
        self.field_embeddings = field_embeddings
        self.item_embeddings = item_embeddings

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        record = self.records[index]
        return {
            'title': torch.from_numpy(self.field_embeddings['title'][index]),
            'description': torch.from_numpy(self.field_embeddings['description'][index]),
            'category': torch.from_numpy(self.field_embeddings['category'][index]),
            'items': torch.from_numpy(np.stack([self.item_embeddings[asin] for asin in record.item_asins])),
            'target': torch.from_numpy(self.item_embeddings[record.anchor_asin]),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-root',
        type=Path,
        default=REPOSITORY_ROOT / 'data' / 'AmazonReviews2014' / 'Sports_and_Outdoors',
    )
    parser.add_argument('--max-bundles', type=int, default=512)
    parser.add_argument('--max-items', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--output', type=Path, default=Path('reports/real_bundle_smoke.json'))
    return parser.parse_args()


def clean_text(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, list):
        return ' '.join(clean_text(part) for part in value)
    return str(value).strip()


def load_records(data_root: Path, max_bundles: int, max_items: int, seed: int) -> list[BundleRecord]:
    processed = data_root / 'processed'
    raw_metadata = data_root / 'raw' / 'meta_Sports_and_Outdoors.json.gz'
    with (processed / 'id_mapping.json').open() as file:
        item_asins = set(json.load(file)['item2id'])

    candidates: list[BundleRecord] = []
    with gzip.open(raw_metadata, 'rt') as file:
        for line in file:
            metadata = ast.literal_eval(line)
            anchor_asin = metadata.get('asin')
            if anchor_asin not in item_asins:
                continue
            co_bought = metadata.get('related', {}).get('bought_together', [])
            related_asins = [asin for asin in co_bought if asin in item_asins and asin != anchor_asin]
            if not related_asins:
                continue
            item_bundle = tuple([anchor_asin, *related_asins[: max_items - 1]])
            candidates.append(
                BundleRecord(
                    anchor_asin=anchor_asin,
                    item_asins=item_bundle,
                    title=clean_text(metadata.get('title')),
                    description=clean_text(metadata.get('description')),
                    category=clean_text(metadata.get('categories')),
                )
            )

    if len(candidates) < max_bundles:
        raise ValueError(f'Only {len(candidates)} usable bought-together bundles were found.')
    random.Random(seed).shuffle(candidates)
    return candidates[:max_bundles]


def encode_texts(records: list[BundleRecord], input_dim: int) -> dict[str, np.ndarray]:
    embeddings: dict[str, np.ndarray] = {}
    for field_name in ('title', 'description', 'category'):
        texts = [getattr(record, field_name) or '[missing]' for record in records]
        matrix = TfidfVectorizer(max_features=2048, stop_words='english').fit_transform(texts)
        components = min(input_dim, matrix.shape[0] - 1, matrix.shape[1] - 1)
        if components < 1:
            raise ValueError(f'Insufficient text variation in {field_name}.')
        dense = TruncatedSVD(n_components=components, random_state=2026).fit_transform(matrix)
        padded = np.zeros((len(records), input_dim), dtype=np.float32)
        padded[:, :components] = dense.astype(np.float32)
        embeddings[field_name] = padded
    return embeddings


def load_item_embeddings(data_root: Path) -> dict[str, np.ndarray]:
    processed = data_root / 'processed'
    with (processed / 'id_mapping.json').open() as file:
        id_mapping = json.load(file)
    vectors = np.fromfile(processed / 'sentence-t5-base.sent_emb', dtype=np.float32).reshape(-1, 128)
    return {asin: vectors[index - 1] for asin, index in id_mapping['item2id'].items() if index > 0}


def collate_batch(examples: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    max_items = max(example['items'].shape[0] for example in examples)
    input_dim = examples[0]['items'].shape[1]
    items = torch.zeros(len(examples), max_items, input_dim)
    item_mask = torch.zeros(len(examples), max_items, dtype=torch.bool)
    for index, example in enumerate(examples):
        length = example['items'].shape[0]
        items[index, :length] = example['items']
        item_mask[index, :length] = True
    return {
        'fields': {name: torch.stack([example[name] for example in examples]) for name in ('title', 'description', 'category')},
        'items': items,
        'item_mask': item_mask,
        'target': torch.stack([example['target'] for example in examples]),
    }


def contrastive_loss(representations: Tensor, targets: Tensor) -> Tensor:
    logits = functional.normalize(representations, dim=-1) @ functional.normalize(targets, dim=-1).T
    return functional.cross_entropy(logits / 0.07, torch.arange(logits.shape[0], device=logits.device))


def run_epoch(model: BundleEncoder, loader: DataLoader[dict[str, Tensor]], optimizer: torch.optim.Optimizer | None) -> float:
    losses: list[float] = []
    model.train(optimizer is not None)
    for batch in loader:
        output = model(batch['fields'], batch['items'], batch['item_mask'])
        loss = contrastive_loss(output.embedding, batch['target'])
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def evaluate_retrieval(model: BundleEncoder, loader: DataLoader[dict[str, Tensor]]) -> tuple[float, float]:
    model.eval()
    representations: list[Tensor] = []
    targets: list[Tensor] = []
    attentions: list[Tensor] = []
    with torch.no_grad():
        for batch in loader:
            output = model(batch['fields'], batch['items'], batch['item_mask'])
            representations.append(functional.normalize(output.embedding, dim=-1))
            targets.append(functional.normalize(batch['target'], dim=-1))
            attentions.append(output.item_attention)
    scores = torch.cat(representations) @ torch.cat(targets).T
    ranks = scores.argsort(dim=1, descending=True).argsort(dim=1).diagonal() + 1
    recall_at_10 = float((ranks <= 10).float().mean())
    mean_first_item_attention = float(torch.cat(attentions)[:, 0].mean())
    return recall_at_10, mean_first_item_attention


def main() -> None:
    args = parse_args()
    if args.max_bundles < 20:
        raise ValueError('--max-bundles must be at least 20.')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    records = load_records(args.data_root, args.max_bundles, args.max_items, args.seed)
    item_embeddings = load_item_embeddings(args.data_root)
    field_embeddings = encode_texts(records, input_dim=128)
    split = int(len(records) * 0.8)
    train_dataset = BundleDataset(records[:split], {name: values[:split] for name, values in field_embeddings.items()}, item_embeddings)
    validation_dataset = BundleDataset(records[split:], {name: values[split:] for name, values in field_embeddings.items()}, item_embeddings)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    model = BundleEncoder(input_dim=128, hidden_dim=128, output_dim=128, dropout=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_losses = [run_epoch(model, train_loader, optimizer) for _ in range(args.epochs)]
    validation_loss = run_epoch(model, validation_loader, optimizer=None)
    recall_at_10, mean_anchor_attention = evaluate_retrieval(model, validation_loader)

    result = {
        'dataset': 'AmazonReviews2014/Sports_and_Outdoors',
        'bundle_definition': 'anchor product plus raw metadata related.bought_together items',
        'max_bundles': args.max_bundles,
        'train_bundles': len(train_dataset),
        'validation_bundles': len(validation_dataset),
        'max_items_per_bundle': args.max_items,
        'seed': args.seed,
        'epochs': args.epochs,
        'train_losses': train_losses,
        'validation_contrastive_loss': validation_loss,
        'validation_anchor_recall_at_10': recall_at_10,
        'validation_mean_anchor_attention': mean_anchor_attention,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
