"""Train a balanced semantic-behavior RQ-VAE and export Semantic IDs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from pathfusionrec.tiger.content import load_vector_artifact, standardize_train_items
from pathfusionrec.tiger.rq_vae import FusionResidualQuantizedVAE, append_collision_codes


def _training_item_ids(split: dict[str, Any]) -> set[int]:
  items: set[int] = set()
  for sample in split['train']:
    items.update(int(item_id) for item_id in sample['history'])
    items.add(int(sample['target']))
  if not items:
    raise ValueError('The training split does not contain any item IDs.')
  return items


def run_fusion_rqvae(
    semantic_dir: Path,
    behavior_dir: Path,
    split_path: Path,
    output_dir: Path,
    *,
    epochs: int = 100,
    batch_size: int = 256,
    branch_dim: int = 128,
    latent_dim: int = 32,
    codebook_sizes: Sequence[int] = (4, 16, 256),
    lr: float = 1e-3,
    commitment_weight: float = 0.25,
    seed: int = 2026,
    initialize_codebooks: bool = True,
    device: str = 'cuda',
) -> dict[str, Any]:
  """Fit one shared RQ-VAE with equal semantic and behavior reconstruction loss."""
  if min(epochs, batch_size, branch_dim, latent_dim) <= 0:
    raise ValueError('Epochs, batch size, branch dimension, and latent dimension must be positive.')
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  semantic_vectors, semantic_item_ids, semantic_manifest = load_vector_artifact(semantic_dir)
  behavior_vectors, behavior_item_ids, behavior_manifest = load_vector_artifact(behavior_dir)
  if semantic_item_ids != behavior_item_ids:
    raise ValueError('Semantic and behavior artifacts must use identical ordered item IDs.')
  split = json.loads(split_path.read_text(encoding='utf-8'))
  item_to_row = {item_id: row for row, item_id in enumerate(semantic_item_ids)}
  train_item_ids = _training_item_ids(split)
  missing = train_item_ids.difference(item_to_row)
  if missing:
    raise ValueError(f'Training items missing from vector artifacts: {sorted(missing)[:5]}')
  train_rows = np.array(sorted(item_to_row[item_id] for item_id in train_item_ids))
  standardized_semantic, semantic_statistics = standardize_train_items(semantic_vectors, train_rows)
  standardized_behavior, behavior_statistics = standardize_train_items(behavior_vectors, train_rows)

  torch_device = torch.device(device)
  model = FusionResidualQuantizedVAE(
      semantic_vectors.shape[1],
      behavior_vectors.shape[1],
      branch_dim,
      latent_dim,
      codebook_sizes,
      commitment_weight,
  ).to(torch_device)
  train_semantic = torch.from_numpy(standardized_semantic[train_rows])
  train_behavior = torch.from_numpy(standardized_behavior[train_rows])
  if initialize_codebooks:
    model.initialize_codebooks_kmeans(train_semantic, train_behavior, seed)
  loader = DataLoader(TensorDataset(train_semantic, train_behavior), batch_size=batch_size, shuffle=True)
  optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
  history: list[dict[str, float | int]] = []
  for epoch in range(1, epochs + 1):
    model.train()
    losses: list[float] = []
    semantic_losses: list[float] = []
    behavior_losses: list[float] = []
    for semantic_batch, behavior_batch in loader:
      output = model(semantic_batch.to(torch_device), behavior_batch.to(torch_device))
      optimizer.zero_grad()
      output.loss.backward()
      optimizer.step()
      losses.append(float(output.loss.detach().cpu()))
      semantic_losses.append(float(output.semantic_reconstruction_loss.detach().cpu()))
      behavior_losses.append(float(output.behavior_reconstruction_loss.detach().cpu()))
    history.append(
        {
            'epoch': epoch,
            'loss': float(np.mean(losses)),
            'semantic_reconstruction_loss': float(np.mean(semantic_losses)),
            'behavior_reconstruction_loss': float(np.mean(behavior_losses)),
        }
    )

  full_semantic = torch.from_numpy(standardized_semantic).to(torch_device)
  full_behavior = torch.from_numpy(standardized_behavior).to(torch_device)
  codes = model.semantic_ids(full_semantic, full_behavior)
  semantic_ids = append_collision_codes(semantic_item_ids, codes)
  if len(set(semantic_ids)) != len(semantic_item_ids):
    raise RuntimeError('Collision codes did not produce unique Semantic IDs.')
  diagnostics = model.codebook_diagnostics(codes)
  if any(int(report['used_codes']) == 0 for report in diagnostics):
    raise RuntimeError('An RQ-VAE codebook is completely unused.')

  output_dir.mkdir(parents=True, exist_ok=True)
  semantic_id_map = {
      str(item_id): list(semantic_id)
      for item_id, semantic_id in zip(semantic_item_ids, semantic_ids, strict=True)
  }
  reverse_map = {
      ','.join(str(value) for value in semantic_id): item_id
      for item_id, semantic_id in zip(semantic_item_ids, semantic_ids, strict=True)
  }
  torch.save(
      {
          'state_dict': model.state_dict(),
          'semantic_dim': model.semantic_dim,
          'behavior_dim': model.behavior_dim,
          'branch_dim': model.branch_dim,
          'latent_dim': model.latent_dim,
          'codebook_sizes': model.codebook_sizes,
          'commitment_weight': model.commitment_weight,
      },
      output_dir / 'rqvae_checkpoint.pt',
  )
  (output_dir / 'semantic_ids.json').write_text(
      json.dumps(semantic_id_map, indent=2, sort_keys=True) + '\n', encoding='utf-8'
  )
  (output_dir / 'semantic_id_index.json').write_text(
      json.dumps(reverse_map, indent=2) + '\n', encoding='utf-8'
  )
  (output_dir / 'standardization.json').write_text(
      json.dumps(
          {
              'semantic': {key: value.tolist() for key, value in semantic_statistics.items()},
              'behavior': {key: value.tolist() for key, value in behavior_statistics.items()},
          },
          indent=2,
      )
      + '\n',
      encoding='utf-8',
  )
  (output_dir / 'codebook_diagnostics.json').write_text(
      json.dumps(diagnostics, indent=2) + '\n', encoding='utf-8'
  )
  configuration = {
      'semantic_manifest': semantic_manifest,
      'behavior_manifest': behavior_manifest,
      'semantic_dimension': semantic_vectors.shape[1],
      'behavior_dimension': behavior_vectors.shape[1],
      'branch_dim': branch_dim,
      'latent_dim': latent_dim,
      'codebook_sizes': list(codebook_sizes),
      'epochs': epochs,
      'batch_size': batch_size,
      'lr': lr,
      'commitment_weight': commitment_weight,
      'seed': seed,
      'device': device,
      'training_item_count': len(train_item_ids),
  }
  (output_dir / 'config.json').write_text(
      json.dumps(configuration, indent=2, sort_keys=True) + '\n', encoding='utf-8'
  )
  (output_dir / 'training_history.json').write_text(
      json.dumps(history, indent=2) + '\n', encoding='utf-8'
  )
  (output_dir / 'delivery_validation.txt').write_text(
      'RESULT VALIDATION: PASS\n'
      f'catalog_items: {len(semantic_item_ids)}\n'
      f'unique_semantic_ids: {len(set(semantic_ids))}\n'
      f'codebook_used: {[report["used_codes"] for report in diagnostics]}\n',
      encoding='utf-8',
  )
  return {'semantic_ids': semantic_id_map, 'diagnostics': diagnostics, 'history': history}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--semantic-dir', type=Path, required=True)
  parser.add_argument('--behavior-dir', type=Path, required=True)
  parser.add_argument('--protocol-dir', type=Path, required=True)
  parser.add_argument('--output-dir', type=Path, required=True)
  parser.add_argument('--epochs', type=int, default=100)
  parser.add_argument('--batch-size', type=int, default=256)
  parser.add_argument('--branch-dim', type=int, default=128)
  parser.add_argument('--latent-dim', type=int, default=32)
  parser.add_argument('--codebook-sizes', type=int, nargs=3, default=[4, 16, 256])
  parser.add_argument('--lr', type=float, default=1e-3)
  parser.add_argument('--commitment-weight', type=float, default=0.25)
  parser.add_argument('--seed', type=int, default=2026)
  parser.add_argument('--device', default='cuda')
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  run_fusion_rqvae(
      args.semantic_dir,
      args.behavior_dir,
      args.protocol_dir / 'split.json',
      args.output_dir,
      epochs=args.epochs,
      batch_size=args.batch_size,
      branch_dim=args.branch_dim,
      latent_dim=args.latent_dim,
      codebook_sizes=args.codebook_sizes,
      lr=args.lr,
      commitment_weight=args.commitment_weight,
      seed=args.seed,
      device=args.device,
  )


if __name__ == '__main__':
  main()
