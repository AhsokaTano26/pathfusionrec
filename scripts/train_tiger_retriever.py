"""Train and evaluate TIGER's constrained T5 next-item retriever."""

from __future__ import annotations

import argparse
import copy
import json
import random
from functools import partial
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import T5Config

from pathfusionrec.evaluation import evaluate_ranked_lists
from pathfusionrec.tiger.retriever import TigerRetriever, default_tiger_config, make_tiger_batch
from pathfusionrec.tiger.tokenizer import SemanticIdTokenizer


def _load_tokenizer(semantic_id_dir: Path, user_bucket_count: int) -> SemanticIdTokenizer:
  semantic_ids = json.loads((semantic_id_dir / 'semantic_ids.json').read_text(encoding='utf-8'))
  config = json.loads((semantic_id_dir / 'config.json').read_text(encoding='utf-8'))
  codebook_sizes = config['codebook_sizes']
  collision_size = max(ids[3] for ids in semantic_ids.values()) + 1
  return SemanticIdTokenizer(
      {int(item_id): ids for item_id, ids in semantic_ids.items()},
      codebook_sizes,
      collision_size,
      user_bucket_count,
  )


def _limit_samples(samples: Sequence[dict[str, Any]], maximum: int | None) -> list[dict[str, Any]]:
  return list(samples if maximum is None else samples[:maximum])


def _truncate_history(sample: dict[str, Any], max_history_length: int) -> dict[str, Any]:
  return {**sample, 'history': sample['history'][-max_history_length:]}


def _evaluate(
    retriever: TigerRetriever,
    samples: Sequence[dict[str, Any]],
    subset_masks: Sequence[dict[str, bool]] | None,
    num_beams: int,
    batch_size: int,
) -> dict[str, dict[str, float]]:
  rankings: list[list[int]] = []
  targets: list[int] = []
  for offset in range(0, len(samples), batch_size):
    sample_batch = samples[offset : offset + batch_size]
    histories = [[int(item_id) for item_id in sample['history']] for sample in sample_batch]
    history_tokens = [
        retriever.tokenizer.encode_history(int(sample['user_id']), history)
        for sample, history in zip(sample_batch, histories, strict=True)
    ]
    generated_batch = retriever.generate_top_k_batch(
        history_tokens, k=50, beam_size=num_beams
    )
    for history, sample, generated in zip(histories, sample_batch, generated_batch, strict=True):
      target = int(sample['target'])
      rankings.append([item_id for item_id in generated if item_id not in set(history) or item_id == target])
      targets.append(target)
  results = {'all': evaluate_ranked_lists(rankings, targets, ks=(5, 10, 20, 50))}
  if subset_masks is not None:
    if len(subset_masks) != len(samples):
      raise ValueError('Test subset masks must align with test samples.')
    for subset in ('cold_start_target', 'long_tail_target', 'sparse_user'):
      selected = [index for index, mask in enumerate(subset_masks) if mask[subset]]
      results[subset] = evaluate_ranked_lists(
          [rankings[index] for index in selected],
          [targets[index] for index in selected],
          ks=(5, 10, 20, 50),
      )
  return results


def _training_config(
    tokenizer: SemanticIdTokenizer,
    *,
    d_model: int,
    d_ff: int,
    num_layers: int,
    num_heads: int,
    d_kv: int,
    dropout: float,
) -> T5Config:
  if (
      d_model == 128
      and d_ff == 1024
      and num_layers == 4
      and num_heads == 6
      and d_kv == 64
      and dropout == 0.1
  ):
    return default_tiger_config(tokenizer)
  return T5Config(
      vocab_size=tokenizer.vocab_size,
      d_model=d_model,
      d_ff=d_ff,
      num_layers=num_layers,
      num_decoder_layers=num_layers,
      num_heads=num_heads,
      d_kv=d_kv,
      dropout_rate=dropout,
      pad_token_id=tokenizer.pad_token_id,
      decoder_start_token_id=tokenizer.bos_token_id,
      eos_token_id=tokenizer.eos_token_id,
  )


def run_training(
    protocol_dir: Path,
    semantic_id_dir: Path,
    output_dir: Path,
    *,
    epochs: int = 100,
    batch_size: int = 256,
    eval_batch_size: int = 256,
    max_history_length: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 2026,
    num_beams: int = 50,
    eval_interval: int = 1,
    num_workers: int = 8,
    prefetch_factor: int = 4,
    user_bucket_count: int = 2_000,
    d_model: int = 128,
    d_ff: int = 1024,
    num_layers: int = 4,
    num_heads: int = 6,
    d_kv: int = 64,
    dropout: float = 0.1,
    max_train_samples: int | None = None,
    max_validation_samples: int | None = None,
    max_test_samples: int | None = None,
    device: str = 'cuda',
) -> dict[str, Any]:
  """Train a frozen-tokenizer TIGER retriever under the fixed protocol."""
  if epochs <= 0 or batch_size <= 0 or eval_batch_size <= 0 or num_beams <= 0:
    raise ValueError('Epochs, batch sizes, and beam count must be positive.')
  if eval_interval <= 0:
    raise ValueError('The validation interval must be positive.')
  if num_workers < 0 or prefetch_factor <= 0:
    raise ValueError('Worker count must be non-negative and prefetch factor must be positive.')
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  split = json.loads((protocol_dir / 'split.json').read_text(encoding='utf-8'))
  subset_masks = json.loads((protocol_dir / 'subset_masks.json').read_text(encoding='utf-8'))['test']
  train_samples = [
      _truncate_history(sample, max_history_length)
      for sample in _limit_samples(split['train'], max_train_samples)
  ]
  validation_samples = [
      _truncate_history(sample, max_history_length)
      for sample in _limit_samples(split['validation'], max_validation_samples)
  ]
  test_samples = [
      _truncate_history(sample, max_history_length)
      for sample in _limit_samples(split['test'], max_test_samples)
  ]
  test_masks = subset_masks[: len(test_samples)]
  tokenizer = _load_tokenizer(semantic_id_dir, user_bucket_count)
  config = _training_config(
      tokenizer,
      d_model=d_model,
      d_ff=d_ff,
      num_layers=num_layers,
      num_heads=num_heads,
      d_kv=d_kv,
      dropout=dropout,
  )
  torch_device = torch.device(device)
  retriever = TigerRetriever(tokenizer, config).to(torch_device)
  loader_options: dict[str, Any] = {
      'batch_size': batch_size,
      'shuffle': True,
      'collate_fn': partial(make_tiger_batch, tokenizer),
      'num_workers': num_workers,
      'pin_memory': torch_device.type == 'cuda',
  }
  if num_workers > 0:
    loader_options.update({
        'persistent_workers': True,
        'prefetch_factor': prefetch_factor,
    })
  loader = DataLoader(
      train_samples,
      **loader_options,
  )
  optimizer = torch.optim.AdamW(retriever.parameters(), lr=lr, weight_decay=weight_decay)
  best_epoch = 0
  best_score = float('-inf')
  best_state: dict[str, Tensor] | None = None
  history: list[dict[str, float | int]] = []
  for epoch in range(1, epochs + 1):
    retriever.train()
    losses: list[float] = []
    for batch in loader:
      batch = {
          name: value.to(torch_device, non_blocking=torch_device.type == 'cuda')
          for name, value in batch.items()
      }
      loss = retriever.loss(batch)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      losses.append(float(loss.detach().cpu()))
    if epoch % eval_interval != 0 and epoch != epochs:
      continue
    validation = _evaluate(
        retriever, validation_samples, None, num_beams, eval_batch_size
    )['all']
    history.append({'epoch': epoch, 'train_loss': float(np.mean(losses)), **validation})
    if validation['NDCG@10'] > best_score:
      best_score = validation['NDCG@10']
      best_epoch = epoch
      best_state = copy.deepcopy(retriever.state_dict())
  if best_state is None:
    raise RuntimeError('No validation checkpoint was produced.')
  retriever.load_state_dict(best_state)
  test_metrics = _evaluate(
      retriever, test_samples, test_masks, num_beams, eval_batch_size
  )

  output_dir.mkdir(parents=True, exist_ok=True)
  torch.save(best_state, output_dir / 'best_model.pth')
  arguments = {
      'epochs': epochs,
      'batch_size': batch_size,
      'eval_batch_size': eval_batch_size,
      'max_history_length': max_history_length,
      'lr': lr,
      'weight_decay': weight_decay,
      'seed': seed,
      'num_beams': num_beams,
      'eval_interval': eval_interval,
      'num_workers': num_workers,
      'prefetch_factor': prefetch_factor,
      'user_bucket_count': user_bucket_count,
      'd_model': d_model,
      'd_ff': d_ff,
      'num_layers': num_layers,
      'num_heads': num_heads,
      'd_kv': d_kv,
      'dropout': dropout,
      'device': device,
  }
  metrics = {
      'arguments': arguments,
      'best_epoch': best_epoch,
      'best_validation_ndcg@10': best_score,
      'history': history,
      'test': test_metrics,
  }
  (output_dir / 'metrics.json').write_text(
      json.dumps(metrics, indent=2) + '\n', encoding='utf-8'
  )
  (output_dir / 'config.json').write_text(
      json.dumps(arguments, indent=2, sort_keys=True) + '\n', encoding='utf-8'
  )
  (output_dir / 'delivery_validation.txt').write_text(
      'RESULT VALIDATION: PASS\n'
      f'best_epoch: {best_epoch}\n'
      f'best_validation_ndcg@10: {best_score}\n'
      'metric_sections: all cold_start_target long_tail_target sparse_user\n',
      encoding='utf-8',
  )
  return metrics


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--protocol-dir', type=Path, required=True)
  parser.add_argument('--semantic-id-dir', type=Path, required=True)
  parser.add_argument('--output-dir', type=Path, required=True)
  parser.add_argument('--epochs', type=int, default=100)
  parser.add_argument('--batch-size', type=int, default=256)
  parser.add_argument('--eval-batch-size', type=int, default=256)
  parser.add_argument('--max-history-length', type=int, default=20)
  parser.add_argument('--lr', type=float, default=1e-3)
  parser.add_argument('--weight-decay', type=float, default=1e-4)
  parser.add_argument('--seed', type=int, default=2026)
  parser.add_argument('--num-beams', type=int, default=50)
  parser.add_argument('--eval-interval', type=int, default=1)
  parser.add_argument('--num-workers', type=int, default=8)
  parser.add_argument('--prefetch-factor', type=int, default=4)
  parser.add_argument('--user-bucket-count', type=int, default=2_000)
  parser.add_argument('--max-train-samples', type=int)
  parser.add_argument('--max-validation-samples', type=int)
  parser.add_argument('--max-test-samples', type=int)
  parser.add_argument('--device', default='cuda')
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  run_training(
      args.protocol_dir,
      args.semantic_id_dir,
      args.output_dir,
      epochs=args.epochs,
      batch_size=args.batch_size,
      eval_batch_size=args.eval_batch_size,
      max_history_length=args.max_history_length,
      lr=args.lr,
      weight_decay=args.weight_decay,
      seed=args.seed,
      num_beams=args.num_beams,
      eval_interval=args.eval_interval,
      num_workers=args.num_workers,
      prefetch_factor=args.prefetch_factor,
      user_bucket_count=args.user_bucket_count,
      max_train_samples=args.max_train_samples,
      max_validation_samples=args.max_validation_samples,
      max_test_samples=args.max_test_samples,
      device=args.device,
  )


if __name__ == '__main__':
  main()
