"""Extract frozen PFR interaction embeddings as TIGER behavior vectors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from pathfusionrec.tiger.content import (
    load_ordered_item_texts,
    save_vector_artifact,
    sha256_file,
)


INTERACTION_WEIGHT_KEY = 'interaction_embedding.weight'


def _state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
  if not isinstance(checkpoint, dict):
    raise ValueError('PFR checkpoint must contain a state dictionary.')
  state_dict = checkpoint.get('state_dict', checkpoint)
  if not isinstance(state_dict, dict):
    raise ValueError('PFR checkpoint state dictionary is invalid.')
  return state_dict


def build_behavior_artifact(
    checkpoint_path: Path,
    item_ids: Sequence[int],
    output_dir: Path,
) -> None:
  """Write behavior vectors in the supplied canonical catalog order."""
  try:
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
  except TypeError:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
  weights = _state_dict(checkpoint).get(INTERACTION_WEIGHT_KEY)
  if not isinstance(weights, torch.Tensor) or weights.ndim != 2:
    raise ValueError(f'Checkpoint is missing a rank-2 {INTERACTION_WEIGHT_KEY!r} tensor.')
  numeric_item_ids = [int(item_id) for item_id in item_ids]
  if not numeric_item_ids or min(numeric_item_ids) <= 0:
    raise ValueError('Catalog item IDs must be positive integers.')
  if max(numeric_item_ids) >= weights.shape[0]:
    raise ValueError('Checkpoint interaction embedding does not cover the full catalog.')
  vectors = weights[numeric_item_ids].detach().cpu().numpy().astype(np.float32, copy=False)
  save_vector_artifact(
      output_dir,
      vectors,
      numeric_item_ids,
      {
          'representation': 'pfr_interaction_embedding',
          'source_checkpoint': checkpoint_path.name,
          'source_checkpoint_sha256': sha256_file(checkpoint_path),
          'source_tensor': INTERACTION_WEIGHT_KEY,
      },
  )


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--checkpoint', type=Path, required=True)
  parser.add_argument('--processed-dir', type=Path, required=True)
  parser.add_argument('--output-dir', type=Path, required=True)
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  item_ids, _ = load_ordered_item_texts(args.processed_dir)
  build_behavior_artifact(args.checkpoint, item_ids, args.output_dir)
  manifest = json.loads((args.output_dir / 'manifest.json').read_text(encoding='utf-8'))
  print(json.dumps(manifest, sort_keys=True))


if __name__ == '__main__':
  main()
