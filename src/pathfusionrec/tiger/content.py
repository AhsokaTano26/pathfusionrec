"""Content embedding artifacts for TIGER semantic-ID training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


CONTENT_DIMENSION = 768


def sha256_file(path: Path) -> str:
  """Return the SHA-256 digest of a file."""
  digest = hashlib.sha256()
  with path.open('rb') as file:
    for block in iter(lambda: file.read(1024 * 1024), b''):
      digest.update(block)
  return digest.hexdigest()


def load_ordered_item_texts(processed_dir: Path) -> tuple[list[int], list[str]]:
  """Load item sentences in the canonical numeric item-ID order."""
  mapping_path = processed_dir / 'id_mapping.json'
  metadata_path = processed_dir / 'metadata.sentence.json'
  mapping = json.loads(mapping_path.read_text(encoding='utf-8'))
  item_sentences = json.loads(metadata_path.read_text(encoding='utf-8'))
  id_to_item = mapping['id2item']

  item_ids: list[int] = []
  sentences: list[str] = []
  for item_id in range(1, len(id_to_item)):
    item = id_to_item[item_id]
    sentence = item_sentences.get(item)
    if not isinstance(sentence, str) or not sentence.strip():
      raise ValueError(f'Missing item sentence for numeric item ID {item_id}.')
    item_ids.append(item_id)
    sentences.append(sentence)
  return item_ids, sentences


def save_content_artifact(
    output_dir: Path,
    vectors: np.ndarray,
    item_ids: Sequence[int],
    manifest: dict[str, Any],
) -> None:
  """Save a validated Sentence-T5 content artifact."""
  if vectors.dtype != np.float32 or vectors.ndim != 2:
    raise ValueError('Sentence-T5 content vectors must be a float32 matrix.')
  if vectors.shape[1] != CONTENT_DIMENSION:
    raise ValueError(
        f'Sentence-T5 content vectors must have dimension {CONTENT_DIMENSION}.'
    )
  if vectors.shape[0] != len(item_ids) or len(set(item_ids)) != len(item_ids):
    raise ValueError('Content vectors and unique item IDs must have equal length.')

  output_dir.mkdir(parents=True, exist_ok=True)
  np.save(output_dir / 'vectors.npy', vectors)
  (output_dir / 'item_ids.json').write_text(
      json.dumps([int(item_id) for item_id in item_ids]) + '\n',
      encoding='utf-8',
  )
  artifact_manifest = {
      **manifest,
      'dtype': 'float32',
      'dimension': CONTENT_DIMENSION,
      'num_items': len(item_ids),
  }
  (output_dir / 'manifest.json').write_text(
      json.dumps(artifact_manifest, indent=2, sort_keys=True) + '\n',
      encoding='utf-8',
  )


def load_content_artifact(
    content_dir: Path,
) -> tuple[np.ndarray, list[int], dict[str, Any]]:
  """Load and validate a previously generated content artifact."""
  vectors = np.load(content_dir / 'vectors.npy')
  item_ids = json.loads((content_dir / 'item_ids.json').read_text(encoding='utf-8'))
  manifest = json.loads((content_dir / 'manifest.json').read_text(encoding='utf-8'))
  if vectors.dtype != np.float32 or vectors.ndim != 2:
    raise ValueError('Content artifact vectors must be a float32 matrix.')
  if vectors.shape[1] != CONTENT_DIMENSION:
    raise ValueError(f'Content artifact must have dimension {CONTENT_DIMENSION}.')
  if vectors.shape[0] != len(item_ids) or len(set(item_ids)) != len(item_ids):
    raise ValueError('Content artifact item IDs do not match vectors.')
  if manifest.get('dimension') != CONTENT_DIMENSION:
    raise ValueError('Content artifact manifest has an invalid dimension.')
  return vectors, [int(item_id) for item_id in item_ids], manifest


def standardize_train_items(
    vectors: np.ndarray, train_rows: np.ndarray
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
  """Standardize vectors with statistics fitted on training-item rows only."""
  if vectors.ndim != 2 or vectors.dtype != np.float32:
    raise ValueError('Vectors must be a two-dimensional float32 array.')
  if train_rows.ndim != 1 or train_rows.size == 0:
    raise ValueError('At least one training-item row is required.')
  if train_rows.min() < 0 or train_rows.max() >= vectors.shape[0]:
    raise ValueError('Training-item rows are outside the content artifact.')

  training_vectors = vectors[train_rows]
  mean = training_vectors.mean(axis=0, dtype=np.float64).astype(np.float32)
  std = training_vectors.std(axis=0, dtype=np.float64).astype(np.float32)
  std = np.where(std > 0.0, std, np.float32(1.0))
  return (vectors - mean) / std, {'mean': mean, 'std': std}
