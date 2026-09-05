"""Build a non-PCA 768-dimensional Sentence-T5 artifact for TIGER."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pathfusionrec.tiger.content import (
    CONTENT_DIMENSION,
    load_ordered_item_texts,
    save_content_artifact,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--processed-dir', type=Path, required=True)
  parser.add_argument('--output-dir', type=Path, required=True)
  parser.add_argument('--model', default='sentence-transformers/sentence-t5-base')
  parser.add_argument('--model-revision', default='fc5d4628481afbbaaacd7af6bb07cf9d3865f781')
  parser.add_argument('--batch-size', type=int, default=128)
  parser.add_argument('--device', default='cuda')
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  try:
    from sentence_transformers import SentenceTransformer
  except ImportError as error:
    raise RuntimeError(
        'sentence-transformers is required; install the project dependencies.'
    ) from error

  item_ids, sentences = load_ordered_item_texts(args.processed_dir)
  encoder = SentenceTransformer(args.model, revision=args.model_revision, device=args.device)
  vectors = encoder.encode(
      sentences,
      batch_size=args.batch_size,
      convert_to_numpy=True,
      show_progress_bar=True,
      device=args.device,
  ).astype(np.float32, copy=False)
  if vectors.shape != (len(item_ids), CONTENT_DIMENSION):
    raise ValueError(
        f'Expected [{len(item_ids)}, {CONTENT_DIMENSION}] Sentence-T5 vectors, '
        f'got {list(vectors.shape)}.'
    )
  save_content_artifact(
      args.output_dir,
      vectors,
      item_ids,
      {
          'model': args.model,
          'model_revision': args.model_revision,
          'metadata_sentence_sha256': sha256_file(
              args.processed_dir / 'metadata.sentence.json'
          ),
          'item_mapping_sha256': sha256_file(args.processed_dir / 'id_mapping.json'),
      },
  )


if __name__ == '__main__':
  main()
