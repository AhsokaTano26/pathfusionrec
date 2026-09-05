"""Integration tests for TIGER RQ-VAE artifact generation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pathfusionrec.tiger.content import save_content_artifact
from scripts.train_tiger_rqvae import run_rqvae


class TigerRqvaeScriptTest(unittest.TestCase):

  def test_rqvae_run_writes_unique_catalog_semantic_ids(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      content_dir = root / 'content'
      item_ids = list(range(1, 9))
      save_content_artifact(
          content_dir,
          np.random.default_rng(7).normal(size=(8, 768)).astype(np.float32),
          item_ids,
          {'model': 'test'},
      )
      split_path = root / 'split.json'
      split_path.write_text(
          json.dumps(
              {
                  'train': [
                      {'history': [1, 2], 'target': 3},
                      {'history': [4, 5], 'target': 6},
                  ]
              }
          ),
          encoding='utf-8',
      )

      run_rqvae(
          content_dir,
          split_path,
          root / 'output',
          epochs=1,
          batch_size=2,
          codebook_sizes=(2, 2, 2),
          initialize_codebooks=False,
          device='cpu',
      )

      semantic_ids = json.loads((root / 'output' / 'semantic_ids.json').read_text())
      self.assertEqual(len(semantic_ids), 8)
      self.assertEqual(len({tuple(ids) for ids in semantic_ids.values()}), 8)


if __name__ == '__main__':
  unittest.main()
