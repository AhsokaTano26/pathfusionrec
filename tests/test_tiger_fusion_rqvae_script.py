"""Integration tests for multimodal TIGER RQ-VAE tokenization."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pathfusionrec.tiger.content import save_vector_artifact
from scripts.train_tiger_fusion_rqvae import run_fusion_rqvae


class TigerFusionRqvaeScriptTest(unittest.TestCase):

  def test_fusion_runner_writes_unique_catalog_semantic_ids(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      item_ids = list(range(1, 9))
      generator = np.random.default_rng(9)
      save_vector_artifact(
          root / 'semantic', generator.normal(size=(8, 6)).astype(np.float32), item_ids, {}
      )
      save_vector_artifact(
          root / 'behavior', generator.normal(size=(8, 2)).astype(np.float32), item_ids, {}
      )
      split_path = root / 'split.json'
      split_path.write_text(
          json.dumps({'train': [{'history': [1, 2], 'target': 3}, {'history': [4, 5], 'target': 6}]}),
          encoding='utf-8',
      )

      run_fusion_rqvae(
          root / 'semantic',
          root / 'behavior',
          split_path,
          root / 'output',
          epochs=1,
          batch_size=2,
          branch_dim=4,
          latent_dim=3,
          codebook_sizes=(2, 2, 2),
          initialize_codebooks=False,
          device='cpu',
      )

      semantic_ids = json.loads((root / 'output' / 'semantic_ids.json').read_text())
      self.assertEqual(len(semantic_ids), 8)
      self.assertEqual(len({tuple(values) for values in semantic_ids.values()}), 8)
      config = json.loads((root / 'output' / 'config.json').read_text())
      self.assertEqual(config['semantic_dimension'], 6)
      self.assertEqual(config['behavior_dimension'], 2)
      self.assertEqual(config['branch_dim'], 4)
