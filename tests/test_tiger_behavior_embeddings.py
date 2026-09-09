"""Tests for extracting frozen PFR behavior vectors for TIGER."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from pathfusionrec.tiger.content import load_vector_artifact
from scripts.build_tiger_behavior_embeddings import build_behavior_artifact


class TigerBehaviorEmbeddingTest(unittest.TestCase):

  def test_extracts_catalog_rows_without_padding_from_pfr_checkpoint(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      checkpoint_path = root / 'pfr.pth'
      weights = torch.arange(20, dtype=torch.float32).reshape(5, 4)
      torch.save({'interaction_embedding.weight': weights}, checkpoint_path)

      build_behavior_artifact(checkpoint_path, [1, 2, 3, 4], root / 'behavior')

      vectors, item_ids, manifest = load_vector_artifact(root / 'behavior')
      np.testing.assert_array_equal(vectors, weights[1:].numpy())
      self.assertEqual(item_ids, [1, 2, 3, 4])
      self.assertEqual(manifest['representation'], 'pfr_interaction_embedding')
      self.assertEqual(manifest['dimension'], 4)
