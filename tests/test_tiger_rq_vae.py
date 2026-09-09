"""Tests for TIGER residual-quantized VAE semantic IDs."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from pathfusionrec.tiger.rq_vae import (
    FusionResidualQuantizedVAE,
    ResidualQuantizedVAE,
    append_collision_codes,
)


class ResidualQuantizedVaeTest(unittest.TestCase):

  def test_rq_vae_emits_one_code_per_residual_level(self) -> None:
    model = ResidualQuantizedVAE(8, 4, [2, 3, 5])
    output = model(torch.randn(6, 8))

    self.assertEqual(output.reconstruction.shape, (6, 8))
    self.assertEqual(output.codes.shape, (6, 3))
    self.assertTrue(torch.all(output.codes[:, 0] < 2))
    self.assertTrue(torch.all(output.codes[:, 1] < 3))
    self.assertTrue(torch.all(output.codes[:, 2] < 5))

  def test_backward_updates_encoder(self) -> None:
    model = ResidualQuantizedVAE(8, 4, [2, 3, 5])
    model(torch.randn(6, 8)).loss.backward()

    self.assertIsNotNone(model.encoder[0].weight.grad)

  def test_collision_code_makes_duplicate_tuples_unique(self) -> None:
    semantic_ids = append_collision_codes(
        [9, 4], np.array([[1, 2, 3], [1, 2, 3]], dtype=np.int64)
    )

    self.assertEqual(semantic_ids, [(1, 2, 3, 0), (1, 2, 3, 1)])

  def test_fusion_rqvae_balances_two_modal_reconstructions(self) -> None:
    model = FusionResidualQuantizedVAE(6, 2, 4, 3, [2, 3, 5])
    output = model(torch.randn(6, 6), torch.randn(6, 2))

    self.assertEqual(output.semantic_reconstruction.shape, (6, 6))
    self.assertEqual(output.behavior_reconstruction.shape, (6, 2))
    self.assertEqual(output.codes.shape, (6, 3))
    output.loss.backward()
    self.assertIsNotNone(model.semantic_encoder[0].weight.grad)
    self.assertIsNotNone(model.behavior_encoder[0].weight.grad)


if __name__ == '__main__':
  unittest.main()
