"""Unit tests for the structured bundle semantic encoder."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from pathfusionrec import BundleEncoder


class BundleEncoderTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.encoder = BundleEncoder(
            input_dim=8,
            hidden_dim=12,
            output_dim=10,
            role_feature_dim=2,
            dropout=0.0,
        )
        self.fields = {
            'title': torch.randn(2, 8),
            'description': torch.randn(2, 8),
            'category': torch.randn(2, 8),
        }
        self.items = torch.randn(2, 4, 8)
        self.mask = torch.tensor([[True, True, False, False], [True, True, True, False]])
        self.roles = torch.randn(2, 4, 2)

    def test_returns_expected_shapes_and_normalized_attention(self) -> None:
        output = self.encoder(self.fields, self.items, self.mask, self.roles)

        self.assertEqual(output.embedding.shape, (2, 10))
        self.assertEqual(output.global_embedding.shape, (2, 12))
        self.assertEqual(output.local_embedding.shape, (2, 12))
        self.assertEqual(output.field_attention.shape, (2, 3))
        self.assertEqual(output.item_attention.shape, (2, 4))
        torch.testing.assert_close(output.field_attention.sum(dim=1), torch.ones(2))
        torch.testing.assert_close(output.item_attention.sum(dim=1), torch.ones(2))
        torch.testing.assert_close(output.item_attention[~self.mask], torch.zeros(3))

    def test_rejects_a_bundle_without_items(self) -> None:
        invalid_mask = self.mask.clone()
        invalid_mask[1] = False

        with self.assertRaisesRegex(ValueError, 'at least one'):
            self.encoder(self.fields, self.items, invalid_mask, self.roles)

    def test_role_features_require_role_configuration(self) -> None:
        encoder = BundleEncoder(input_dim=8, hidden_dim=12, output_dim=10, dropout=0.0)

        with self.assertRaisesRegex(ValueError, 'role_feature_dim'):
            encoder(self.fields, self.items, self.mask, self.roles)


if __name__ == '__main__':
    unittest.main()
