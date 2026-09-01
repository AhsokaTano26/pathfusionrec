"""Tests for the unified semantic-interaction next-item model."""

from __future__ import annotations

import unittest

import torch

from pathfusionrec import PathFusionNextItemModel


def candidate_inputs() -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    return (
        torch.tensor([1, 2, 3, 4]),
        {name: torch.randn(4, 6) for name in ('title', 'description', 'category')},
        torch.randn(4, 3, 6),
        torch.tensor([[True, True, False], [True, False, False], [True, True, True], [True, True, False]]),
    )


class PathFusionNextItemModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.history_ids = torch.tensor([[1, 2, 0], [3, 4, 5]])
        self.history_semantics = torch.randn(2, 3, 6)
        self.history_mask = self.history_ids.ne(0)
        self.candidate_ids, self.fields, self.bundle_items, self.bundle_mask = candidate_inputs()

    def test_all_modes_return_rankable_vectors(self) -> None:
        for mode in ('semantic', 'interaction', 'fusion'):
            model = PathFusionNextItemModel(
                num_items=8,
                semantic_dim=6,
                hidden_dim=5,
                mode=mode,
            )
            output = model(
                self.history_ids,
                self.history_semantics,
                self.history_mask,
                self.candidate_ids,
                self.fields,
                self.bundle_items,
                self.bundle_mask,
            )
            self.assertEqual(output.query.shape, (2, 5))
            self.assertEqual(output.candidates.shape, (4, 5))
            self.assertEqual(PathFusionNextItemModel.score(output).shape, (2, 4))

    def test_gated_fusion_exposes_gate_values(self) -> None:
        model = PathFusionNextItemModel(
            num_items=8,
            semantic_dim=6,
            hidden_dim=5,
            mode='fusion',
            fusion_strategy='gated',
        )
        output = model(
            self.history_ids,
            self.history_semantics,
            self.history_mask,
            self.candidate_ids,
            self.fields,
            self.bundle_items,
            self.bundle_mask,
        )
        assert output.gate is not None
        self.assertTrue(torch.all((output.gate >= 0.0) & (output.gate <= 1.0)))

    def test_empty_history_is_rejected(self) -> None:
        model = PathFusionNextItemModel(
            num_items=8,
            semantic_dim=6,
            hidden_dim=5,
            mode='semantic',
        )
        with self.assertRaises(ValueError):
            model.encode_history(
                torch.tensor([[0, 0]]),
                torch.zeros(1, 2, 6),
                torch.tensor([[False, False]]),
            )


if __name__ == '__main__':
    unittest.main()
