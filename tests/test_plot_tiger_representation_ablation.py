"""Tests for the Semantic-ID input representation ablation figures and tables."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.plot_tiger_representation_ablation import (
    build_ablation_figures,
    load_ablation_data,
    write_summary_tables,
)


class TigerRepresentationAblationPlotTest(unittest.TestCase):

  def _write_metrics(self, root: Path, representation: str, ndcg_at_10: float) -> None:
    path = (
        root
        / 'experiment_results/05_tiger_semantic_id/TIGER-representation-ablation'
        / f'seed2026/{representation}/metrics.json'
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        'best_epoch': 80,
        'best_validation_ndcg@10': ndcg_at_10 + 0.001,
        'history': [
            {
                'epoch': 10,
                'train_loss': 1.9,
                'NDCG@10': ndcg_at_10 - 0.002,
                'Recall@10': ndcg_at_10 - 0.001,
            },
            {
                'epoch': 20,
                'train_loss': 1.8,
                'NDCG@10': ndcg_at_10,
                'Recall@10': ndcg_at_10 + 0.001,
            },
        ],
        'test': {
            'all': {
                'NDCG@5': ndcg_at_10 - 0.001,
                'Recall@5': ndcg_at_10 + 0.001,
                'NDCG@10': ndcg_at_10,
                'Recall@10': ndcg_at_10 + 0.002,
                'NDCG@20': ndcg_at_10 + 0.003,
                'Recall@20': ndcg_at_10 + 0.004,
                'NDCG@50': ndcg_at_10 + 0.005,
                'Recall@50': ndcg_at_10 + 0.006,
            },
            'cold_start_target': {'NDCG@10': 0.0, 'Recall@10': 0.0},
            'long_tail_target': {'NDCG@10': ndcg_at_10 / 100, 'Recall@10': ndcg_at_10 / 50},
            'sparse_user': {'NDCG@10': ndcg_at_10 + 0.0005, 'Recall@10': ndcg_at_10 + 0.0025},
        },
    }
    path.write_text(json.dumps(metrics), encoding='utf-8')

  def _write_all_metrics(self, root: Path) -> None:
    self._write_metrics(root, 'semantic', 0.014)
    self._write_metrics(root, 'behavior', 0.013)
    self._write_metrics(root, 'fusion', 0.017)

  def test_loads_representations_in_fixed_order(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      self._write_all_metrics(root)

      rows = load_ablation_data(root)

      self.assertEqual([row['representation'] for row in rows], ['semantic', 'behavior', 'fusion'])
      self.assertEqual(rows[2]['name'], 'Balanced fusion (semantic + behavior)')
      self.assertEqual(rows[0]['all']['NDCG@10'], 0.014)
      self.assertEqual(rows[1]['subsets']['sparse_user']['NDCG@10'], 0.0135)
      self.assertEqual([point['epoch'] for point in rows[0]['history']], [10, 20])

  def test_writes_figures_in_png_and_svg(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      output_dir = root / 'summary'
      self._write_all_metrics(root)

      outputs = build_ablation_figures(load_ablation_data(root), output_dir)

      self.assertEqual(
          set(outputs),
          {
              'representation_ablation_metrics.png',
              'representation_ablation_metrics.svg',
              'representation_ablation_subsets.png',
              'representation_ablation_subsets.svg',
              'representation_ablation_training_curves.png',
              'representation_ablation_training_curves.svg',
          },
      )
      for output in outputs.values():
        self.assertTrue(output.exists())
        self.assertGreater(output.stat().st_size, 0)
      with Image.open(outputs['representation_ablation_metrics.png']) as image:
        self.assertEqual(image.getpixel((0, 0))[3], 255)

  def test_writes_results_and_subsets_tables(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      output_dir = root / 'summary'
      self._write_all_metrics(root)

      outputs = write_summary_tables(load_ablation_data(root), output_dir)

      self.assertEqual(set(outputs), {'ablation_results.csv', 'ablation_subsets.csv'})
      with outputs['ablation_results.csv'].open(encoding='utf-8') as source:
        results = list(csv.DictReader(source))
      self.assertEqual(len(results), 3)
      self.assertEqual(results[0]['representation'], 'semantic')
      self.assertEqual(results[0]['best_epoch'], '80')
      self.assertAlmostEqual(float(results[2]['NDCG@10']), 0.017)
      with outputs['ablation_subsets.csv'].open(encoding='utf-8') as source:
        subsets = list(csv.DictReader(source))
      self.assertEqual(len(subsets), 12)
      self.assertEqual(
          [(row['representation'], row['subset']) for row in subsets[:2]],
          [('semantic', 'all'), ('semantic', 'sparse_user')],
      )
      self.assertAlmostEqual(float(subsets[0]['NDCG@10']), 0.014)


if __name__ == '__main__':
  unittest.main()
