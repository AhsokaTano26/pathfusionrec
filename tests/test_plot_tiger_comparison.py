"""Tests for the reproducible TIGER-versus-PFR comparison plots."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.plot_tiger_comparison import build_comparison_figures, load_comparison_data


class TigerComparisonPlotTest(unittest.TestCase):

  def _write_metrics(
      self,
      root: Path,
      relative_path: str,
      ndcg_at_10: float,
  ) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
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
    self._write_metrics(
        root,
        'experiment_results/05_tiger_semantic_id/TIGER-v1/seed2026/metrics.json',
        0.014,
    )
    self._write_metrics(
        root,
        'experiment_results/04_unified_next_item/PFR-Interaction-v1/seed2026/metrics.json',
        0.016,
    )
    self._write_metrics(
        root,
        'experiment_results/04_unified_next_item/PFR-Semantic-v1/seed2026/metrics.json',
        0.004,
    )
    self._write_metrics(
        root,
        'experiment_results/04_unified_next_item/PFR-Fusion-Concat-v1/seed2026/metrics.json',
        0.003,
    )

  def test_loads_models_in_paper_comparison_order(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      self._write_all_metrics(root)

      rows = load_comparison_data(root)

      self.assertEqual(
          [row['name'] for row in rows],
          [
              'TIGER (external baseline)',
              'PathFusionRec (proposed)',
              'PFR semantic-only ablation',
              'PFR concat-fusion ablation',
          ],
      )
      self.assertEqual(rows[0]['all']['NDCG@10'], 0.014)
      self.assertEqual(rows[1]['subsets']['sparse_user']['NDCG@10'], 0.0165)

  def test_writes_all_metric_and_subgroup_figures_in_png_and_svg(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      output_dir = root / 'figures'
      self._write_all_metrics(root)

      outputs = build_comparison_figures(root, output_dir)

      self.assertEqual(
          set(outputs),
          {
              'comparison_all_metrics.png',
              'comparison_all_metrics.svg',
              'comparison_subgroups.png',
              'comparison_subgroups.svg',
          },
      )
      for output in outputs.values():
        self.assertTrue(output.exists())
        self.assertGreater(output.stat().st_size, 0)
      with Image.open(outputs['comparison_all_metrics.png']) as image:
        self.assertEqual(image.getpixel((0, 0))[3], 255)


if __name__ == '__main__':
  unittest.main()
