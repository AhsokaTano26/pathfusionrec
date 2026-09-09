"""Create reproducible TIGER-versus-PFR test-set comparison figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt


COMPARISON_SOURCES = (
    (
        'TIGER (external baseline)',
        Path('experiment_results/05_tiger_semantic_id/TIGER-v1/seed2026/metrics.json'),
    ),
    (
        'PathFusionRec (proposed)',
        Path('experiment_results/04_unified_next_item/PFR-Interaction-v1/seed2026/metrics.json'),
    ),
    (
        'PFR semantic-only ablation',
        Path('experiment_results/04_unified_next_item/PFR-Semantic-v1/seed2026/metrics.json'),
    ),
    (
        'PFR concat-fusion ablation',
        Path('experiment_results/04_unified_next_item/PFR-Fusion-Concat-v1/seed2026/metrics.json'),
    ),
)
ALL_METRICS = ('NDCG@5', 'NDCG@10', 'NDCG@20', 'NDCG@50')
SUBSETS = ('all', 'sparse_user', 'cold_start_target', 'long_tail_target')
COLORS = ('#0072B2', '#D55E00', '#009E73', '#CC79A7')


def load_comparison_data(root: Path) -> list[dict[str, Any]]:
  """Load the fixed, protocol-compatible TIGER and PFR test metrics."""
  rows: list[dict[str, Any]] = []
  for name, relative_path in COMPARISON_SOURCES:
    with (root / relative_path).open(encoding='utf-8') as source:
      test_metrics = json.load(source)['test']
    rows.append(
        {
            'name': name,
            'all': test_metrics['all'],
            'subsets': {subset: test_metrics[subset] for subset in SUBSETS},
        }
    )
  return rows


def _configure_style() -> None:
  plt.style.use('default')
  plt.rcParams.update(
      {
          'axes.spines.top': False,
          'axes.spines.right': False,
          'axes.grid': True,
          'axes.axisbelow': True,
          'grid.alpha': 0.25,
          'grid.linewidth': 0.7,
          'font.family': 'DejaVu Sans',
          'font.size': 10,
          'axes.labelsize': 10,
          'axes.titlesize': 12,
          'legend.fontsize': 8,
          'svg.fonttype': 'none',
      }
  )


def _save_pair(figure: plt.Figure, output_dir: Path, stem: str) -> dict[str, Path]:
  outputs = {
      f'{stem}.png': output_dir / f'{stem}.png',
      f'{stem}.svg': output_dir / f'{stem}.svg',
  }
  figure.savefig(
      outputs[f'{stem}.png'], dpi=300, bbox_inches='tight', facecolor='white', transparent=False
  )
  figure.savefig(outputs[f'{stem}.svg'], bbox_inches='tight', facecolor='white', transparent=False)
  plt.close(figure)
  return outputs


def _plot_all_metrics(rows: list[dict[str, Any]], output_dir: Path) -> dict[str, Path]:
  figure, axes = plt.subplots(1, 2, figsize=(10.6, 3.8), sharex=True)
  ranks = (5, 10, 20, 50)
  for row, color in zip(rows, COLORS):
    axes[0].plot(
        ranks,
        [row['all'][metric] for metric in ALL_METRICS],
        marker='o',
        linewidth=2,
        markersize=5,
        color=color,
        label=row['name'],
    )
    axes[1].plot(
        ranks,
        [row['all'][metric.replace('NDCG', 'Recall')] for metric in ALL_METRICS],
        marker='o',
        linewidth=2,
        markersize=5,
        color=color,
        label=row['name'],
    )
  for axis, title, ylabel in zip(axes, ('All users: NDCG', 'All users: Recall'), ('NDCG', 'Recall')):
    axis.set_title(title)
    axis.set_xlabel('Rank K')
    axis.set_ylabel(ylabel)
    axis.set_xticks(ranks)
  axes[1].legend(loc='upper left', frameon=False)
  figure.suptitle('Proposed PathFusionRec vs. TIGER baseline and PFR ablations', y=1.02)
  figure.tight_layout()
  return _save_pair(figure, output_dir, 'comparison_all_metrics')


def _plot_subgroups(rows: list[dict[str, Any]], output_dir: Path) -> dict[str, Path]:
  figure, axes = plt.subplots(1, 2, figsize=(11.0, 3.8), sharex=True)
  positions = np.arange(len(SUBSETS))
  width = 0.19
  labels = ('All', 'Sparse users', 'Cold-start target', 'Long-tail target')
  for index, (row, color) in enumerate(zip(rows, COLORS)):
    offset = (index - (len(rows) - 1) / 2) * width
    axes[0].bar(
        positions + offset,
        [row['subsets'][subset]['NDCG@10'] for subset in SUBSETS],
        width,
        color=color,
        label=row['name'],
    )
    axes[1].bar(
        positions + offset,
        [row['subsets'][subset]['Recall@10'] for subset in SUBSETS],
        width,
        color=color,
        label=row['name'],
    )
  for axis, title, ylabel in zip(axes, ('Test subgroups: NDCG@10', 'Test subgroups: Recall@10'), ('NDCG@10', 'Recall@10')):
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.set_xticks(positions, labels, rotation=16, ha='right')
  axes[1].legend(loc='upper right', frameon=False)
  figure.suptitle('Proposed PathFusionRec vs. TIGER baseline and PFR ablations', y=1.02)
  figure.tight_layout()
  return _save_pair(figure, output_dir, 'comparison_subgroups')


def build_comparison_figures(root: Path, output_dir: Path) -> dict[str, Path]:
  """Write all direct-comparison figures in both PNG and SVG formats."""
  _configure_style()
  output_dir.mkdir(parents=True, exist_ok=True)
  rows = load_comparison_data(root)
  outputs = _plot_all_metrics(rows, output_dir)
  outputs.update(_plot_subgroups(rows, output_dir))
  return outputs


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
  parser.add_argument(
      '--output-dir',
      type=Path,
      default=Path(__file__).resolve().parents[1]
      / 'experiment_results/05_tiger_semantic_id/TIGER-v1/seed2026',
  )
  arguments = parser.parse_args()
  for path in build_comparison_figures(arguments.root, arguments.output_dir).values():
    print(path)


if __name__ == '__main__':
  main()
