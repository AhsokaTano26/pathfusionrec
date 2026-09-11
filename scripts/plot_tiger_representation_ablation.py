"""Summarise the TIGER Semantic-ID input-representation ablation.

This experiment isolates the RQ-VAE *input representation* while keeping the
TIGER decoder, tokenizer, candidate constraint and beam width fixed.  It is a
Semantic-ID input ablation, not a PathFusionRec method ablation and not an
external-baseline comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt


BASE_DIR = Path(
    'experiment_results/05_tiger_semantic_id/TIGER-representation-ablation/seed2026'
)
REPRESENTATIONS = (
    ('semantic', 'Semantic input (Sentence-T5 768-d)'),
    ('behavior', 'Behavior input (PFR interaction 128-d)'),
    ('fusion', 'Balanced fusion (semantic + behavior)'),
)
SUBSETS = ('all', 'sparse_user', 'cold_start_target', 'long_tail_target')
SUBSET_LABELS = ('All', 'Sparse users', 'Cold-start target', 'Long-tail target')
ALL_METRICS = ('NDCG@5', 'NDCG@10', 'NDCG@20', 'NDCG@50')
COLORS = ('#0072B2', '#D55E00', '#009E73')


def load_ablation_data(root: Path) -> list[dict[str, Any]]:
  """Load the three representations' test metrics in the fixed ablation order."""
  rows: list[dict[str, Any]] = []
  for representation, name in REPRESENTATIONS:
    metrics_path = root / BASE_DIR / representation / 'metrics.json'
    with metrics_path.open(encoding='utf-8') as source:
      metrics = json.load(source)
    test_metrics = metrics['test']
    rows.append(
        {
            'representation': representation,
            'name': name,
            'best_epoch': metrics['best_epoch'],
            'best_validation_ndcg@10': metrics['best_validation_ndcg@10'],
            'all': test_metrics['all'],
            'subsets': {subset: test_metrics[subset] for subset in SUBSETS},
            'history': metrics['history'],
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
  for axis, title, ylabel in zip(
      axes, ('All items: NDCG', 'All items: Recall'), ('NDCG', 'Recall')
  ):
    axis.set_title(title)
    axis.set_xlabel('Rank K')
    axis.set_ylabel(ylabel)
    axis.set_xticks(ranks)
  axes[1].legend(loc='upper left', frameon=False)
  figure.suptitle('TIGER Semantic-ID input representation ablation (test set)', y=1.02)
  figure.tight_layout()
  return _save_pair(figure, output_dir, 'representation_ablation_metrics')


def _plot_subsets(rows: list[dict[str, Any]], output_dir: Path) -> dict[str, Path]:
  figure, axes = plt.subplots(1, 2, figsize=(11.0, 3.8), sharex=True)
  positions = np.arange(len(SUBSETS))
  width = 0.26
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
  for axis, title, ylabel in zip(
      axes,
      ('Test subsets: NDCG@10', 'Test subsets: Recall@10'),
      ('NDCG@10', 'Recall@10'),
  ):
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.set_xticks(positions, SUBSET_LABELS, rotation=16, ha='right')
  axes[1].legend(loc='upper right', frameon=False)
  figure.suptitle('TIGER Semantic-ID input representation ablation (test subgroups)', y=1.02)
  figure.tight_layout()
  return _save_pair(figure, output_dir, 'representation_ablation_subsets')


def _plot_training_curves(rows: list[dict[str, Any]], output_dir: Path) -> dict[str, Path]:
  figure, axes = plt.subplots(1, 2, figsize=(10.6, 3.8))
  for row, color in zip(rows, COLORS):
    epochs = [point['epoch'] for point in row['history']]
    axes[0].plot(
        epochs,
        [point['NDCG@10'] for point in row['history']],
        marker='o',
        linewidth=2,
        markersize=4,
        color=color,
        label=row['name'],
    )
    axes[1].plot(
        epochs,
        [point['train_loss'] for point in row['history']],
        marker='o',
        linewidth=2,
        markersize=4,
        color=color,
        label=row['name'],
    )
  for axis, title, ylabel in zip(
      axes,
      ('Validation NDCG@10', 'Training loss'),
      ('Validation NDCG@10', 'Train loss'),
  ):
    axis.set_title(title)
    axis.set_xlabel('Epoch')
    axis.set_ylabel(ylabel)
  axes[0].legend(loc='lower right', frameon=False)
  figure.suptitle('TIGER Semantic-ID input representation ablation (training)', y=1.02)
  figure.tight_layout()
  return _save_pair(figure, output_dir, 'representation_ablation_training_curves')


def build_ablation_figures(rows: list[dict[str, Any]], output_dir: Path) -> dict[str, Path]:
  """Write the ablation figures in both PNG and SVG formats."""
  _configure_style()
  output_dir.mkdir(parents=True, exist_ok=True)
  outputs = _plot_all_metrics(rows, output_dir)
  outputs.update(_plot_subsets(rows, output_dir))
  outputs.update(_plot_training_curves(rows, output_dir))
  return outputs


def write_summary_tables(rows: list[dict[str, Any]], output_dir: Path) -> dict[str, Path]:
  """Write the machine-readable ablation result and subset tables."""
  output_dir.mkdir(parents=True, exist_ok=True)
  metric_columns = [
      column
      for rank in (5, 10, 20, 50)
      for column in (f'NDCG@{rank}', f'Recall@{rank}')
  ]
  results_path = output_dir / 'ablation_results.csv'
  with results_path.open('w', encoding='utf-8', newline='') as sink:
    writer = csv.DictWriter(
        sink,
        fieldnames=['representation', 'best_epoch', 'best_validation_ndcg@10', *metric_columns],
        lineterminator='\n',
    )
    writer.writeheader()
    for row in rows:
      writer.writerow(
          {
              'representation': row['representation'],
              'best_epoch': row['best_epoch'],
              'best_validation_ndcg@10': f"{row['best_validation_ndcg@10']:.6f}",
              **{column: f"{row['all'][column]:.6f}" for column in metric_columns},
          }
      )
  subsets_path = output_dir / 'ablation_subsets.csv'
  with subsets_path.open('w', encoding='utf-8', newline='') as sink:
    writer = csv.DictWriter(
        sink,
        fieldnames=['representation', 'subset', *metric_columns],
        lineterminator='\n',
    )
    writer.writeheader()
    for row in rows:
      for subset in SUBSETS:
        writer.writerow(
            {
                'representation': row['representation'],
                'subset': subset,
                **{
                    column: (
                        f"{value:.6f}"
                        if (value := row['subsets'][subset].get(column)) is not None
                        else ''
                    )
                    for column in metric_columns
                },
            }
        )
  return {'ablation_results.csv': results_path, 'ablation_subsets.csv': subsets_path}


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
  parser.add_argument(
      '--output-dir',
      type=Path,
      default=Path(__file__).resolve().parents[1] / BASE_DIR / 'summary',
  )
  arguments = parser.parse_args()
  rows = load_ablation_data(arguments.root)
  for path in (
      *build_ablation_figures(rows, arguments.output_dir).values(),
      *write_summary_tables(rows, arguments.output_dir).values(),
  ):
    print(path)


if __name__ == '__main__':
  main()
