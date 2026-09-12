"""Assemble the per-run delivery files required by docs/03 for one experiment run.

`train_tiger_retriever.py` writes `metrics.json`, `config.json`,
`delivery_validation.txt` and `best_model.pth`. The delivery spec additionally
asks for `training_history.csv`, `run.log`, `command.txt`, `environment.txt`
and a hash record, which are assembled here so that a finished run leaves a
complete, checkable directory instead of a scatter of shell one-liners.

Run it once per run directory:

  PYTHONPATH=src:. python scripts/archive_experiment_run.py \
    --output-dir experiment_results/.../seed2026/behavior-200ep \
    --run-log /root/autodl-tmp/logs/tiger_behavior200.log \
    --command-file /root/autodl-tmp/logs/tiger_behavior200_command.txt

Upstream RQ-VAE directories are plain static files (`config.json`,
`codebook_diagnostics.json`, `standardization.json`, `training_history.json`,
`delivery_validation.txt`); copy them with `cp`/`rsync` and then call
`write_hashes` on the destination.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

HISTORY_FIELDS = (
    'epoch', 'train_loss', 'NDCG@5', 'Recall@5', 'NDCG@10', 'Recall@10',
    'NDCG@20', 'Recall@20', 'NDCG@50', 'Recall@50',
)


def write_training_history(output_dir: Path) -> Path:
  """Expand metrics.json's history into the flat per-epoch CSV."""
  metrics = json.loads((output_dir / 'metrics.json').read_text(encoding='utf-8'))
  history = metrics['history']
  missing = [
      field for field in HISTORY_FIELDS
      if any(field not in point for point in history)
  ]
  if missing:
    raise ValueError(f'metrics.json history is missing fields: {missing}')
  path = output_dir / 'training_history.csv'
  with path.open('w', encoding='utf-8', newline='') as sink:
    writer = csv.DictWriter(sink, fieldnames=HISTORY_FIELDS, lineterminator='\n')
    writer.writeheader()
    for point in history:
      writer.writerow({field: point[field] for field in HISTORY_FIELDS})
  return path


def write_hashes(directory: Path) -> Path:
  """Record SHA-256 for every regular file in the directory except the record."""
  target = directory / 'sha256sums.txt'
  rows = [
      f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}'
      for path in sorted(directory.iterdir())
      if path.is_file() and path.name != target.name
  ]
  target.write_text('\n'.join(rows) + '\n', encoding='utf-8')
  return target


def collect_environment() -> str:
  """Describe the interpreter, libraries and GPU that produced the run."""
  import matplotlib
  import numpy
  import torch
  import transformers

  lines = [
      f'python: {platform.python_version()} ({sys.executable})',
      f'platform: {platform.platform()}',
      f'torch: {torch.__version__}',
      f'torch.version.cuda: {torch.version.cuda}',
      f'cudnn: {torch.backends.cudnn.version()}',
      f'transformers: {transformers.__version__}',
      f'numpy: {numpy.__version__}',
      f'matplotlib: {matplotlib.__version__}',
      f'gpu: {torch.cuda.get_device_name(0)}',
      f'gpu_capability: {torch.cuda.get_device_capability(0)}',
  ]
  try:
    summary = subprocess.run(
        ['nvidia-smi', '--query-gpu=driver_version,memory.total', '--format=csv,noheader'],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    lines.append(f'nvidia_smi: {summary}')
  except (OSError, subprocess.CalledProcessError) as error:
    lines.append(f'nvidia_smi: unavailable ({error})')
  return '\n'.join(lines) + '\n'


def resolve_commit(repo_root: Path) -> str:
  return subprocess.run(
      ['git', 'rev-parse', 'HEAD'], cwd=repo_root, capture_output=True,
      text=True, check=True,
  ).stdout.strip()


def run_archive(
    output_dir: Path,
    *,
    run_log: Path,
    command: str,
    commit: str,
    environment: str | None = None,
    repo_root: Path | None = None,
) -> dict[str, Path]:
  """Write the delivery files for one finished run directory."""
  if not (output_dir / 'metrics.json').is_file():
    raise FileNotFoundError(f'No metrics.json in {output_dir}')
  log_text = run_log.read_text(encoding='utf-8')
  if environment is None:
    environment = collect_environment()
  if not commit:
    if repo_root is None:
      raise ValueError('Either commit or repo_root must be supplied.')
    commit = resolve_commit(repo_root)

  write_training_history(output_dir)
  (output_dir / 'run.log').write_text(log_text, encoding='utf-8')
  (output_dir / 'command.txt').write_text(command.rstrip('\n') + '\n', encoding='utf-8')
  (output_dir / 'git_commit.txt').write_text(commit + '\n', encoding='utf-8')
  (output_dir / 'environment.txt').write_text(environment, encoding='utf-8')
  write_hashes(output_dir)
  return {
      name: output_dir / name
      for name in ('training_history.csv', 'run.log', 'command.txt',
                   'git_commit.txt', 'environment.txt', 'sha256sums.txt')
  }


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output-dir', type=Path, required=True)
  parser.add_argument('--run-log', type=Path, required=True)
  source = parser.add_mutually_exclusive_group(required=True)
  source.add_argument('--command-file', type=Path)
  source.add_argument('--command')
  parser.add_argument('--commit', help='Defaults to git rev-parse HEAD.')
  parser.add_argument(
      '--repo-root', type=Path, default=Path(__file__).resolve().parents[1]
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  command = (
      args.command if args.command is not None
      else args.command_file.read_text(encoding='utf-8')
  )
  for path in run_archive(
      args.output_dir,
      run_log=args.run_log,
      command=command,
      commit=args.commit or '',
      repo_root=args.repo_root,
  ).values():
    print(path)


if __name__ == '__main__':
  main()
