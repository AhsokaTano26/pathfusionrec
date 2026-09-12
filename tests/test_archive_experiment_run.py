"""Tests for the experiment-run delivery archiver."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.archive_experiment_run import (
    HISTORY_FIELDS,
    run_archive,
    write_hashes,
    write_training_history,
)


def _point(epoch: int, ndcg_at_10: float) -> dict[str, float]:
  """A history entry in a deliberately scrambled key order."""
  return {
      'NDCG@10': ndcg_at_10,
      'epoch': epoch,
      'train_loss': 1.7 - epoch / 1000.0,
      'Recall@10': ndcg_at_10 * 2,
      'NDCG@5': ndcg_at_10 * 0.75,
      'Recall@5': ndcg_at_10 * 1.5,
      'NDCG@20': ndcg_at_10 * 1.25,
      'Recall@20': ndcg_at_10 * 3,
      'NDCG@50': ndcg_at_10 * 1.7,
      'Recall@50': ndcg_at_10 * 5.5,
  }


class ArchiveExperimentRunTest(unittest.TestCase):

  def _make_run(self, root: Path, history: list[dict[str, float]]) -> Path:
    output_dir = root / 'run'
    output_dir.mkdir(parents=True)
    (output_dir / 'metrics.json').write_text(
        json.dumps({'best_epoch': 120, 'history': history}),
        encoding='utf-8',
    )
    return output_dir

  def test_training_history_uses_canonical_column_order(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      history = [_point(20, 0.018645), _point(120, 0.020067)]
      output_dir = self._make_run(root, history)

      path = write_training_history(output_dir)

      self.assertEqual(path, output_dir / 'training_history.csv')
      text = path.read_text(encoding='utf-8')
      self.assertNotIn('\r', text)
      with path.open(encoding='utf-8') as source:
        rows = list(csv.DictReader(source))
      self.assertEqual(list(rows[0]), list(HISTORY_FIELDS))
      self.assertEqual([row['epoch'] for row in rows], ['20', '120'])
      self.assertAlmostEqual(float(rows[1]['NDCG@10']), 0.020067)
      self.assertAlmostEqual(float(rows[0]['Recall@50']), 0.018645 * 5.5)

  def test_writes_delivery_files_and_records_hashes(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      output_dir = self._make_run(root, [_point(20, 0.018645)])
      run_log = root / 'tiger_run.log'
      run_log.write_text('launch 2026-09-11T02:17:24Z\nexit=0\n', encoding='utf-8')
      command = (
          'PYTHONPATH=src:. python scripts/train_tiger_retriever.py \\\n'
          '  --epochs 200 --eval-interval 20'
      )
      commit = 'bca28672d5c9019510a008e92cb70ddb4a97bc7a'

      outputs = run_archive(
          output_dir,
          run_log=run_log,
          command=command,
          commit=commit,
          environment='torch: test\n',
      )

      self.assertEqual(
          set(outputs),
          {
              'training_history.csv',
              'run.log',
              'command.txt',
              'git_commit.txt',
              'environment.txt',
              'sha256sums.txt',
          },
      )
      self.assertEqual((output_dir / 'run.log').read_text(encoding='utf-8'),
                       run_log.read_text(encoding='utf-8'))
      self.assertEqual((output_dir / 'command.txt').read_text(encoding='utf-8'),
                       command + '\n')
      self.assertEqual((output_dir / 'git_commit.txt').read_text(encoding='utf-8'),
                       commit + '\n')
      self.assertEqual((output_dir / 'environment.txt').read_text(encoding='utf-8'),
                       'torch: test\n')

      recorded = dict(
          line.split('  ', 1)[::-1]
          for line in (output_dir / 'sha256sums.txt').read_text(encoding='utf-8').splitlines()
      )
      self.assertNotIn('sha256sums.txt', recorded)
      self.assertEqual(
          sorted(recorded),
          ['command.txt', 'environment.txt', 'git_commit.txt', 'metrics.json',
           'run.log', 'training_history.csv'],
      )
      for name, digest in recorded.items():
        actual = hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
        self.assertEqual(digest, actual, name)

  def test_command_trailing_newline_is_not_doubled(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      output_dir = self._make_run(root, [_point(20, 0.018645)])
      run_log = root / 'run.log'
      run_log.write_text('x\n', encoding='utf-8')

      run_archive(
          output_dir,
          run_log=run_log,
          command='python train.py --epochs 200\n',
          commit='deadbeef',
          environment='e\n',
      )

      self.assertEqual((output_dir / 'command.txt').read_text(encoding='utf-8'),
                       'python train.py --epochs 200\n')

  def test_missing_run_log_fails_loudly_without_writing_anything(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      output_dir = self._make_run(root, [_point(20, 0.018645)])

      with self.assertRaises(FileNotFoundError):
        run_archive(
            output_dir,
            run_log=root / 'does-not-exist.log',
            command='python train.py',
            commit='deadbeef',
            environment='e\n',
        )

      self.assertEqual(
          sorted(path.name for path in output_dir.iterdir()), ['metrics.json']
      )

  def test_write_hashes_is_idempotent_and_excludes_itself(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      directory = Path(temp_dir)
      (directory / 'a.txt').write_text('a', encoding='utf-8')
      (directory / 'b.txt').write_text('b', encoding='utf-8')

      first = write_hashes(directory).read_text(encoding='utf-8')
      second = write_hashes(directory).read_text(encoding='utf-8')

      self.assertEqual(first, second)
      self.assertEqual(
          [line.split('  ', 1)[1] for line in first.splitlines()],
          ['a.txt', 'b.txt'],
      )


if __name__ == '__main__':
  unittest.main()
