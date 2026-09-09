"""Integration tests for the TIGER formal training runner."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.train_tiger_retriever import run_training


class TigerTrainingScriptTest(unittest.TestCase):

  def test_runner_reports_all_subgroups(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      protocol_dir = root / 'protocol'
      semantic_id_dir = root / 'semantic_ids'
      protocol_dir.mkdir()
      semantic_id_dir.mkdir()
      split = {
          'train': [
              {'user_id': 1, 'history': [1], 'target': 2},
              {'user_id': 2, 'history': [2], 'target': 3},
          ],
          'validation': [{'user_id': 1, 'history': [1], 'target': 2}],
          'test': [{'user_id': 1, 'history': [1], 'target': 2}],
      }
      (protocol_dir / 'split.json').write_text(json.dumps(split), encoding='utf-8')
      (protocol_dir / 'subset_masks.json').write_text(
          json.dumps({'test': [{'cold_start_target': False, 'long_tail_target': False, 'sparse_user': True}]}),
          encoding='utf-8',
      )
      (semantic_id_dir / 'semantic_ids.json').write_text(
          json.dumps({'1': [0, 0, 0, 0], '2': [1, 0, 0, 0], '3': [1, 1, 0, 0]}),
          encoding='utf-8',
      )
      (semantic_id_dir / 'config.json').write_text(
          json.dumps({'codebook_sizes': [2, 2, 2]}), encoding='utf-8'
      )

      metrics = run_training(
          protocol_dir,
          semantic_id_dir,
          root / 'run',
          epochs=1,
          batch_size=2,
          eval_batch_size=1,
          d_model=32,
          d_ff=64,
          num_layers=1,
          num_heads=2,
          d_kv=16,
          num_beams=4,
          num_workers=0,
          device='cpu',
      )

      self.assertEqual(
          set(metrics['test']),
          {'all', 'cold_start_target', 'long_tail_target', 'sparse_user'},
      )
      self.assertTrue((root / 'run' / 'best_model.pth').exists())
      self.assertEqual(metrics['arguments']['num_workers'], 0)

  def test_runner_respects_eval_interval(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      protocol_dir = root / 'protocol'
      semantic_id_dir = root / 'semantic_ids'
      protocol_dir.mkdir()
      semantic_id_dir.mkdir()
      split = {
          'train': [
              {'user_id': 1, 'history': [1], 'target': 2},
              {'user_id': 2, 'history': [2], 'target': 3},
              {'user_id': 3, 'history': [3], 'target': 1},
          ],
          'validation': [{'user_id': 1, 'history': [1], 'target': 2}],
          'test': [{'user_id': 1, 'history': [1], 'target': 2}],
      }
      (protocol_dir / 'split.json').write_text(json.dumps(split), encoding='utf-8')
      (protocol_dir / 'subset_masks.json').write_text(
          json.dumps({'test': [{'cold_start_target': False, 'long_tail_target': False, 'sparse_user': True}]}),
          encoding='utf-8',
      )
      (semantic_id_dir / 'semantic_ids.json').write_text(
          json.dumps({'1': [0, 0, 0, 0], '2': [1, 0, 0, 0], '3': [1, 1, 0, 0]}),
          encoding='utf-8',
      )
      (semantic_id_dir / 'config.json').write_text(
          json.dumps({'codebook_sizes': [2, 2, 2]}), encoding='utf-8'
      )

      metrics = run_training(
          protocol_dir,
          semantic_id_dir,
          root / 'run',
          epochs=4,
          eval_interval=2,
          batch_size=3,
          eval_batch_size=1,
          d_model=32,
          d_ff=64,
          num_layers=1,
          num_heads=2,
          d_kv=16,
          num_beams=4,
          num_workers=0,
          device='cpu',
      )

      evaluated_epochs = [entry['epoch'] for entry in metrics['history']]
      self.assertEqual(evaluated_epochs, [2, 4])
      self.assertEqual(metrics['arguments']['eval_interval'], 2)
      self.assertIn(metrics['best_epoch'], evaluated_epochs)

  def test_runner_evaluates_final_epoch_when_not_on_interval(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      protocol_dir = root / 'protocol'
      semantic_id_dir = root / 'semantic_ids'
      protocol_dir.mkdir()
      semantic_id_dir.mkdir()
      split = {
          'train': [{'user_id': 1, 'history': [1], 'target': 2}],
          'validation': [{'user_id': 1, 'history': [1], 'target': 2}],
          'test': [{'user_id': 1, 'history': [1], 'target': 2}],
      }
      (protocol_dir / 'split.json').write_text(json.dumps(split), encoding='utf-8')
      (protocol_dir / 'subset_masks.json').write_text(
          json.dumps({'test': [{'cold_start_target': False, 'long_tail_target': False, 'sparse_user': False}]}),
          encoding='utf-8',
      )
      (semantic_id_dir / 'semantic_ids.json').write_text(
          json.dumps({'1': [0, 0, 0, 0], '2': [1, 0, 0, 0]}), encoding='utf-8'
      )
      (semantic_id_dir / 'config.json').write_text(
          json.dumps({'codebook_sizes': [2, 2, 2]}), encoding='utf-8'
      )

      metrics = run_training(
          protocol_dir,
          semantic_id_dir,
          root / 'run',
          epochs=3,
          eval_interval=5,
          batch_size=1,
          eval_batch_size=1,
          d_model=32,
          d_ff=64,
          num_layers=1,
          num_heads=2,
          d_kv=16,
          num_beams=4,
          num_workers=0,
          device='cpu',
      )

      self.assertEqual([entry['epoch'] for entry in metrics['history']], [3])
      self.assertEqual(metrics['best_epoch'], 3)


if __name__ == '__main__':
  unittest.main()
