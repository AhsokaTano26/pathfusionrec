"""Tests for TIGER 768-dimensional content artifacts."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from pathfusionrec.tiger.content import (
    save_content_artifact,
    standardize_train_items,
)


class TigerContentTest(unittest.TestCase):

  def test_content_artifact_rejects_non_768_vectors(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      with self.assertRaisesRegex(ValueError, '768'):
        save_content_artifact(
            Path(temp_dir), np.zeros((2, 127), dtype=np.float32), [1, 2], {}
        )

  def test_standardization_uses_only_train_rows(self) -> None:
    vectors = np.array([[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]], dtype=np.float32)
    standardized, statistics = standardize_train_items(vectors, np.array([0, 1]))

    np.testing.assert_allclose(statistics['mean'], [2.0, 2.0])
    np.testing.assert_allclose(statistics['std'], [1.0, 1.0])
    self.assertGreater(float(standardized[2, 0]), 90.0)


if __name__ == '__main__':
  unittest.main()
