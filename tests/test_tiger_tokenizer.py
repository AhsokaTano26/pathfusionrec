"""Tests for TIGER Semantic-ID tokenization and prefix constraints."""

from __future__ import annotations

import unittest

from pathfusionrec.tiger.tokenizer import SemanticIdTokenizer


class SemanticIdTokenizerTest(unittest.TestCase):

  def setUp(self) -> None:
    self.tokenizer = SemanticIdTokenizer(
        {1: [0, 3, 7, 0], 2: [1, 4, 8, 0]}, [4, 16, 256], 1, 2_000
    )

  def test_trie_only_allows_catalog_id_continuations(self) -> None:
    prefix = [self.tokenizer.user_token(12), self.tokenizer.level_token(0, 0)]

    self.assertEqual(
        self.tokenizer.allowed_next(prefix), {self.tokenizer.level_token(1, 3)}
    )

  def test_item_tokens_are_decoded_back_to_the_catalog_item(self) -> None:
    semantic_tokens = self.tokenizer.encode_item(2)

    self.assertEqual(self.tokenizer.decode_ids(semantic_tokens), 2)
    self.assertNotEqual(
        self.tokenizer.level_token(0, 0), self.tokenizer.level_token(1, 0)
    )


if __name__ == '__main__':
  unittest.main()
