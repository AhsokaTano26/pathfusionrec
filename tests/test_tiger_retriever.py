"""Tests for TIGER T5 batch construction and constrained decoding."""

from __future__ import annotations

import unittest

from transformers import T5Config

from pathfusionrec.tiger.retriever import TigerRetriever
from pathfusionrec.tiger.tokenizer import SemanticIdTokenizer


class TigerRetrieverTest(unittest.TestCase):

  def setUp(self) -> None:
    self.tokenizer = SemanticIdTokenizer(
        {1: [0, 3, 7, 0], 2: [1, 4, 8, 0], 3: [1, 4, 9, 0]},
        [4, 16, 256],
        1,
        8,
    )
    config = T5Config(
        vocab_size=self.tokenizer.vocab_size,
        d_model=32,
        d_ff=64,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        d_kv=16,
        pad_token_id=self.tokenizer.pad_token_id,
        decoder_start_token_id=self.tokenizer.bos_token_id,
        eos_token_id=self.tokenizer.eos_token_id,
    )
    self.retriever = TigerRetriever(self.tokenizer, config)

  def test_teacher_forcing_target_is_four_id_tokens_and_eos(self) -> None:
    batch = self.retriever.make_batch(
        [{'user_id': 7, 'history': [1], 'target': 2}]
    )

    self.assertEqual(tuple(batch['labels'].shape), (1, 5))
    self.assertEqual(int(batch['labels'][0, -1]), self.tokenizer.eos_token_id)

  def test_generation_only_returns_catalog_items(self) -> None:
    items = self.retriever.generate_top_k(
        self.tokenizer.encode_history(1, [1]), k=2, beam_size=4
    )

    self.assertTrue(set(items).issubset({1, 2, 3}))


if __name__ == '__main__':
  unittest.main()
