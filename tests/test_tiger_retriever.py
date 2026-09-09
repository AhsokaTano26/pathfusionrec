"""Tests for TIGER T5 batch construction and constrained decoding."""

from __future__ import annotations

import unittest

import torch
from torch.nn import functional as functional
from transformers import T5Config

from pathfusionrec.tiger.retriever import TigerRetriever
from pathfusionrec.tiger.tokenizer import SemanticIdTokenizer


def _sequential_reference_generate(
    retriever: TigerRetriever, history_tokens: list[int], k: int, beam_size: int
) -> list[int]:
  """Per-beam sequential decode used to cross-check the batched decoder."""
  device = next(retriever.parameters()).device
  input_ids = torch.tensor([list(history_tokens)], dtype=torch.long, device=device)
  attention_mask = input_ids.ne(retriever.tokenizer.pad_token_id)
  beams: list[tuple[list[int], float]] = [
      ([retriever.tokenizer.bos_token_id], 0.0)
  ]
  completed: list[tuple[int, float]] = []
  retriever.eval()
  for _ in range(5):
    candidates: list[tuple[list[int], float]] = []
    for tokens, score in beams:
      allowed = retriever.tokenizer.allowed_next(tokens)
      if not allowed:
        continue
      decoder_input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
      logits = retriever.model(
          input_ids=input_ids,
          attention_mask=attention_mask,
          decoder_input_ids=decoder_input_ids,
      ).logits[0, -1]
      log_probabilities = functional.log_softmax(logits, dim=-1)
      for token in allowed:
        candidates.append((tokens + [token], score + float(log_probabilities[token])))
    candidates.sort(key=lambda item: item[1], reverse=True)
    beams = []
    for tokens, score in candidates[:beam_size]:
      if tokens[-1] == retriever.tokenizer.eos_token_id:
        try:
          item_id = retriever.tokenizer.decode_ids(tokens[1:-1])
        except ValueError:
          continue
        completed.append((item_id, score))
      else:
        beams.append((tokens, score))
    if not beams:
      break
  completed.sort(key=lambda item: item[1], reverse=True)
  results: list[int] = []
  for item_id, _ in completed:
    if item_id not in results:
      results.append(item_id)
    if len(results) == k:
      break
  return results


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

  def test_batch_generation_matches_per_history(self) -> None:
    histories = [
        self.tokenizer.encode_history(1, [1]),
        self.tokenizer.encode_history(2, [2, 3]),
        self.tokenizer.encode_history(3, [3, 1, 2]),
    ]

    generated = self.retriever.generate_top_k_batch(histories, k=2, beam_size=4)

    self.assertEqual(
        generated,
        [
            self.retriever.generate_top_k(history, k=2, beam_size=4)
            for history in histories
        ],
    )


class TigerRetrieverBatchEquivalenceTest(unittest.TestCase):
  """The batched beam decoder must match the sequential decoder exactly."""

  def setUp(self) -> None:
    catalog = {
        item_id: [item_id % 4, (item_id * 7) % 16, item_id, 0]
        for item_id in range(1, 61)
    }
    self.tokenizer = SemanticIdTokenizer(catalog, [4, 16, 256], 1, 16)
    config = T5Config(
        vocab_size=self.tokenizer.vocab_size,
        d_model=48,
        d_ff=128,
        num_layers=2,
        num_decoder_layers=2,
        num_heads=3,
        d_kv=16,
        pad_token_id=self.tokenizer.pad_token_id,
        decoder_start_token_id=self.tokenizer.bos_token_id,
        eos_token_id=self.tokenizer.eos_token_id,
    )
    self.retriever = TigerRetriever(self.tokenizer, config)
    self.retriever.eval()

  def _histories(self) -> list[list[int]]:
    torch.manual_seed(0)
    return [
        self.tokenizer.encode_history(3, [1]),
        self.tokenizer.encode_history(7, [2, 9, 40]),
        self.tokenizer.encode_history(11, [60, 1, 2, 3]),
        self.tokenizer.encode_history(5, [i for i in range(1, 21)]),
        self.tokenizer.encode_history(2, [30, 45, 12, 8, 59]),
    ]

  def test_batched_matches_sequential_exactly(self) -> None:
    for history in self._histories():
      for k, beam_size in [(3, 2), (5, 4), (10, 8), (10, 16), (20, 10)]:
        batched = self.retriever.generate_top_k(history, k=k, beam_size=beam_size)
        sequential = _sequential_reference_generate(
            self.retriever, history, k=k, beam_size=beam_size
        )
        self.assertEqual(
            batched,
            sequential,
            f'mismatch history={history[:4]}... k={k} beam_size={beam_size}',
        )

  def test_batched_results_are_valid_catalog_items(self) -> None:
    for history in self._histories():
      results = self.retriever.generate_top_k(history, k=5, beam_size=8)
      self.assertTrue(set(results).issubset(set(range(1, 61))))
      self.assertEqual(len(results), len(set(results)))


if __name__ == '__main__':
  unittest.main()
