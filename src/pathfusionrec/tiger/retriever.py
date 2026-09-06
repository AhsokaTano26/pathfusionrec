"""TIGER T5 retriever with valid Semantic-ID constrained generation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as functional
from transformers import T5Config, T5ForConditionalGeneration

from pathfusionrec.tiger.tokenizer import SemanticIdTokenizer


def default_tiger_config(tokenizer: SemanticIdTokenizer) -> T5Config:
  """Return the randomly initialized T5 configuration used for formal runs."""
  return T5Config(
      vocab_size=tokenizer.vocab_size,
      d_model=128,
      d_ff=1024,
      num_layers=4,
      num_decoder_layers=4,
      num_heads=6,
      d_kv=64,
      dropout_rate=0.1,
      pad_token_id=tokenizer.pad_token_id,
      decoder_start_token_id=tokenizer.bos_token_id,
      eos_token_id=tokenizer.eos_token_id,
  )


class TigerRetriever(nn.Module):
  """Generate a next-item Semantic ID with a randomly initialized T5."""

  def __init__(self, tokenizer: SemanticIdTokenizer, config: T5Config | None = None) -> None:
    super().__init__()
    self.tokenizer = tokenizer
    self.config = config or default_tiger_config(tokenizer)
    self.model = T5ForConditionalGeneration(self.config)

  def make_batch(self, samples: Sequence[Mapping[str, Any]]) -> dict[str, Tensor]:
    """Create padded T5 inputs and teacher-forced labels from protocol samples."""
    if not samples:
      raise ValueError('At least one next-item sample is required.')
    inputs: list[list[int]] = []
    labels: list[list[int]] = []
    for sample in samples:
      user_id = int(sample.get('user_id', sample.get('user')))
      history = [int(item_id) for item_id in sample['history']]
      target = int(sample['target'])
      inputs.append(self.tokenizer.encode_history(user_id, history))
      labels.append([*self.tokenizer.encode_item(target), self.tokenizer.eos_token_id])
    max_input_length = max(len(row) for row in inputs)
    max_label_length = max(len(row) for row in labels)
    input_ids = torch.full(
        (len(samples), max_input_length), self.tokenizer.pad_token_id, dtype=torch.long
    )
    label_ids = torch.full((len(samples), max_label_length), -100, dtype=torch.long)
    for row, (input_tokens, label_tokens) in enumerate(zip(inputs, labels, strict=True)):
      input_ids[row, : len(input_tokens)] = torch.tensor(input_tokens)
      label_ids[row, : len(label_tokens)] = torch.tensor(label_tokens)
    return {
        'input_ids': input_ids,
        'attention_mask': input_ids.ne(self.tokenizer.pad_token_id),
        'labels': label_ids,
    }

  def loss(self, batch: Mapping[str, Tensor]) -> Tensor:
    """Return teacher-forcing cross-entropy loss for a batch."""
    return self.model(**batch).loss

  @torch.no_grad()
  def generate_top_k(
      self, history_tokens: Sequence[int], k: int, beam_size: int
  ) -> list[int]:
    """Generate up to k valid catalog items with prefix-constrained beam search."""
    if k <= 0 or beam_size <= 0:
      raise ValueError('k and beam_size must be positive.')
    device = next(self.parameters()).device
    input_ids = torch.tensor([list(history_tokens)], dtype=torch.long, device=device)
    attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
    beams: list[tuple[list[int], float]] = [([self.tokenizer.bos_token_id], 0.0)]
    completed: list[tuple[int, float]] = []
    self.eval()
    # Encode the history once and reuse the hidden states for every beam step.
    encoder_outputs = self.model.encoder(
        input_ids=input_ids, attention_mask=attention_mask
    )
    for _ in range(5):
      if not beams:
        break
      max_length = max(len(tokens) for tokens, _ in beams)
      decoder_input_ids = torch.full(
          (len(beams), max_length),
          self.tokenizer.pad_token_id,
          dtype=torch.long,
          device=device,
      )
      prefix_lengths: list[int] = []
      for row, (tokens, _) in enumerate(beams):
        decoder_input_ids[row, : len(tokens)] = torch.tensor(
            tokens, dtype=torch.long, device=device
        )
        prefix_lengths.append(len(tokens))
      # One batched decoder forward across all beams for this step.
      logits = self.model(
          input_ids=input_ids,
          attention_mask=attention_mask,
          encoder_outputs=encoder_outputs,
          decoder_input_ids=decoder_input_ids,
      ).logits
      candidates: list[tuple[list[int], float]] = []
      for row, (tokens, score) in enumerate(beams):
        allowed = self.tokenizer.allowed_next(tokens)
        if not allowed:
          continue
        next_logits = logits[row, prefix_lengths[row] - 1]
        log_probabilities = functional.log_softmax(next_logits, dim=-1)
        for token in allowed:
          candidates.append((tokens + [token], score + float(log_probabilities[token])))
      candidates.sort(key=lambda item: item[1], reverse=True)
      beams = []
      for tokens, score in candidates[:beam_size]:
        if tokens[-1] == self.tokenizer.eos_token_id:
          try:
            item_id = self.tokenizer.decode_ids(tokens[1:-1])
          except ValueError:
            continue
          completed.append((item_id, score))
        else:
          beams.append((tokens, score))
    completed.sort(key=lambda item: item[1], reverse=True)
    results: list[int] = []
    for item_id, _ in completed:
      if item_id not in results:
        results.append(item_id)
      if len(results) == k:
        break
    return results
