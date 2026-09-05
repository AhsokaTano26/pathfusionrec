"""Semantic-ID vocabulary and valid-prefix constraints for TIGER."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence


class SemanticIdTokenizer:
  """Map unique RQ-VAE IDs to disjoint vocabulary tokens and a prefix trie."""

  def __init__(
      self,
      item_ids: Mapping[int, Sequence[int]],
      codebook_sizes: Sequence[int],
      collision_size: int,
      user_bucket_count: int = 2_000,
  ) -> None:
    if len(codebook_sizes) != 3 or any(size <= 1 for size in codebook_sizes):
      raise ValueError('TIGER requires three residual codebook sizes greater than one.')
    if collision_size <= 0 or user_bucket_count <= 0:
      raise ValueError('Collision and user token counts must be positive.')
    self.codebook_sizes = tuple(int(size) for size in codebook_sizes)
    self.collision_size = collision_size
    self.user_bucket_count = user_bucket_count
    self.pad_token_id = 0
    self.bos_token_id = 1
    self.eos_token_id = 2
    self._level_offsets = (3, 3 + self.codebook_sizes[0], 3 + sum(self.codebook_sizes[:2]))
    self._collision_offset = 3 + sum(self.codebook_sizes)
    self._user_offset = self._collision_offset + collision_size
    self.vocab_size = self._user_offset + user_bucket_count
    self._item_ids = {int(item_id): tuple(int(value) for value in values) for item_id, values in item_ids.items()}
    self._validate_item_ids()
    self._reverse_item_ids = {
        self._encode_semantic_tuple(semantic_id): item_id
        for item_id, semantic_id in self._item_ids.items()
    }
    self._trie: dict[int, dict] = {}
    for semantic_tokens in self._reverse_item_ids:
      current = self._trie
      for token in semantic_tokens:
        current = current.setdefault(token, {})

  def _validate_item_ids(self) -> None:
    if not self._item_ids:
      raise ValueError('At least one Semantic ID is required.')
    raw_ids = set()
    for item_id, semantic_id in self._item_ids.items():
      if item_id <= 0 or len(semantic_id) != 4:
        raise ValueError('Every catalog item requires a positive ID and four tokens.')
      if semantic_id in raw_ids:
        raise ValueError('Semantic IDs must be unique before vocabulary encoding.')
      raw_ids.add(semantic_id)
      for level, code in enumerate(semantic_id[:3]):
        if code < 0 or code >= self.codebook_sizes[level]:
          raise ValueError('Residual code is outside its codebook range.')
      if semantic_id[3] < 0 or semantic_id[3] >= self.collision_size:
        raise ValueError('Collision code is outside its vocabulary range.')

  def level_token(self, level: int, code: int) -> int:
    """Return the disjoint vocabulary token for one residual code."""
    if level < 0 or level >= len(self.codebook_sizes):
      raise ValueError('Residual level is outside the tokenizer range.')
    if code < 0 or code >= self.codebook_sizes[level]:
      raise ValueError('Residual code is outside the codebook range.')
    return self._level_offsets[level] + code

  def collision_token(self, code: int) -> int:
    if code < 0 or code >= self.collision_size:
      raise ValueError('Collision code is outside the tokenizer range.')
    return self._collision_offset + code

  def user_token(self, user_id: int) -> int:
    """Return a stable hashed user token without Python hash randomization."""
    digest = hashlib.blake2b(str(user_id).encode('utf-8'), digest_size=8).digest()
    bucket = int.from_bytes(digest, 'little') % self.user_bucket_count
    return self._user_offset + bucket

  def _encode_semantic_tuple(self, semantic_id: Sequence[int]) -> tuple[int, int, int, int]:
    return (
        self.level_token(0, int(semantic_id[0])),
        self.level_token(1, int(semantic_id[1])),
        self.level_token(2, int(semantic_id[2])),
        self.collision_token(int(semantic_id[3])),
    )

  def encode_item(self, item_id: int) -> list[int]:
    """Return the four vocabulary tokens for a catalog item."""
    try:
      return list(self._encode_semantic_tuple(self._item_ids[item_id]))
    except KeyError as error:
      raise ValueError(f'Unknown catalog item ID {item_id}.') from error

  def encode_history(self, user_id: int, history: Sequence[int]) -> list[int]:
    """Flatten a user history into one user token and item Semantic IDs."""
    return [
        self.user_token(user_id),
        *(token for item_id in history for token in self.encode_item(item_id)),
    ]

  def decode_ids(self, tokens: Sequence[int]) -> int:
    """Map a complete four-token Semantic ID back to its unique catalog item."""
    token_tuple = tuple(int(token) for token in tokens)
    if len(token_tuple) != 4 or token_tuple not in self._reverse_item_ids:
      raise ValueError('Tokens do not form a valid catalog Semantic ID.')
    return self._reverse_item_ids[token_tuple]

  def _semantic_prefix(self, prefix: Sequence[int]) -> tuple[int, ...]:
    tokens = tuple(int(token) for token in prefix)
    if tokens and (
        tokens[0] == self.bos_token_id
        or self._user_offset <= tokens[0] < self.vocab_size
    ):
      tokens = tokens[1:]
    return tokens

  def allowed_next(self, prefix: Sequence[int]) -> set[int]:
    """Return the legal next token set for a partial target Semantic ID."""
    semantic_prefix = self._semantic_prefix(prefix)
    if len(semantic_prefix) > 4:
      return set()
    current = self._trie
    for token in semantic_prefix:
      if token not in current:
        return set()
      current = current[token]
    if len(semantic_prefix) == 4:
      return {self.eos_token_id}
    return set(current)
