"""Residual-quantized VAE tokenizer used by the TIGER retriever."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as functional


@dataclass(frozen=True)
class RQVAEOutput:
  """Outputs and loss terms of one RQ-VAE forward pass."""

  reconstruction: Tensor
  quantized: Tensor
  codes: Tensor
  reconstruction_loss: Tensor
  commitment_loss: Tensor
  codebook_loss: Tensor
  loss: Tensor


class ResidualQuantizedVAE(nn.Module):
  """Encode vectors into a tuple of residual-quantization code indices."""

  def __init__(
      self,
      input_dim: int,
      latent_dim: int,
      codebook_sizes: Sequence[int],
      commitment_weight: float = 0.25,
  ) -> None:
    super().__init__()
    if input_dim <= 0 or latent_dim <= 0 or not codebook_sizes:
      raise ValueError('Input dimension, latent dimension, and codebooks are required.')
    if any(size <= 1 for size in codebook_sizes):
      raise ValueError('Every residual codebook must contain at least two entries.')
    self.input_dim = input_dim
    self.latent_dim = latent_dim
    self.codebook_sizes = tuple(int(size) for size in codebook_sizes)
    self.commitment_weight = commitment_weight
    self.encoder = nn.Sequential(
        nn.Linear(input_dim, latent_dim),
        nn.GELU(),
        nn.Linear(latent_dim, latent_dim),
    )
    self.decoder = nn.Sequential(
        nn.Linear(latent_dim, latent_dim),
        nn.GELU(),
        nn.Linear(latent_dim, input_dim),
    )
    self.codebooks = nn.ModuleList(
        nn.Embedding(size, latent_dim) for size in self.codebook_sizes
    )

  def _quantize(self, latent: Tensor) -> tuple[Tensor, Tensor]:
    residual = latent
    quantized = torch.zeros_like(latent)
    codes: list[Tensor] = []
    for codebook in self.codebooks:
      distances = torch.cdist(residual, codebook.weight)
      code = distances.argmin(dim=1)
      selected = codebook(code)
      quantized = quantized + selected
      residual = residual - selected
      codes.append(code)
    return quantized, torch.stack(codes, dim=1)

  def forward(self, inputs: Tensor) -> RQVAEOutput:
    """Reconstruct inputs from their residual-quantized latent vectors."""
    if inputs.ndim != 2 or inputs.shape[1] != self.input_dim:
      raise ValueError('Inputs must have shape [batch_size, input_dim].')
    latent = self.encoder(inputs)
    quantized, codes = self._quantize(latent)
    quantized_st = latent + (quantized - latent).detach()
    reconstruction = self.decoder(quantized_st)
    reconstruction_loss = functional.mse_loss(reconstruction, inputs)
    commitment_loss = functional.mse_loss(latent, quantized.detach())
    codebook_loss = functional.mse_loss(quantized, latent.detach())
    loss = reconstruction_loss + self.commitment_weight * commitment_loss + codebook_loss
    return RQVAEOutput(
        reconstruction=reconstruction,
        quantized=quantized,
        codes=codes,
        reconstruction_loss=reconstruction_loss,
        commitment_loss=commitment_loss,
        codebook_loss=codebook_loss,
        loss=loss,
    )

  @torch.no_grad()
  def semantic_ids(self, vectors: Tensor) -> np.ndarray:
    """Return one residual-code tuple for each vector."""
    self.eval()
    latent = self.encoder(vectors)
    _, codes = self._quantize(latent)
    return codes.cpu().numpy().astype(np.int64, copy=False)

  @torch.no_grad()
  def initialize_codebooks_kmeans(self, vectors: Tensor, seed: int) -> None:
    """Initialize each codebook from the residuals of input vectors."""
    try:
      from sklearn.cluster import KMeans
    except ImportError as error:
      raise RuntimeError('scikit-learn is required for codebook initialization.') from error
    if vectors.ndim != 2 or vectors.shape[1] != self.input_dim:
      raise ValueError('Vectors must have shape [n_items, input_dim].')
    vectors = vectors.to(device=next(self.parameters()).device)
    latent = self.encoder(vectors).cpu().numpy()
    residual = latent
    for level, codebook in enumerate(self.codebooks):
      if residual.shape[0] < self.codebook_sizes[level]:
        raise ValueError('Codebook size cannot exceed available training items.')
      clusters = KMeans(
          n_clusters=self.codebook_sizes[level], n_init=10, random_state=seed
      ).fit(residual)
      centers = torch.from_numpy(clusters.cluster_centers_).to(
          device=codebook.weight.device, dtype=codebook.weight.dtype
      )
      codebook.weight.copy_(centers)
      residual = residual - centers[clusters.labels_].cpu().numpy()

  @torch.no_grad()
  def codebook_diagnostics(self, codes: np.ndarray) -> list[dict[str, float | int]]:
    """Summarize use and entropy for every residual codebook."""
    if codes.ndim != 2 or codes.shape[1] != len(self.codebooks):
      raise ValueError('Codes must have one column per residual codebook.')
    diagnostics: list[dict[str, float | int]] = []
    for level, size in enumerate(self.codebook_sizes):
      counts = np.bincount(codes[:, level], minlength=size)
      probabilities = counts / counts.sum()
      nonzero = probabilities[probabilities > 0]
      diagnostics.append(
          {
              'level': level,
              'size': size,
              'used_codes': int((counts > 0).sum()),
              'dead_codes': int((counts == 0).sum()),
              'perplexity': float(np.exp(-(nonzero * np.log(nonzero)).sum())),
          }
      )
    return diagnostics


def append_collision_codes(
    item_ids: Sequence[int], codes: np.ndarray
) -> list[tuple[int, int, int, int]]:
  """Append deterministic suffixes to make residual-code tuples unique."""
  if codes.ndim != 2 or codes.shape[1] != 3 or len(item_ids) != codes.shape[0]:
    raise ValueError('Exactly three residual codes are required for every item.')
  groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
  for index, code in enumerate(codes.tolist()):
    groups[tuple(int(value) for value in code)].append(index)
  result: list[tuple[int, int, int, int] | None] = [None] * len(item_ids)
  for code, indices in groups.items():
    for collision_code, index in enumerate(indices):
      result[index] = (*code, collision_code)
  return [semantic_id for semantic_id in result if semantic_id is not None]
