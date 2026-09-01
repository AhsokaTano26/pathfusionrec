"""Semantic, interaction, and fusion encoders for unified next-item ranking."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as functional

from pathfusionrec.models.bundle_encoder import BundleEncoder, BundleEncoderOutput


@dataclass
class NextItemOutput:
    """Query and candidate vectors used by a next-item scorer."""

    query: Tensor
    candidates: Tensor
    gate: Tensor | None = None


class PathFusionNextItemModel(nn.Module):
    """Compare semantic, interaction, concat-fusion, and gated-fusion models.

    Semantic inputs use a fixed upstream sentence embedding for each item and
    a ``BundleEncoder`` for the candidate's global text and local bundle
    structure. Interaction inputs use a trainable item-ID embedding. All four
    variants share the same scoring interface, so the training split and
    full-catalog evaluation implementation remain identical.
    """

    MODES = ('semantic', 'interaction', 'fusion')
    FUSION_STRATEGIES = ('concat', 'gated')

    def __init__(
        self,
        *,
        num_items: int,
        semantic_dim: int,
        hidden_dim: int,
        mode: str,
        fusion_strategy: str = 'concat',
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_items < 2:
            raise ValueError('num_items must include padding and at least one item.')
        if mode not in self.MODES:
            raise ValueError(f'mode must be one of {self.MODES}.')
        if fusion_strategy not in self.FUSION_STRATEGIES:
            raise ValueError(f'fusion_strategy must be one of {self.FUSION_STRATEGIES}.')

        self.mode = mode
        self.fusion_strategy = fusion_strategy
        self.semantic_dim = semantic_dim
        self.hidden_dim = hidden_dim
        self.interaction_embedding = nn.Embedding(num_items, hidden_dim, padding_idx=0)
        self.bundle_encoder = BundleEncoder(
            input_dim=semantic_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            field_names=('title', 'description', 'category'),
            dropout=dropout,
        )
        self.semantic_history_projection = nn.Sequential(
            nn.Linear(semantic_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.concat_query_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.concat_candidate_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.query_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.candidate_gate = nn.Linear(hidden_dim * 2, hidden_dim)

    def encode_bundle_candidates(
        self,
        global_fields: Mapping[str, Tensor],
        bundle_item_embeddings: Tensor,
        bundle_mask: Tensor,
    ) -> BundleEncoderOutput:
        """Encode candidate items from global fields and local bundles."""
        return self.bundle_encoder(global_fields, bundle_item_embeddings, bundle_mask)

    def encode_history(
        self,
        history_item_ids: Tensor,
        history_semantic_embeddings: Tensor,
        history_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return interaction and semantic history representations."""
        if history_item_ids.ndim != 2:
            raise ValueError('history_item_ids must have shape [batch, history_length].')
        if history_semantic_embeddings.shape != (*history_item_ids.shape, self.semantic_dim):
            raise ValueError('history_semantic_embeddings has an unexpected shape.')
        if history_mask.shape != history_item_ids.shape or history_mask.dtype != torch.bool:
            raise ValueError('history_mask must be a boolean tensor matching history_item_ids.')
        if not torch.all(history_mask.any(dim=1)):
            raise ValueError('Each history must contain at least one item.')

        weights = history_mask.unsqueeze(-1).to(history_semantic_embeddings.dtype)
        denominator = weights.sum(dim=1).clamp_min(1.0)
        interaction_history = (self.interaction_embedding(history_item_ids) * weights).sum(dim=1) / denominator
        semantic_history = (history_semantic_embeddings * weights).sum(dim=1) / denominator
        return interaction_history, self.semantic_history_projection(semantic_history)

    def forward(
        self,
        history_item_ids: Tensor,
        history_semantic_embeddings: Tensor,
        history_mask: Tensor,
        candidate_item_ids: Tensor,
        candidate_global_fields: Mapping[str, Tensor],
        candidate_bundle_embeddings: Tensor,
        candidate_bundle_mask: Tensor,
    ) -> NextItemOutput:
        """Encode one batch of histories and a shared candidate batch."""
        query, query_gate = self.encode_query(
            history_item_ids, history_semantic_embeddings, history_mask
        )
        candidates, candidate_gate = self.encode_candidates(
            candidate_item_ids,
            candidate_global_fields,
            candidate_bundle_embeddings,
            candidate_bundle_mask,
        )
        gate = None
        if query_gate is not None and candidate_gate is not None:
            gate = torch.cat([query_gate, candidate_gate], dim=0)
        return NextItemOutput(query=query, candidates=candidates, gate=gate)

    def encode_query(
        self,
        history_item_ids: Tensor,
        history_semantic_embeddings: Tensor,
        history_mask: Tensor,
    ) -> tuple[Tensor, Tensor | None]:
        """Encode histories independently so full-catalog vectors can be cached."""
        interaction_history, semantic_history = self.encode_history(
            history_item_ids, history_semantic_embeddings, history_mask
        )
        if self.mode == 'semantic':
            return semantic_history, None
        if self.mode == 'interaction':
            return interaction_history, None
        inputs = torch.cat([interaction_history, semantic_history], dim=-1)
        if self.fusion_strategy == 'concat':
            return self.concat_query_projection(inputs), None
        gate = torch.sigmoid(self.query_gate(inputs))
        return gate * interaction_history + (1.0 - gate) * semantic_history, gate

    def encode_candidates(
        self,
        candidate_item_ids: Tensor,
        candidate_global_fields: Mapping[str, Tensor],
        candidate_bundle_embeddings: Tensor,
        candidate_bundle_mask: Tensor,
    ) -> tuple[Tensor, Tensor | None]:
        """Encode candidate items independently for memory-efficient ranking."""
        bundle_output = self.encode_bundle_candidates(
            candidate_global_fields, candidate_bundle_embeddings, candidate_bundle_mask
        )
        interaction_candidates = self.interaction_embedding(candidate_item_ids)
        if self.mode == 'semantic':
            return bundle_output.embedding, None
        if self.mode == 'interaction':
            return interaction_candidates, None
        inputs = torch.cat([interaction_candidates, bundle_output.embedding], dim=-1)
        if self.fusion_strategy == 'concat':
            return self.concat_candidate_projection(inputs), None
        gate = torch.sigmoid(self.candidate_gate(inputs))
        return gate * interaction_candidates + (1.0 - gate) * bundle_output.embedding, gate

    @staticmethod
    def score(output: NextItemOutput) -> Tensor:
        """Return normalized dot-product scores for every query/candidate pair."""
        return functional.normalize(output.query, dim=-1) @ functional.normalize(
            output.candidates, dim=-1
        ).transpose(0, 1)
