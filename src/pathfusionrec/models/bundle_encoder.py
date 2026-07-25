"""Structured semantic encoder for variable-length product bundles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn


@dataclass
class BundleEncoderOutput:
    """Fused bundle representation and interpretable intermediate values."""

    embedding: Tensor
    global_embedding: Tensor
    local_embedding: Tensor
    field_attention: Tensor
    item_attention: Tensor


class BundleEncoder(nn.Module):
    """Encode global bundle semantics and context-aware item importance.

    Inputs are precomputed upstream embeddings. This deliberately keeps the
    choice of text model (e.g. sentence-transformers or BERT) separate from
    the structured bundle aggregation model.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        field_names: Sequence[str] = ('title', 'description', 'category'),
        attention_dim: int | None = None,
        dropout: float = 0.1,
        role_feature_dim: int | None = None,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or output_dim <= 0:
            raise ValueError('input_dim, hidden_dim, and output_dim must be positive.')
        if not field_names or len(set(field_names)) != len(field_names):
            raise ValueError('field_names must be non-empty and unique.')
        if role_feature_dim is not None and role_feature_dim <= 0:
            raise ValueError('role_feature_dim must be positive when provided.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.field_names = tuple(field_names)
        self.role_feature_dim = role_feature_dim
        attention_dim = attention_dim or hidden_dim

        self.field_projections = nn.ModuleDict(
            {name: nn.Linear(input_dim, hidden_dim) for name in self.field_names}
        )
        self.field_score = nn.Linear(hidden_dim, 1, bias=False)
        self.item_projection = nn.Linear(input_dim, hidden_dim)
        self.context_projection = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.item_attention_projection = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.role_projection = (
            nn.Linear(role_feature_dim, attention_dim, bias=False)
            if role_feature_dim is not None
            else None
        )
        self.item_attention_score = nn.Linear(attention_dim, 1, bias=False)
        self.global_norm = nn.LayerNorm(hidden_dim)
        self.local_norm = nn.LayerNorm(hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(output_dim),
        )

    def forward(
        self,
        global_fields: Mapping[str, Tensor],
        item_embeddings: Tensor,
        item_mask: Tensor,
        item_role_features: Tensor | None = None,
    ) -> BundleEncoderOutput:
        """Return a representation for each bundle.

        ``global_fields`` maps each configured name to ``[batch, input_dim]``.
        ``item_embeddings`` is ``[batch, max_items, input_dim]`` and
        ``item_mask`` is boolean with ``True`` for a real item. Optional role
        features can encode signals such as item quantity or price bucket.
        """

        batch_size = self._validate_inputs(
            global_fields, item_embeddings, item_mask, item_role_features
        )
        projected_fields = torch.stack(
            [torch.tanh(self.field_projections[name](global_fields[name])) for name in self.field_names],
            dim=1,
        )
        field_attention = torch.softmax(self.field_score(projected_fields).squeeze(-1), dim=1)
        global_embedding = self.global_norm(
            torch.sum(projected_fields * field_attention.unsqueeze(-1), dim=1)
        )

        item_hidden = torch.tanh(self.item_projection(item_embeddings))
        attention_hidden = self.item_attention_projection(item_hidden)
        attention_hidden = attention_hidden + self.context_projection(global_embedding).unsqueeze(1)
        if item_role_features is not None:
            if self.role_projection is None:
                raise ValueError('item_role_features requires role_feature_dim at initialization.')
            attention_hidden = attention_hidden + self.role_projection(item_role_features)

        item_logits = self.item_attention_score(torch.tanh(attention_hidden)).squeeze(-1)
        item_logits = item_logits.masked_fill(~item_mask, torch.finfo(item_logits.dtype).min)
        item_attention = torch.softmax(item_logits, dim=1)
        local_embedding = self.local_norm(
            torch.sum(item_hidden * item_attention.unsqueeze(-1), dim=1)
        )
        embedding = self.fusion(torch.cat([global_embedding, local_embedding], dim=-1))

        if embedding.shape[0] != batch_size:
            raise RuntimeError('Unexpected batch size change in BundleEncoder.')
        return BundleEncoderOutput(
            embedding=embedding,
            global_embedding=global_embedding,
            local_embedding=local_embedding,
            field_attention=field_attention,
            item_attention=item_attention,
        )

    def _validate_inputs(
        self,
        global_fields: Mapping[str, Tensor],
        item_embeddings: Tensor,
        item_mask: Tensor,
        item_role_features: Tensor | None,
    ) -> int:
        missing_fields = set(self.field_names).difference(global_fields)
        unexpected_fields = set(global_fields).difference(self.field_names)
        if missing_fields or unexpected_fields:
            raise ValueError(
                f'global_fields must contain exactly {self.field_names}; '
                f'missing={sorted(missing_fields)}, unexpected={sorted(unexpected_fields)}.'
            )
        if item_embeddings.ndim != 3 or item_embeddings.shape[-1] != self.input_dim:
            raise ValueError('item_embeddings must have shape [batch, max_items, input_dim].')
        if item_mask.ndim != 2 or item_mask.shape != item_embeddings.shape[:2]:
            raise ValueError('item_mask must have shape [batch, max_items].')
        if item_mask.dtype != torch.bool:
            raise TypeError('item_mask must be a boolean tensor.')
        if not torch.all(item_mask.any(dim=1)):
            raise ValueError('Each bundle must contain at least one unmasked item.')

        batch_size = item_embeddings.shape[0]
        for name in self.field_names:
            if global_fields[name].shape != (batch_size, self.input_dim):
                raise ValueError(f"global_fields['{name}'] must have shape [batch, input_dim].")
        if item_role_features is not None and self.role_feature_dim is not None:
            expected_shape = (*item_embeddings.shape[:2], self.role_feature_dim)
            if item_role_features.shape != expected_shape:
                raise ValueError(
                    'item_role_features must have shape [batch, max_items, role_feature_dim].'
                )
        return batch_size
