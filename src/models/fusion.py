"""Cross-attention fusion between visual and text modalities.

Design choice: Cross-attention vs Simple Concatenation
=====================================================
- **Simple concatenation** (concat [CLS; question_emb]) loses spatial structure
  and cannot align specific image regions to question tokens.
- **Cross-attention** allows each question token to attend to relevant image
  patches. This is crucial for medical VQA where the answer depends on
  localized findings (e.g., "Is there a nodule in the upper left lobe?").

The fusion module projects visual patch embeddings as keys/values and uses
the LLM's question token embeddings as queries in a multi-head attention layer.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFusion(nn.Module):
    """Lightweight cross-attention fusion between vision and text.

    Architecture:
        Query: text token embeddings (from LLM embedding layer)
        Key/Value: visual patch embeddings (from vision encoder)
        Output: fused representations with visual context injected into text

    Args:
        d_model: Model dimension (4096 for Mistral-7B).
        n_heads: Number of attention heads.
        dropout: Dropout rate.
        use_residual: Whether to add residual connection.
    """

    def __init__(
        self, d_model: int = 4096, n_heads: int = 4, dropout: float = 0.1, use_residual: bool = True
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_residual = use_residual

        # Cross-attention projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Layer norm and dropout
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Learned visual type embedding (distinguishes visual tokens from text tokens)
        self.visual_type_embedding = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Scale factor for attention
        self.scale = self.head_dim**-0.5

    def _reshape_for_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape for multi-head attention.

        Args:
            x: (B, seq_len, d_model)

        Returns:
            (B, n_heads, seq_len, head_dim)
        """
        B, seq_len, _ = x.shape
        return x.view(B, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        text_embeddings: torch.Tensor,
        visual_tokens: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        visual_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fuse visual information into text embeddings via cross-attention.

        Args:
            text_embeddings: Text token embeddings (B, T, d_model) from LLM.
            visual_tokens: Visual patch embeddings (B, V, d_model) from vision encoder.
            text_mask: Text attention mask (B, T) — 1 for valid tokens.
            visual_mask: Visual attention mask (B, V) — 1 for valid patches.

        Returns:
            Fused text embeddings (B, T, d_model) with visual context.
        """
        B, T, D = text_embeddings.shape

        # Add visual type embedding to visual tokens
        visual_tokens = visual_tokens + self.visual_type_embedding

        # Project to Q, K, V
        q = self.q_proj(text_embeddings)  # (B, T, D)
        k = self.k_proj(visual_tokens)  # (B, V, D)
        v = self.v_proj(visual_tokens)  # (B, V, D)

        # Reshape for multi-head attention
        q = self._reshape_for_attention(q)  # (B, n_heads, T, head_dim)
        k = self._reshape_for_attention(k)  # (B, n_heads, V, head_dim)
        v = self._reshape_for_attention(v)  # (B, n_heads, V, head_dim)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, n_heads, T, V)

        # Apply visual mask (if provided)
        if visual_mask is not None:
            # Expand mask: (B, V) -> (B, 1, 1, V)
            visual_mask_expanded = visual_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(~visual_mask_expanded.bool(), float("-inf"))

        # Softmax over visual tokens
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of visual values
        fused = torch.matmul(attn_weights, v)  # (B, n_heads, T, head_dim)

        # Reshape back
        fused = fused.transpose(1, 2).contiguous().view(B, T, D)

        # Output projection
        fused = self.out_proj(fused)
        fused = self.dropout(fused)

        # Residual connection
        if self.use_residual:
            fused = fused + text_embeddings

        # Final layer norm
        fused = self.norm(fused)

        return fused
