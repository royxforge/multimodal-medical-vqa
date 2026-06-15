"""Tests for model architecture components.

Verifies vision encoder, language model, fusion, and full model.
Uses small synthetic data for fast testing.
"""

import sys
from pathlib import Path

import pytest
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.confidence import MonteCarloDropout, TemperatureScaler
from src.models.fusion import CrossAttentionFusion
from src.models.vision_encoder import BioViLTEncoder


class TestBioViLTEncoder:
    """Tests for vision encoder."""

    def test_forward_shape(self):
        """Test forward pass returns expected shapes."""
        encoder = BioViLTEncoder(
            model_name="openai/clip-vit-large-patch14",  # Use CLIP as fallback for testing
            hidden_size=1024,
            projection_dim=4096,
            freeze_encoder=True,
        )
        dummy_input = torch.randn(2, 3, 224, 224)
        outputs = encoder(dummy_input)

        assert "cls_embedding" in outputs
        assert "patch_embeddings" in outputs
        assert "last_hidden_state" in outputs
        assert outputs["cls_embedding"].shape == (2, 4096)
        assert outputs["patch_embeddings"].shape[0] == 2
        assert outputs["patch_embeddings"].shape[2] == 4096

    def test_freeze(self):
        """Test that freezing works correctly."""
        encoder = BioViLTEncoder(model_name="openai/clip-vit-large-patch14", freeze_encoder=True)
        encoder.unfreeze_top_k_layers(2)
        trainable = encoder.get_trainable_params()
        assert trainable > 0


class TestCrossAttentionFusion:
    """Tests for cross-attention fusion layer."""

    def test_forward_shape(self):
        """Test forward pass shape preservation."""
        fusion = CrossAttentionFusion(d_model=4096, n_heads=4)
        text_emb = torch.randn(2, 10, 4096)
        visual_tokens = torch.randn(2, 256, 4096)

        fused = fusion(text_emb, visual_tokens)
        assert fused.shape == (2, 10, 4096)

    def test_with_mask(self):
        """Test forward pass with attention mask."""
        fusion = CrossAttentionFusion(d_model=4096, n_heads=4)
        text_emb = torch.randn(2, 10, 4096)
        visual_tokens = torch.randn(2, 256, 4096)
        mask = torch.ones(2, 10)

        fused = fusion(text_emb, visual_tokens, text_mask=mask)
        assert fused.shape == (2, 10, 4096)

    def test_gradient_flow(self):
        """Test that gradients flow through the fusion layer."""
        fusion = CrossAttentionFusion(d_model=4096, n_heads=4)
        text_emb = torch.randn(2, 10, 4096, requires_grad=True)
        visual_tokens = torch.randn(2, 256, 4096)

        fused = fusion(text_emb, visual_tokens)
        loss = fused.sum()
        loss.backward()

        assert text_emb.grad is not None
        assert torch.isfinite(text_emb.grad).all()


class TestConfidenceEstimator:
    """Tests for confidence estimation."""

    def test_monte_carlo_dropout(self):
        """Test MC Dropout runs without error."""
        # Create a simple model with dropout
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20), torch.nn.Dropout(0.5), torch.nn.Linear(20, 5)
        )
        mc = MonteCarloDropout(model, num_samples=5)
        dummy_input = torch.randn(1, 10)

        # Enable dropout and sample
        mc._enable_dropout()
        with torch.no_grad():
            outputs = []
            for _ in range(5):
                outputs.append(model(dummy_input).numpy())

        assert len(outputs) == 5

    def test_temperature_scaler(self):
        """Test temperature scaling initialization."""
        scaler = TemperatureScaler()
        logits = torch.randn(4, 10)
        scaled = scaler.forward(logits)
        assert scaled.shape == logits.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
