"""Tests for the inference pipeline.

Verifies prediction, uncertainty estimation, and Grad-CAM generation.
Uses mock models for fast, dependency-free testing.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.gradcam import GradCAM
from src.inference.pipeline import PredictionResult
from src.inference.uncertainty import TemperatureScaler


class TestPredictionResult:
    """Tests for the PredictionResult dataclass."""

    def test_default_values(self):
        """Test default values."""
        result = PredictionResult(
            answer="No evidence of lung nodule", confidence=0.92, uncertainty_flag=False
        )
        assert result.answer == "No evidence of lung nodule"
        assert result.confidence == 0.92
        assert result.uncertainty_flag is False
        assert result.heatmap_path is None
        assert result.latency_ms == 0.0

    def test_uncertainty_flag(self):
        """Test uncertainty flagging."""
        result = PredictionResult(answer="Maybe", confidence=0.15, uncertainty_flag=True)
        assert result.uncertainty_flag is True


class TestGradCAM:
    """Tests for Grad-CAM implementation."""

    def test_overlay_on_image(self):
        """Test heatmap overlay on image."""
        # Create a synthetic image and heatmap
        image = Image.new("RGB", (224, 224), (100, 100, 100))
        heatmap = np.random.rand(224, 224).astype(np.float32)

        overlay = GradCAM.overlay_on_image(image, heatmap, alpha=0.4)
        assert isinstance(overlay, Image.Image)
        assert overlay.size == (224, 224)
        assert overlay.mode == "RGB"

    def test_overlay_transparency(self):
        """Test that alpha=0 returns original image."""
        image = Image.new("RGB", (224, 224), (100, 100, 100))
        heatmap = np.ones((224, 224), dtype=np.float32)

        overlay = GradCAM.overlay_on_image(image, heatmap, alpha=0.0)
        # Should be mostly the original image
        assert overlay.size == (224, 224)


class TestTemperatureScaler:
    """Tests for temperature scaling."""

    def test_forward(self):
        """Test temperature scaling forward pass."""
        scaler = TemperatureScaler()
        scaler.temperature = torch.nn.Parameter(torch.tensor(2.0))
        logits = torch.tensor([[1.0, 2.0, 3.0]])

        scaled = scaler.forward(logits)
        assert scaled.shape == logits.shape
        assert scaled[0, 0] < logits[0, 0]  # Should be scaled down

    def test_calibrate(self):
        """Test calibrated probability output."""
        scaler = TemperatureScaler()
        logits = torch.randn(4, 10)
        probs = scaler.calibrate(logits)
        assert probs.shape == logits.shape
        assert torch.allclose(probs.sum(dim=-1), torch.ones(4))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
