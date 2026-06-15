"""Confidence estimation module for MedVQA.

Port and adaptation of the Self-Diagnosing Neural Models prior work.
Provides calibrated confidence scores and uncertainty flags via
temperature scaling and Monte Carlo Dropout.

The canonical implementations of MonteCarloDropout and TemperatureScaler
live in src.inference.uncertainty; this module re-exports them and adds
the ConfidenceResult dataclass and the high-level ConfidenceEstimator.

Reference: Self-Diagnosing Neural Models — confidence estimation through
temperature scaling and predictive entropy.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

# Import canonical implementations from inference module
from ..inference.uncertainty import MonteCarloDropout as _MonteCarloDropout
from ..inference.uncertainty import TemperatureScaler as _TemperatureScaler


@dataclass
class ConfidenceResult:
    """Structured output from the confidence estimator.

    Attributes:
        confidence: Calibrated confidence score in [0, 1].
        uncertainty_flag: True if model is not confident.
        predictive_entropy: Entropy of the predictive distribution.
        mutual_information: Mutual information (MC Dropout measure).
        mean_prediction: Mean prediction across MC samples.
    """

    confidence: float
    uncertainty_flag: bool
    predictive_entropy: float
    mutual_information: float
    mean_prediction: np.ndarray


# Re-export canonical classes so existing imports keep working:
#   from src.models.confidence import MonteCarloDropout
#   from src.models.confidence import TemperatureScaler
MonteCarloDropout = _MonteCarloDropout
TemperatureScaler = _TemperatureScaler


class ConfidenceEstimator(nn.Module):
    """Combined confidence estimator with temperature scaling and MC Dropout.

    This module is the direct adaptation of the Self-Diagnosing Neural Models
    approach, providing calibrated confidence scores and uncertainty flags.

    Args:
        base_model: The MedVQAModel instance.
        temperature_scaling: Whether to use temperature scaling.
        mc_dropout_samples: Number of MC Dropout samples.
        uncertainty_threshold: Threshold below which to flag uncertainty.
    """

    def __init__(
        self,
        base_model: torch.nn.Module,
        temperature_scaling: bool = True,
        mc_dropout_samples: int = 20,
        uncertainty_threshold: float = 0.3,
    ):
        super().__init__()
        self.base_model = base_model
        self.temperature_scaler = TemperatureScaler() if temperature_scaling else None
        self.mc_dropout = MonteCarloDropout(base_model, mc_dropout_samples)
        self.uncertainty_threshold = uncertainty_threshold

    def forward(
        self, images: torch.Tensor, input_ids: torch.LongTensor, attention_mask: torch.FloatTensor
    ) -> dict:
        """Estimate confidence for a batch of predictions.

        Args:
            images: Batch of images (B, 3, H, W).
            input_ids: Tokenized question IDs (B, T).
            attention_mask: Question attention mask (B, T).

        Returns:
            Dict with:
                - predictions: Generated answer token IDs (B, gen_len)
                - confidence_results: List of ConfidenceResult objects
                - answer_texts: Decoded answer strings
        """
        # Get MC Dropout samples
        mc_samples = self.mc_dropout.sample(images, input_ids, attention_mask)

        # Compute uncertainty metrics (returns dict with confidence, entropy, etc.)
        uncertainty_dict = self.mc_dropout.compute_uncertainty(
            mc_samples["predictions"],
            mc_samples["logits"],
            temperature_scaler=self.temperature_scaler,
        )

        # Convert to ConfidenceResult list for API consistency
        confidence_results = [
            ConfidenceResult(
                confidence=uncertainty_dict["confidence"],
                uncertainty_flag=uncertainty_dict["confidence"] < self.uncertainty_threshold,
                predictive_entropy=uncertainty_dict["predictive_entropy"],
                mutual_information=uncertainty_dict["mutual_information"],
                mean_prediction=uncertainty_dict["mean_probs"],
            )
        ]

        # Generate final answer using the base model
        outputs = self.base_model.generate(
            images=images, input_ids=input_ids, attention_mask=attention_mask
        )

        return {"predictions": outputs, "confidence_results": confidence_results}
