"""Uncertainty estimation utilities for inference.

Extends the confidence module with inference-specific functionality:
- Monte Carlo Dropout inference
- Temperature scaling post-hoc calibration
- Uncertainty-aware answer decoding
"""

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


class MonteCarloDropout:
    """Monte Carlo Dropout for uncertainty-aware inference.

    Runs multiple forward passes with dropout active to estimate
    prediction uncertainty. Higher variance = less certain.

    Args:
        model: The MedVQA model.
        num_samples: Number of MC samples (default: 20).
    """

    def __init__(self, model: torch.nn.Module, num_samples: int = 20):
        self.model = model
        self.num_samples = num_samples

    def _enable_dropout(self):
        """Enable dropout layers for MC sampling."""
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.train()

    @torch.no_grad()
    def sample(
        self,
        images: torch.Tensor,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor,
        vision_outputs: Optional[dict[str, torch.Tensor]] = None,
    ) -> dict[str, np.ndarray]:
        """Run MC Dropout sampling.

        Args:
            images: Batch of images (1, 3, H, W).
            input_ids: Tokenized question (1, T).
            attention_mask: Attention mask (1, T).

        Returns:
            Dict with predictions and logits arrays.
        """
        self.model.eval()
        self._enable_dropout()

        all_logits = []
        all_predictions = []

        for _ in range(self.num_samples):
            outputs = self.model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                vision_outputs=vision_outputs,
            )
            logits = outputs["logits"]
            last_logits = logits[:, -1, :]  # (1, vocab_size)
            probs = F.softmax(last_logits, dim=-1)
            pred = probs.argmax(dim=-1)

            all_logits.append(last_logits.cpu().numpy())
            all_predictions.append(pred.cpu().numpy())

        return {
            "predictions": np.stack(all_predictions, axis=1),  # (1, N)
            "logits": np.stack(all_logits, axis=1),  # (1, N, vocab_size)
        }

    def compute_uncertainty(
        self,
        predictions: np.ndarray,
        logits: np.ndarray,
        temperature_scaler: Optional["TemperatureScaler"] = None,
    ) -> dict:
        """Compute uncertainty metrics from MC samples.

        Args:
            predictions: MC sample predictions (1, N).
            logits: MC sample logits (1, N, vocab_size).

        Returns:
            Dict with confidence, entropy, mutual information.
        """
        from scipy.special import softmax

        if temperature_scaler is not None and temperature_scaler._fitted:
            scaled_logits = logits[0] / float(temperature_scaler.temperature.item())
            probs = softmax(scaled_logits, axis=-1)
        else:
            probs = softmax(logits[0], axis=-1)  # (N, vocab_size)

        mean_probs = probs.mean(axis=0)  # (vocab_size,)

        # Predictive entropy
        entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10))

        # Mutual information
        expected_entropy = -np.mean(np.sum(probs * np.log(probs + 1e-10), axis=1))
        mutual_info = entropy - expected_entropy

        # Confidence = max mean probability
        confidence = float(np.max(mean_probs))

        return {
            "confidence": confidence,
            "predictive_entropy": float(entropy),
            "mutual_information": float(mutual_info),
            "mean_probs": mean_probs,
        }


class TemperatureScaler:
    """Temperature scaling for post-hoc confidence calibration.

    Learns a temperature parameter T > 0 that scales logits:
        p = softmax(logits / T)

    T > 1 → more uniform (less confident)
    T < 1 → more peaked (more confident)

    Fitted on the validation set after training.
    """

    def __init__(self):
        self.temperature = torch.nn.Parameter(torch.ones(1) * 1.5)
        self._fitted = False

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling.

        Args:
            logits: Raw logits (B, C).

        Returns:
            Temperature-scaled logits.
        """
        return logits / self.temperature

    def fit(self, val_logits: list[torch.Tensor], val_labels: list[torch.Tensor]):
        """Fit temperature to minimize NLL on validation set.

        Args:
            val_logits: List of validation logit tensors.
            val_labels: List of validation label tensors.
        """
        import torch.optim as optim

        all_logits = torch.cat([logit.detach().cpu() for logit in val_logits])
        all_labels = torch.cat([label.detach().cpu() for label in val_labels])

        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        loss_fn = torch.nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()
            loss_val = loss_fn(self.forward(all_logits), all_labels)
            loss_val.backward()
            return loss_val

        optimizer.step(closure)
        self._fitted = True

        print(f"✅ Temperature fitted: T = {self.temperature.item():.4f}")

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """Get calibrated probabilities.

        Args:
            logits: Raw logits.

        Returns:
            Calibrated probabilities.
        """
        if not self._fitted:
            print("⚠️ Temperature not fitted yet. Using T=1.0")
        return F.softmax(self.forward(logits), dim=-1)
