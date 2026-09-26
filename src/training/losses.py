"""Loss functions for MedVQA training.

Combines:
- Binary cross-entropy with label smoothing (yes/no questions)
- Causal language modeling loss (open-ended questions)
- Optional contrastive loss between visual CLS and answer embeddings

The combined loss uses a tunable alpha parameter to balance closed-ended
and open-ended question losses.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def closed_ended_loss(
    logits: torch.Tensor, labels: torch.Tensor, label_smoothing: float = 0.1
) -> torch.Tensor:
    """Binary cross-entropy loss for yes/no questions with label smoothing.

    Label smoothing prevents the model from becoming overconfident, which
    improves confidence calibration — a key requirement for medical AI.

    Args:
        logits: Raw logits from the model (B, 2) — [no, yes] or [yes, no].
        labels: Binary labels (B,) — 0 for no, 1 for yes.
        label_smoothing: Smoothing factor (default: 0.1).

    Returns:
        Scalar loss tensor.
    """
    # Apply label smoothing
    if label_smoothing > 0:
        n_classes = 2
        smooth_labels = labels * (1 - label_smoothing) + label_smoothing / n_classes
    else:
        smooth_labels = labels

    # Binary cross-entropy
    loss = F.binary_cross_entropy_with_logits(
        logits[:, 1].float(),
        smooth_labels.float(),  # Take the "yes" logit
    )
    return loss


def open_ended_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Causal language modeling loss for open-ended answer generation.

    Computes cross-entropy only over answer tokens (padding tokens are
    masked with -100 in the labels tensor).

    Args:
        logits: Model output logits (B, seq_len, vocab_size).
        labels: Target token IDs with -100 for masked positions (B, seq_len).
        label_mask: Optional additional mask (B, seq_len).
        reduction: "mean" (legacy, mean over all valid tokens in the batch)
            or "none" (one mean value per batch row, 0.0 for rows whose
            labels are entirely -100). "none" lets callers exclude specific
            rows before averaging.

    Returns:
        Scalar loss tensor ("mean") or a (B,) tensor ("none").
    """
    # Align labels if a visual token was prepended to logits
    if labels.shape[1] == logits.shape[1] - 1:
        pad = torch.full((labels.shape[0], 1), -100, device=labels.device, dtype=labels.dtype)
        labels = torch.cat([pad, labels], dim=1)

    # Shift logits and labels for next-token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten
    batch_size = shift_labels.shape[0]
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)

    # Standard cross-entropy (ignores -100 positions automatically)
    per_token = F.cross_entropy(
        shift_logits.float(), shift_labels, ignore_index=-100, reduction="none"
    )
    valid = (shift_labels != -100).float()

    if reduction == "none":
        per_token = per_token.view(batch_size, -1)
        valid = valid.view(batch_size, -1)
        sums = (per_token * valid).sum(dim=1)
        counts = valid.sum(dim=1)
        return torch.where(counts > 0, sums / counts.clamp(min=1.0), torch.zeros_like(sums))

    # Match torch's reduction="mean": sum over valid tokens / valid count
    # (per_token.mean() would dilute by the -100 masked positions).
    total = per_token.sum()
    count = valid.sum()
    return total / count if count > 0 else total


def contrastive_loss(
    visual_features: torch.Tensor, answer_features: torch.Tensor, temperature: float = 0.07
) -> torch.Tensor:
    """Contrastive loss between visual CLS and answer embeddings.

    This is an optional auxiliary loss that helps align visual and textual
    representations in the shared embedding space. When enabled, it encourages
    the model to learn that similar images have similar answers.

    Based on the CLIP contrastive loss formulation.

    Args:
        visual_features: Visual CLS embeddings (B, D).
        answer_features: Answer text embeddings (B, D).
        temperature: Softmax temperature (default: 0.07).

    Returns:
        Scalar loss tensor.
    """
    B = visual_features.shape[0]

    # Normalize features
    visual_features = F.normalize(visual_features, dim=-1)
    answer_features = F.normalize(answer_features, dim=-1)

    # Compute similarity matrix
    logits = torch.matmul(visual_features, answer_features.T) / temperature

    # Labels: diagonal (matching image-answer pairs)
    labels = torch.arange(B, device=visual_features.device)

    # Symmetric cross-entropy loss
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)

    return (loss_i + loss_t) / 2


class MedVQALoss(nn.Module):
    """Combined loss function for MedVQA training.

    Combines closed-ended and open-ended losses with a tunable alpha weight.
    Optionally includes a contrastive auxiliary loss.

    The closed-ended loss uses the model's yesno_head (nn.Linear(4096, 2))
    which produces two logits: [no_logit, yes_logit]. Binary cross-entropy
    with label smoothing is applied.

    The open-ended loss uses causal language modeling cross-entropy.

    Args:
        closed_ended_alpha: Weight for closed-ended loss (default: 0.5).
        label_smoothing: Label smoothing for closed-ended questions.
        use_contrastive: Whether to include contrastive loss.
        contrastive_weight: Weight for contrastive loss if enabled.
    """

    def __init__(
        self,
        closed_ended_alpha: float = 0.5,
        label_smoothing: float = 0.1,
        use_contrastive: bool = False,
        contrastive_weight: float = 0.1,
    ):
        super().__init__()
        self.closed_ended_alpha = closed_ended_alpha
        self.label_smoothing = label_smoothing
        self.use_contrastive = use_contrastive
        self.contrastive_weight = contrastive_weight

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        is_yesno: torch.Tensor,
        yesno_logits: Optional[torch.Tensor] = None,
        visual_features: Optional[torch.Tensor] = None,
        answer_features: Optional[torch.Tensor] = None,
        answer_labels: Optional[torch.Tensor] = None,
        lm_loss: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Compute combined loss for a batch.

        Closed-ended loss is applied only to yes/no questions (where
        is_yesno == 1). When the closed head contributes, open-ended loss is
        applied to the remaining (open-ended) questions only - yes/no rows are
        excluded so they are never counted in both branches.

        Args:
            logits: Model LM logits (B, seq_len, vocab_size).
            labels: Target labels with -100 masking (B, seq_len).
            is_yesno: Binary indicator for yes/no questions (B,).
            yesno_logits: Yes/no head logits (B, 2) from medvqa_model.
            visual_features: Visual CLS embeddings (B, D) for contrastive.
            answer_features: Answer embeddings (B, D) for contrastive.
            answer_labels: Binary labels for yes/no questions (B,) where
                          0=no, 1=yes, -1=not yes/no.

        Returns:
            Dict with:
                - 'loss': Combined loss value
                - 'closed_loss': Closed-ended loss component
                - 'open_loss': Open-ended loss component
                - 'contrastive_loss': Contrastive loss component (if enabled)
        """
        total_loss = 0.0
        losses = {}

        # Yes/no rows are trained through the closed-ended head below. When
        # that head contributes they must be excluded from the open-ended LM
        # loss, otherwise every yes/no question is counted twice (once as
        # classification, once as language modelling) despite the alpha
        # weighting implying disjoint groups.
        yesno_mask = (
            (is_yesno > 0).float()
            if is_yesno is not None
            else torch.zeros(logits.shape[0], device=logits.device)
        )

        # Closed-ended loss (yes/no questions) using the classification head
        if yesno_logits is not None:
            # Only compute closed-ended loss for yes/no questions

            if answer_labels is not None:
                # Mask to only use yes/no questions (-1 means not yes/no)
                valid_mask = (answer_labels >= 0).float()
                binary_labels = answer_labels.float().clamp(0, 1)
            else:
                valid_mask = yesno_mask
                binary_labels = torch.full(
                    (yesno_logits.shape[0],), 0.5, device=yesno_logits.device
                )

            # Apply label smoothing (only on valid yes/no samples)
            if self.label_smoothing > 0 and valid_mask.sum() > 0:
                smooth_labels = (
                    binary_labels * (1 - self.label_smoothing) + self.label_smoothing / 2
                )
            else:
                smooth_labels = binary_labels

            # Binary cross-entropy on the yes logit (index 1)
            # Only computed on valid yes/no samples, masked via reduction
            bce_per_sample = F.binary_cross_entropy_with_logits(
                yesno_logits[:, 1].float(),
                smooth_labels.float(),
                reduction="none",  # yes-logit
            )
            # Mask out non-yes/no samples
            masked_bce = bce_per_sample * valid_mask
            closed_loss = masked_bce.sum() / valid_mask.sum().clamp(
                min=1
            )  # 0 when no yes/no samples

            losses["closed_loss"] = closed_loss
            total_loss += self.closed_ended_alpha * closed_loss

        # Open-ended loss (open-ended questions only when the closed head ran;
        # with no closed head the yes/no rows are trained solely via the LM
        # and must stay in).
        if yesno_logits is not None and yesno_mask.sum().item() > 0:
            # Precomputed `lm_loss` is a batch scalar covering every row, so it
            # cannot be un-mixed here: recompute per row and drop yes/no rows.
            per_row = open_ended_loss(logits, labels, reduction="none")
            keep = 1.0 - yesno_mask
            open_loss = (per_row * keep).sum() / keep.sum().clamp(min=1.0)
        else:
            open_loss = lm_loss if lm_loss is not None else open_ended_loss(logits, labels)
        losses["open_loss"] = open_loss
        total_loss += (1 - self.closed_ended_alpha) * open_loss

        # Optional contrastive loss
        if self.use_contrastive and visual_features is not None and answer_features is not None:
            cont_loss = contrastive_loss(visual_features, answer_features)
            losses["contrastive_loss"] = cont_loss
            total_loss += self.contrastive_weight * cont_loss

        losses["loss"] = total_loss
        return losses
