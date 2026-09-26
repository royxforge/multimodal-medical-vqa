"""Unit tests for the combined VQA loss (no model / no downloads required)."""

from __future__ import annotations

import torch

from src.training.losses import MedVQALoss, open_ended_loss


def _mixed_batch(batch: int = 4, seq: int = 6, vocab: int = 11, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(batch, seq, vocab, generator=g)
    labels = torch.randint(0, vocab, (batch, seq), generator=g)
    # Mask positions that survive the shift (labels[:, 1:]) so the masked vs
    # unmasked means genuinely differ: last column of every row.
    labels[:, -1] = -100
    labels[0, 2] = -100
    return logits, labels


def test_open_ended_loss_mean_matches_torch_reference():
    """reduction='mean' must equal torch's own masked mean, not an unmasked one."""
    logits, labels = _mixed_batch()

    ours = open_ended_loss(logits, labels)

    shift_logits = logits[..., :-1, :].reshape(-1, logits.size(-1))
    shift_labels = labels[..., 1:].reshape(-1)
    reference = torch.nn.functional.cross_entropy(
        shift_logits, shift_labels, ignore_index=-100, reduction="mean"
    )
    assert torch.allclose(ours, reference)

    unmasked_mean = torch.nn.functional.cross_entropy(
        shift_logits, shift_labels, ignore_index=-100, reduction="none"
    ).mean()
    assert not torch.allclose(ours, unmasked_mean), "masked mean must divide by valid tokens"


def test_open_ended_loss_none_reduces_per_row():
    logits, labels = _mixed_batch()
    per_row = open_ended_loss(logits, labels, reduction="none")
    assert per_row.shape == (4,)
    assert (per_row >= 0).all()
    # Manual check of row 0
    row_logits = logits[:1, :, :]
    row_labels = labels[:1, :]
    expected = open_ended_loss(row_logits, row_labels, reduction="mean")
    assert torch.allclose(per_row[0], expected)


def test_yes_no_rows_excluded_from_open_loss():
    """Double-counting regression: yes/no rows must not appear in open_loss."""
    logits, labels = _mixed_batch(batch=4, seed=1)
    is_yesno = torch.tensor([1, 0, 1, 0])
    yesno_logits = torch.randn(4, 2, generator=torch.Generator().manual_seed(2))

    loss_fn = MedVQALoss(closed_ended_alpha=0.5)
    out = loss_fn(logits, labels, is_yesno, yesno_logits=yesno_logits)

    # Expected open loss: mean over rows 1 and 3 only.
    per_row = open_ended_loss(logits, labels, reduction="none")
    expected_open = (per_row[1] + per_row[3]) / 2
    assert torch.allclose(out["open_loss"], expected_open, atol=1e-6)

    # Combined loss = alpha*closed + (1-alpha)*open (contrastive disabled)
    expected_total = 0.5 * out["closed_loss"] + 0.5 * out["open_loss"]
    assert torch.allclose(out["loss"], expected_total, atol=1e-6)
    assert torch.isfinite(out["loss"])


def test_batch_lm_loss_overridden_when_yes_no_present():
    """A precomputed scalar lm_loss includes yes/no rows and must not be used."""
    logits, labels = _mixed_batch(batch=4, seed=3)
    is_yesno = torch.tensor([1, 0, 1, 0])
    yesno_logits = torch.randn(4, 2, generator=torch.Generator().manual_seed(4))
    poisoned_lm_loss = torch.tensor(0.0)  # would silently zero the open branch

    loss_fn = MedVQALoss(closed_ended_alpha=0.5)
    out = loss_fn(
        logits,
        labels,
        is_yesno,
        yesno_logits=yesno_logits,
        lm_loss=poisoned_lm_loss,
    )
    per_row = open_ended_loss(logits, labels, reduction="none")
    expected_open = (per_row[1] + per_row[3]) / 2
    assert torch.allclose(out["open_loss"], expected_open, atol=1e-6)
    assert out["open_loss"].item() > 0


def test_all_open_batch_keeps_legacy_lm_loss_path():
    """No yes/no rows -> the precomputed lm_loss is still honoured."""
    logits, labels = _mixed_batch(batch=3, seed=5)
    is_yesno = torch.zeros(3, dtype=torch.long)
    lm_loss = torch.tensor(2.5)

    loss_fn = MedVQALoss(closed_ended_alpha=0.5)
    out = loss_fn(logits, labels, is_yesno, lm_loss=lm_loss)
    assert torch.allclose(out["open_loss"], lm_loss)
    # No closed head -> pure open loss weighting
    assert torch.allclose(out["loss"], 0.5 * lm_loss)


def test_all_yes_no_batch_open_loss_is_zero():
    """Every row trained by the closed head -> open branch contributes 0."""
    logits, labels = _mixed_batch(batch=3, seed=6)
    is_yesno = torch.ones(3, dtype=torch.long)
    yesno_logits = torch.randn(3, 2, generator=torch.Generator().manual_seed(7))

    loss_fn = MedVQALoss(closed_ended_alpha=0.5)
    out = loss_fn(logits, labels, is_yesno, yesno_logits=yesno_logits)
    assert out["open_loss"].item() == 0.0
    assert torch.allclose(
        out["loss"], 0.5 * out["closed_loss"], atol=1e-6
    )
    assert torch.isfinite(out["loss"])
