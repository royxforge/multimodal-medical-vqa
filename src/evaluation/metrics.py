"""Comprehensive evaluation metrics for MedVQA.

Implements all standard VQA metrics plus confidence calibration metrics:

Closed-ended:
- Accuracy (yes/no binary classification)

Open-ended:
- BLEU-1/4 (n-gram overlap)
- ROUGE-L (longest common subsequence)
- BERTScore (semantic similarity using BERT embeddings)

Calibration:
- Expected Calibration Error (ECE) with 15 bins
- Brier Score (mean squared error between confidence and correctness)
- OOD Detection AUROC
"""

from typing import Optional

import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from rouge_score import rouge_scorer
from sklearn.metrics import auc, roc_curve


def compute_closed_ended_accuracy(
    predictions: list[str], references: list[str]
) -> dict[str, float]:
    """Compute accuracy for yes/no questions.

    Args:
        predictions: List of predicted answers.
        references: List of ground-truth answers.

    Returns:
        Dict with 'accuracy' key.
    """

    # Normalize to yes/no
    def normalize(text: str) -> str:
        text = text.strip().lower().rstrip(".!?")
        if text in ["yes", "yeah", "yep", "y"]:
            return "yes"
        if text in ["no", "nope", "n"]:
            return "no"
        return text

    preds_norm = [normalize(p) for p in predictions]
    refs_norm = [normalize(r) for r in references]

    correct = sum(1 for p, r in zip(preds_norm, refs_norm) if p == r)
    accuracy = correct / len(predictions) if predictions else 0.0

    return {"accuracy": accuracy, "correct": correct, "total": len(predictions)}


def compute_bleu(predictions: list[str], references: list[str], max_n: int = 4) -> dict[str, float]:
    """Compute BLEU scores up to n-gram order.

    Args:
        predictions: List of predicted answers.
        references: List of reference answers.
        max_n: Maximum n-gram order (default: 4).

    Returns:
        Dict with BLEU-1 through BLEU-4 scores.
    """
    # Tokenize
    pred_tokens = [p.lower().split() for p in predictions]
    ref_tokens = [[r.lower().split()] for r in references]

    smoothing = SmoothingFunction().method1

    scores = {}
    for n in range(1, max_n + 1):
        weights = tuple(1.0 / n if i < n else 0.0 for i in range(4))
        try:
            score = corpus_bleu(
                ref_tokens, pred_tokens, weights=weights, smoothing_function=smoothing
            )
        except Exception:
            score = 0.0
        scores[f"bleu_{n}"] = score

    return scores


def compute_rouge_l(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Compute ROUGE-L scores.

    Uses the rouge-score library for longest common subsequence matching.

    Args:
        predictions: List of predicted answers.
        references: List of reference answers.

    Returns:
        Dict with 'rouge_l_precision', 'rouge_l_recall', 'rouge_l_f1'.
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    scores = {"rouge_l_precision": [], "rouge_l_recall": [], "rouge_l_f1": []}

    for pred, ref in zip(predictions, references):
        result = scorer.score(ref, pred)
        scores["rouge_l_precision"].append(result["rougeL"].precision)
        scores["rouge_l_recall"].append(result["rougeL"].recall)
        scores["rouge_l_f1"].append(result["rougeL"].fmeasure)

    return {
        "rouge_l_precision": np.mean(scores["rouge_l_precision"]),
        "rouge_l_recall": np.mean(scores["rouge_l_recall"]),
        "rouge_l_f1": np.mean(scores["rouge_l_f1"]),
    }


def compute_bertscore(
    predictions: list[str],
    references: list[str],
    model_type: str = "distilbert-base-uncased",
    lang: str = "en",
) -> dict[str, float]:
    """Compute BERTScore for semantic similarity.

    Args:
        predictions: List of predicted answers.
        references: List of reference answers.
        model_type: BERT model for embeddings.
        lang: Language code.

    Returns:
        Dict with 'bertscore_precision', 'bertscore_recall', 'bertscore_f1'.
    """
    try:
        from bert_score import score as bert_score_fn

        P, R, F1 = bert_score_fn(
            predictions, references, model_type=model_type, lang=lang, verbose=False
        )

        return {
            "bertscore_precision": P.mean().item(),
            "bertscore_recall": R.mean().item(),
            "bertscore_f1": F1.mean().item(),
        }
    except ImportError:
        print("⚠️ bert-score not installed. Skipping BERTScore computation.")
        return {"bertscore_f1": 0.0}


def compute_ece(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> dict[str, float]:
    """Compute Expected Calibration Error (ECE).

    ECE measures the alignment between predicted confidence and actual accuracy.
    Lower is better. Well-calibrated models have ECE ≈ 0.

    Args:
        confidences: Predicted confidence scores (N,).
        correct: Binary correctness indicators (N,).
        n_bins: Number of bins for calibration (default: 15).

    Returns:
        Dict with 'ece', 'max_calibration_error', and per-bin stats.
    """
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    max_mce = 0.0
    bin_stats = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            avg_confidence = np.mean(confidences[in_bin])
            avg_accuracy = np.mean(correct[in_bin])
            calibration_error = np.abs(avg_accuracy - avg_confidence)
            ece += calibration_error * prop_in_bin
            max_mce = max(max_mce, calibration_error)

            bin_stats.append({
                "bin_lower": bin_lower,
                "bin_upper": bin_upper,
                "accuracy": avg_accuracy,
                "confidence": avg_confidence,
                "count": np.sum(in_bin),
            })

    return {"ece": ece, "max_calibration_error": max_mce, "n_bins": n_bins, "bin_stats": bin_stats}


def compute_brier_score(confidences: np.ndarray, correct: np.ndarray) -> float:
    """Compute Brier score (mean squared error) for confidence quality.

    Brier = (1/N) * sum((confidence - correctness)^2)

    Lower is better. Range: [0, 1].

    Args:
        confidences: Predicted confidence scores (N,).
        correct: Binary correctness indicators (N,).

    Returns:
        Brier score.
    """
    return float(np.mean((confidences - correct) ** 2))


def compute_ood_detection_auroc(id_confidences: np.ndarray, ood_confidences: np.ndarray) -> float:
    """Compute AUROC for out-of-distribution detection.

    Uses confidence scores to distinguish in-distribution (ID) from
    out-of-distribution (OOD) samples. Higher AUROC = better OOD detection.

    Args:
        id_confidences: Confidence scores on in-distribution data.
        ood_confidences: Confidence scores on out-of-distribution data.

    Returns:
        AUROC score.
    """
    # Create labels: 1 for ID, 0 for OOD
    labels = np.concatenate([np.ones_like(id_confidences), np.zeros_like(ood_confidences)])
    scores = np.concatenate([id_confidences, ood_confidences])

    # Compute ROC
    fpr, tpr, _ = roc_curve(labels, scores)
    auroc = auc(fpr, tpr)

    return auroc


def compute_all_metrics(
    predictions: list[str],
    references: list[str],
    confidences: Optional[np.ndarray] = None,
    is_yesno: Optional[list[bool]] = None,
    ood_confidences: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """Compute all evaluation metrics.

    This is the main entry point for evaluation, returning a structured
    dict matching the paper table format.

    Args:
        predictions: List of model predictions.
        references: List of ground-truth answers.
        confidences: Optional confidence scores for calibration metrics.
        is_yesno: Optional boolean mask for yes/no questions.
        ood_confidences: Optional OOD confidence scores for AUROC.

    Returns:
        Dict with all computed metrics.
    """
    metrics = {}

    # BLEU scores
    bleu_scores = compute_bleu(predictions, references)
    metrics.update(bleu_scores)

    # ROUGE-L
    rouge_scores = compute_rouge_l(predictions, references)
    metrics.update(rouge_scores)

    # BERTScore
    bertscore = compute_bertscore(predictions, references)
    metrics.update(bertscore)

    # Closed-ended accuracy (if yes/no mask provided)
    if is_yesno is not None and any(is_yesno):
        yesno_preds = [p for p, y in zip(predictions, is_yesno) if y]
        yesno_refs = [r for r, y in zip(references, is_yesno) if y]
        acc = compute_closed_ended_accuracy(yesno_preds, yesno_refs)
        metrics["yesno_accuracy"] = acc["accuracy"]

    # Calibration metrics (if confidences provided)
    if confidences is not None:
        # Compute correctness
        correct = np.array([
            1 if p.strip().lower().rstrip(".!?") == r.strip().lower().rstrip(".!?") else 0
            for p, r in zip(predictions, references)
        ])

        # ECE
        ece_result = compute_ece(confidences, correct)
        metrics["ece"] = ece_result["ece"]
        metrics["max_calibration_error"] = ece_result["max_calibration_error"]

        # Brier score
        metrics["brier_score"] = compute_brier_score(confidences, correct)

        # OOD AUROC
        if ood_confidences is not None:
            metrics["ood_auroc"] = compute_ood_detection_auroc(confidences, ood_confidences)

    return metrics
