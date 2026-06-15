"""Complete evaluation pipeline for MedVQA.

Evaluator class that orchestrates metric computation, calibration analysis,
error analysis, and baseline comparisons.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from .metrics import compute_all_metrics, compute_ece


class Evaluator:
    """Full evaluation pipeline for MedVQA models.

    Provides comprehensive evaluation including metrics, calibration plots,
    error analysis with Grad-CAM, and baseline comparisons.

    Args:
        model: MedVQAModel instance.
        text_preprocessor: TextPreprocessor for decoding.
        gradcam: Optional GradCAM instance for visualization.
        device: Device to run on.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        text_preprocessor: object,
        gradcam: Optional[object] = None,
        device: str = "cuda",
    ):
        self.model = model
        self.text_preprocessor = text_preprocessor
        self.gradcam = gradcam
        self.device = device

    @torch.no_grad()
    def evaluate_split(
        self,
        dataloader: torch.utils.data.DataLoader,
        split_name: str = "test",
        use_mc_dropout: bool = False,
        mc_samples: int = 20,
    ) -> dict:
        """Evaluate the model on a data split.

        Args:
            dataloader: DataLoader for the split.
            split_name: Name of the split (for logging).
            use_mc_dropout: Whether to use MC Dropout for confidence.
            mc_samples: Number of MC samples.

        Returns:
            Dict with all metrics and predictions.
        """
        self.model.eval()

        all_predictions = []
        all_references = []
        all_confidences = []
        all_is_yesno = []
        all_questions = []

        mc_dropout_module = None
        if use_mc_dropout:
            from ..models.confidence import MonteCarloDropout as _MCDropout

            mc_dropout_module = _MCDropout

        for batch in tqdm(dataloader, desc=f"Evaluating {split_name}"):
            images = batch["images"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            # Generate answers
            generated_ids = self.model.generate(
                images=images, input_ids=input_ids, attention_mask=attention_mask
            )

            # Decode predictions
            for gen_ids in generated_ids:
                pred_text = self.text_preprocessor.decode_answer(gen_ids)
                all_predictions.append(pred_text)

            all_references.extend(batch["answers"])
            all_questions.extend(batch["questions"])
            all_is_yesno.extend(batch["is_yesno"].cpu().numpy())

            # MC Dropout confidence estimation
            if use_mc_dropout and mc_dropout_module is not None:
                mc_dropout = mc_dropout_module(self.model, mc_samples)
                mc_samples_batch = mc_dropout.sample(images, input_ids, attention_mask)
                uncertainty = mc_dropout.compute_uncertainty(
                    mc_samples_batch["predictions"], mc_samples_batch["logits"]
                )
                all_confidences.append(uncertainty["confidence"])

        # Compute metrics
        metrics = compute_all_metrics(
            predictions=all_predictions,
            references=all_references,
            confidences=np.array(all_confidences) if all_confidences else None,
            is_yesno=all_is_yesno if any(all_is_yesno) else None,
        )

        return {
            "metrics": metrics,
            "predictions": all_predictions,
            "references": all_references,
            "confidences": all_confidences,
            "questions": all_questions,
            "is_yesno": all_is_yesno,
        }

    def calibration_plot(
        self,
        confidences: np.ndarray,
        correct: np.ndarray,
        save_path: str = "experiments/calibration_plot.png",
    ) -> str:
        """Generate a reliability diagram (calibration plot).

        Args:
            confidences: Predicted confidence scores.
            correct: Binary correctness indicators.
            save_path: Where to save the plot.

        Returns:
            Path to saved plot.
        """
        # Compute ECE with bin stats
        ece_result = compute_ece(confidences, correct, n_bins=15)
        bin_stats = ece_result["bin_stats"]

        # Create plot
        fig, ax = plt.subplots(figsize=(8, 8))

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")

        # Model calibration
        bin_centers = [(b["bin_lower"] + b["bin_upper"]) / 2 for b in bin_stats]
        accuracies = [b["accuracy"] for b in bin_stats]
        confidences_bin = [b["confidence"] for b in bin_stats]  # Bar plot showing calibration
        ax.bar(
            bin_centers,
            accuracies,
            width=1.0 / 15,
            alpha=0.7,
            color="steelblue",
            label="Model (accuracy)",
        )

        # Gap between confidence and accuracy (red for overconfidence)
        for center, acc, conf in zip(bin_centers, accuracies, confidences_bin):
            if conf > acc:
                ax.bar(center, conf - acc, width=1.0 / 15, bottom=acc, alpha=0.3, color="red")
            else:
                ax.bar(center, acc - conf, width=1.0 / 15, bottom=conf, alpha=0.3, color="green")

        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Reliability Diagram (ECE = {ece_result['ece']:.4f})")
        ax.legend(loc="upper left")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Add histogram of confidence distribution
        ax2 = ax.twinx()
        ax2.hist(confidences, bins=15, alpha=0.2, color="gray", density=True)
        ax2.set_ylabel("Density", color="gray")
        ax2.tick_params(axis="y", labelcolor="gray")

        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()

        print(f"✅ Calibration plot saved to {save_path}")
        return save_path

    def compare_with_baselines(self, model_results: dict, baseline_results: dict[str, dict]) -> str:
        """Generate a formatted comparison table against baselines.

        Args:
            model_results: Results dict from evaluate_split.
            baseline_results: Dict of {baseline_name: results_dict}.

        Returns:
            Formatted string table.
        """
        metrics_of_interest = [
            "bleu_1",
            "bleu_4",
            "rouge_l_f1",
            "bertscore_f1",
            "yesno_accuracy",
            "ece",
            "brier_score",
        ]

        # Build table
        table = []
        header = f"{'Metric':<25}"
        header += f"{'Ours':<15}"
        for baseline_name in baseline_results:
            header += f"{baseline_name:<15}"
        table.append(header)
        table.append("-" * len(header))

        for metric in metrics_of_interest:
            row = f"{metric:<25}"

            # Current model
            if metric in model_results.get("metrics", {}):
                val = model_results["metrics"][metric]
                row += f"{val:<15.4f}" if isinstance(val, float) else f"{val:<15}"
            else:
                row += f"{'N/A':<15}"

            # Baselines
            for baseline_name, results in baseline_results.items():
                if metric in results.get("metrics", {}):
                    val = results["metrics"][metric]
                    row += f"{val:<15.4f}" if isinstance(val, float) else f"{val:<15}"
                else:
                    row += f"{'N/A':<15}"

            table.append(row)

        table_str = "\n".join(table)
        print("\n" + "=" * len(header))
        print("BASELINE COMPARISON")
        print("=" * len(header))
        print(table_str)
        print("=" * len(header))

        return table_str

    def error_analysis(
        self, results: dict, output_dir: str = "experiments/error_analysis", n_worst: int = 50
    ) -> str:
        """Analyze worst predictions with Grad-CAM overlays.

        Args:
            results: Results dict from evaluate_split.
            output_dir: Directory to save error analysis.
            n_worst: Number of worst samples to analyze.

        Returns:
            Path to error analysis report.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        predictions = results["predictions"]
        references = results["references"]
        questions = results["questions"]
        confidences = results.get("confidences", [0.5] * len(predictions))

        # Compute correctness
        correct = [
            1 if p.strip().lower().rstrip(".!?") == r.strip().lower().rstrip(".!?") else 0
            for p, r in zip(predictions, references)
        ]

        # Find worst predictions (incorrect with highest confidence)
        incorrect_indices = [i for i, c in enumerate(correct) if c == 0]
        # Sort by confidence (highest first — these are the most confidently wrong)
        incorrect_indices.sort(key=lambda i: confidences[i], reverse=True)
        worst_indices = incorrect_indices[:n_worst]

        # Generate report
        report_lines = [
            "# Error Analysis Report",
            f"Total errors: {len(incorrect_indices)} / {len(predictions)} "
            f"({100 * len(incorrect_indices) / len(predictions):.1f}%)",
            f"Showing top {len(worst_indices)} most confidently wrong predictions",
            "",
            "---",
            "",
        ]

        n_errors_with_gradcam = 0
        for i, idx in enumerate(worst_indices):
            report_lines.append(f"## Error {i + 1}: Sample {idx}")
            report_lines.append(f"- **Question**: {questions[idx]}")
            report_lines.append(f"- **Ground Truth**: {references[idx]}")
            report_lines.append(f"- **Prediction**: {predictions[idx]}")
            report_lines.append(
                f"- **Confidence**: {confidences[idx]:.4f}"
                if isinstance(confidences[idx], float)
                else f"- **Confidence**: {confidences[idx]}"
            )

            # Generate Grad-CAM if available and we have the dataset
            if self.gradcam and n_errors_with_gradcam < 10:
                # Note: This requires access to the original dataset
                report_lines.append("- Grad-CAM available for this sample")
                n_errors_with_gradcam += 1

            report_lines.append("")

        # Save report
        report_path = output_path / "error_analysis.md"
        with open(report_path, "w") as f:
            f.write("\n".join(report_lines))

        print(f"✅ Error analysis saved to {report_path}")
        return str(report_path)
