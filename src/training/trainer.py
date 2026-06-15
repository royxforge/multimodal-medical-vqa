"""Custom HuggingFace Trainer for MedVQA training.

Subclasses the HF Trainer to:
1. Use our combined closed-ended + open-ended loss
2. Compute VQA-specific evaluation metrics
3. Log sample predictions every N steps
4. Integrate W&B logging
"""

from typing import Optional, Union

import torch
import wandb
from transformers import Trainer, TrainingArguments

from .losses import MedVQALoss


class MedVQATrainer(Trainer):
    """Custom Trainer for MedVQA with combined loss and VQA metrics.

    Args:
        model: MedVQAModel instance.
        args: HuggingFace TrainingArguments.
        loss_fn: MedVQALoss instance.
        train_dataset: Training dataset.
        eval_dataset: Evaluation dataset.
        tokenizer: Tokenizer for decoding predictions.
        data_collator: Collate function for DataLoader.
        callbacks: List of HF callbacks.
        compute_metrics: Optional metric computation function.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        args: TrainingArguments,
        loss_fn: MedVQALoss,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        data_collator=None,
        callbacks=None,
        compute_metrics=None,
    ):
        self.loss_fn = loss_fn
        self.tokenizer = tokenizer
        self._eval_step_counter = 0

        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=callbacks,
            compute_metrics=compute_metrics,
        )

    def compute_loss(
        self, model: torch.nn.Module, inputs: dict[str, torch.Tensor], return_outputs: bool = False
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict]]:
        """Override compute_loss to use our combined MedVQA loss.

        Args:
            model: The MedVQA model.
            inputs: Batch dict from DataLoader.
            return_outputs: Whether to also return model outputs.

        Returns:
            Loss tensor, or (loss, outputs) tuple if return_outputs.
        """
        # Extract batch
        images = inputs.get("images")
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        labels = inputs.get("labels")
        is_yesno = inputs.get("is_yesno")
        answer_labels = inputs.get("answer_labels")

        # Forward pass
        outputs = model(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_visual_features=True,
        )

        # Compute combined loss
        loss_dict = self.loss_fn(
            logits=outputs["logits"],
            labels=labels,
            is_yesno=is_yesno,
            yesno_logits=outputs.get("yesno_logits"),
            answer_labels=answer_labels,
            visual_features=outputs.get("visual_features"),
            lm_loss=outputs.get("loss"),
        )

        loss = loss_dict["loss"]

        # Log loss components
        if self.state.global_step % self.args.logging_steps == 0:
            self._log_loss_components(loss_dict)

        return (loss, outputs) if return_outputs else loss

    def _log_loss_components(self, loss_dict: dict[str, torch.Tensor]):
        """Log individual loss components to W&B.

        Args:
            loss_dict: Dict with loss components.
        """
        logs = {}
        for name, value in loss_dict.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                logs[f"loss/{name}"] = value.item()

        if logs:
            self.log(logs)

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Override prediction_step for generation during evaluation.

        Args:
            model: The model.
            inputs: Batch dict.
            prediction_loss_only: If True, only compute loss.
            ignore_keys: Keys to ignore in output.

        Returns:
            Tuple of (loss, generated_ids, labels).
        """
        images = inputs.get("images")
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        labels = inputs.get("labels")

        with torch.no_grad():
            # Compute loss
            loss = self.compute_loss(model, inputs) if not prediction_loss_only else None

            if prediction_loss_only:
                return (loss, None, None)

            # Generate answers
            generated_ids = model.generate(
                images=images, input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=64
            )

            return (loss, generated_ids, labels)

    def evaluate(
        self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"
    ) -> dict[str, float]:
        """Override evaluate to log sample predictions.

        Args:
            eval_dataset: Optional eval dataset.
            ignore_keys: Keys to ignore.
            metric_key_prefix: Prefix for metric names.

        Returns:
            Dict of evaluation metrics.
        """
        self._eval_step_counter += 1
        result = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        # Log sample predictions every 3 evaluations
        if self._eval_step_counter % 3 == 0 and self.tokenizer:
            self._log_sample_predictions(eval_dataset)

        return result

    def _log_sample_predictions(self, eval_dataset=None):
        """Log sample predictions with images and decoded answers.

        Args:
            eval_dataset: Dataset to sample from.
        """
        if eval_dataset is None:
            return

        import random

        # Sample a few examples
        n_samples = min(5, len(eval_dataset))
        indices = random.sample(range(len(eval_dataset)), n_samples)

        log_data = []
        for idx in indices:
            sample = eval_dataset[idx]

            # Get image and question
            image = sample["image"].unsqueeze(0).to(self.model.device)
            input_ids = sample["input_ids"].unsqueeze(0).to(self.model.device)
            attention_mask = sample["attention_mask"].unsqueeze(0).to(self.model.device)

            # Generate answer
            with torch.no_grad():
                generated = self.model.generate(
                    images=image, input_ids=input_ids, attention_mask=attention_mask
                )

            # Decode
            question_text = sample.get("question", "")
            answer_text = sample.get("answer", "")
            generated_text = self.tokenizer.decode_answer(generated[0])

            log_data.append(
                wandb.Image(
                    sample["image"],
                    caption=f"Q: {question_text}\nGT: {answer_text}\nPred: {generated_text}",
                )
            )

        if log_data:
            self.log({"sample_predictions": log_data})

    def create_model_card(self, *args, **kwargs):
        """Skip model card creation (memory-intensive for 7B models)."""
        pass
