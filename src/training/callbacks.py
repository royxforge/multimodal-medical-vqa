"""Training callbacks for MedVQA.

Includes early stopping, LR scheduling, and checkpoint management.
"""

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class EarlyStoppingCallback(TrainerCallback):
    """Early stopping callback based on validation metric.

    Stops training if the monitored metric does not improve for `patience` evaluations.

    Args:
        patience: Number of evaluations to wait for improvement.
        threshold: Minimum change to qualify as improvement.
        monitor: Metric to monitor (e.g., 'eval_loss').
        mode: 'min' or 'max' (whether lower or higher is better).
    """

    def __init__(
        self,
        patience: int = 3,
        threshold: float = 0.01,
        monitor: str = "eval_loss",
        mode: str = "min",
    ):
        self.patience = patience
        self.threshold = threshold
        self.monitor = monitor
        self.mode = mode
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.counter = 0
        self.early_stopped = False

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, float],
        **kwargs,
    ):
        """Check early stopping condition after each evaluation."""
        if self.monitor not in metrics:
            return

        current_value = metrics[self.monitor]

        # Check if improvement
        if self.mode == "min":
            improved = current_value < (self.best_value - self.threshold)
        else:
            improved = current_value > (self.best_value + self.threshold)

        if improved:
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            print(f"⏳ Early stopping counter: {self.counter}/{self.patience}")

            if self.counter >= self.patience:
                print(f"[STOP] Early stopping triggered at step {state.global_step}")
                control.should_training_stop = True
                self.early_stopped = True


class LoggingCallback(TrainerCallback):
    """Enhanced logging callback for training progress."""

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, float],
        **kwargs,
    ):
        """Print training progress with nice formatting."""
        if state.global_step % args.logging_steps == 0:
            step = state.global_step
            epoch = state.epoch

            # Format loss and LR
            loss = logs.get("loss", logs.get("eval_loss"))
            lr = logs.get("learning_rate", 0)

            loss_str = f"{loss:.4f}" if loss is not None else "N/A"
            lr_str = f"{lr:.2e}"

            print(f"  [{epoch:.2f} / Step {step:>6d}]  loss: {loss_str}  lr: {lr_str}")


def get_training_callbacks(
    early_stopping_patience: int = 3,
    early_stopping_threshold: float = 0.01,
    monitor: str = "eval_loss",
) -> list[TrainerCallback]:
    """Get the standard set of training callbacks.

    Args:
        early_stopping_patience: Patience for early stopping.
        early_stopping_threshold: Min improvement threshold.
        monitor: Metric to monitor.

    Returns:
        List of TrainerCallback instances.
    """
    return [
        EarlyStoppingCallback(
            patience=early_stopping_patience, threshold=early_stopping_threshold, monitor=monitor
        ),
        LoggingCallback(),
    ]
