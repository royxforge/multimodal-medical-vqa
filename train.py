#!/usr/bin/env python3
"""Main training script for MedVQA.

Usage:
    python train.py --config configs/default_config.yaml

Loads config, sets up model, data, and trainer, then trains and saves the model.
"""

import argparse
import os
import sys
from pathlib import Path

# Force UTF-8 for stdout/stderr (handles emoji in Windows consoles)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    import wandb
except ImportError:
    wandb = None
    print("⚠️ wandb not available — logging disabled")
from transformers import TrainingArguments

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.augmentation import MedicalAugmentationPipeline
from src.data.loader import PathVQADataset, VQARADDataset, collate_fn
from src.data.preprocessor import MedicalImagePreprocessor, TextPreprocessor
from src.models.fusion import CrossAttentionFusion
from src.models.language_model import MistralQLoRA
from src.models.medvqa_model import MedVQAModel
from src.models.vision_encoder import BioViLTEncoder
from src.training.callbacks import get_training_callbacks
from src.training.losses import MedVQALoss
from src.training.trainer import MedVQATrainer
from src.utils.config import MedVQAConfig
from src.utils.logger import setup_logger
from src.utils.reproducibility import print_system_info, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train MedVQA model")
    parser.add_argument(
        "--config", type=str, default="configs/default_config.yaml", help="Path to YAML config file"
    )
    parser.add_argument(
        "--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument("--debug", action="store_true", help="Debug mode: use small subset of data")
    return parser.parse_args()


def setup_data(config, logger):
    """Set up data preprocessing and datasets.

    Args:
        config: MedVQAConfig instance.
        logger: Logger instance.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, text_preprocessor).
    """
    logger.info("Setting up data pipeline...")

    # Image preprocessor
    image_processor = MedicalImagePreprocessor(image_size=config.data.image_size)

    # Text preprocessor
    text_preprocessor = TextPreprocessor(
        model_name=config.data.tokenizer_name,
        max_question_length=config.data.max_question_length,
        max_answer_length=config.data.max_answer_length,
    )

    # Augmentation pipeline (training only)
    augmentation = MedicalAugmentationPipeline(
        image_size=config.data.image_size,
        rotation_limit=config.data.rotation_degrees,
        brightness_range=config.data.brightness_range,
        contrast_range=config.data.contrast_range,
        min_crop_retention=config.data.min_crop_retention,
        noise_sigma=config.data.gaussian_noise_sigma,
    )

    def train_transform(image):
        if config.data.use_augmentation:
            image = augmentation.augment_pil(image)
        return image_processor(image)

    # Resolve dataset directory based on dataset_name
    dataset_name = config.data.dataset_name.lower().replace("-", "_")
    raw_dir = Path(config.data.raw_dir)
    if dataset_name == "vqa_rad":
        dataset_dir = raw_dir / "vqa_rad"
        dataset_cls = VQARADDataset
    elif dataset_name == "pathvqa":
        dataset_dir = raw_dir / "pathvqa"
        dataset_cls = PathVQADataset
    else:
        dataset_dir = raw_dir
        dataset_cls = VQARADDataset

    if not dataset_dir.exists():
        dataset_dir = raw_dir

    # Datasets
    train_dataset = dataset_cls(
        data_dir=str(dataset_dir),
        split="train",
        image_transform=train_transform,
        text_preprocessor=text_preprocessor,
    )
    val_dataset = dataset_cls(
        data_dir=str(dataset_dir),
        split="val",
        image_transform=image_processor,
        text_preprocessor=text_preprocessor,
    )
    test_dataset = dataset_cls(
        data_dir=str(dataset_dir),
        split="test",
        image_transform=image_processor,
        text_preprocessor=text_preprocessor,
    )

    logger.info(
        f"Datasets: train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)}"
    )

    return train_dataset, val_dataset, test_dataset, text_preprocessor


def setup_model(config, logger):
    """Set up the MedVQA model components.

    Args:
        config: MedVQAConfig instance.
        logger: Logger instance.

    Returns:
        Tuple of (medvqa_model, text_preprocessor).
    """
    logger.info("Setting up model components...")

    # 1. Vision encoder
    logger.info(f"Loading vision encoder: {config.model.vision_encoder_name}")
    vision_encoder = BioViLTEncoder(
        model_name=config.model.vision_encoder_name,
        fallback_model_name=config.model.vision_encoder_fallback,
        hidden_size=config.model.vision_hidden_size,
        projection_dim=config.model.projection_dim,
        freeze_encoder=config.model.freeze_vision_encoder,
    )

    # Unfreeze top k layers if specified
    if config.model.unfreeze_top_k_layers > 0:
        vision_encoder.unfreeze_top_k_layers(config.model.unfreeze_top_k_layers)

    vision_trainable = vision_encoder.get_trainable_params()
    logger.info(f"  Vision encoder trainable params: {vision_trainable:,}")

    # 2. Language model with QLoRA
    logger.info(f"Loading language model: {config.model.lm_model_name}")
    language_model = MistralQLoRA(
        model_name=config.model.lm_model_name,
        load_in_4bit=config.model.load_in_4bit,
        lora_r=config.model.lora_r,
        lora_alpha=config.model.lora_alpha,
        lora_dropout=config.model.lora_dropout,
        lora_target_modules=list(config.model.lora_target_modules),
        gradient_checkpointing=config.training.gradient_checkpointing,
    )
    language_model.print_trainable_params()

    # 3. Fusion layer
    logger.info("Building cross-attention fusion layer")
    fusion = CrossAttentionFusion(
        d_model=config.model.projection_dim,
        n_heads=config.model.fusion_num_heads,
        dropout=config.model.fusion_dropout,
        use_residual=config.model.fusion_use_residual,
    )

    fusion_trainable = sum(p.numel() for p in fusion.parameters() if p.requires_grad)
    logger.info(f"  Fusion trainable params: {fusion_trainable:,}")

    # 4. Full MedVQA model
    logger.info("Assembling MedVQA model")
    medvqa_model = MedVQAModel(
        vision_encoder=vision_encoder,
        language_model=language_model,
        fusion=fusion,
        freeze_vision=config.model.freeze_vision_encoder,
        num_beams=config.model.num_beams,
    )

    if config.training.gradient_checkpointing:
        medvqa_model.enable_gradient_checkpointing()

    medvqa_model.print_param_summary()

    return medvqa_model


def setup_training(config, medvqa_model, train_dataset, val_dataset, text_preprocessor, logger):
    """Set up the training loop.

    Args:
        config: MedVQAConfig instance.
        medvqa_model: The MedVQAModel instance.
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        text_preprocessor: TextPreprocessor for decoding.
        logger: Logger instance.

    Returns:
        MedVQATrainer instance.
    """
    logger.info("Setting up training...")

    # Loss function
    loss_fn = MedVQALoss(
        closed_ended_alpha=config.training.closed_ended_alpha,
        label_smoothing=config.training.label_smoothing,
        use_contrastive=config.training.use_contrastive_loss,
        contrastive_weight=config.training.contrastive_loss_weight,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.training.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=config.training.max_epochs,
        max_steps=config.training.max_steps,
        per_device_train_batch_size=config.training.batch_size,
        per_device_eval_batch_size=config.training.batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_ratio=config.training.warmup_ratio,
        lr_scheduler_type=config.training.lr_scheduler_type,
        logging_steps=config.training.logging_steps,
        evaluation_strategy=config.training.eval_strategy,
        eval_steps=config.training.eval_steps,
        save_strategy=config.training.save_strategy,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=config.training.fp16,
        bf16=config.training.bf16,
        gradient_checkpointing=config.training.gradient_checkpointing,
        optim=config.training.optim,
        report_to=config.training.report_to,
        run_name=config.training.run_name or config.experiment_name,
        dataloader_num_workers=config.data.num_workers,
        remove_unused_columns=False,
        dataloader_pin_memory=config.data.pin_memory,
        max_grad_norm=config.training.gradient_clipping,
        seed=config.training.seed,
        ddp_find_unused_parameters=False,
    )

    # Callbacks
    callbacks = get_training_callbacks(
        early_stopping_patience=config.training.early_stopping_patience,
        early_stopping_threshold=config.training.early_stopping_threshold,
    )

    # Trainer
    trainer = MedVQATrainer(
        model=medvqa_model,
        args=training_args,
        loss_fn=loss_fn,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=text_preprocessor,
        data_collator=collate_fn,
        callbacks=callbacks,
    )

    logger.info("Training setup complete")
    return trainer


def main():
    args = parse_args()

    # Load config
    config = MedVQAConfig.from_yaml(args.config)

    # Override config for debug mode
    if args.debug:
        config.training.max_epochs = 1
        config.training.logging_steps = 1
        config.training.eval_steps = 5
        config.training.save_steps = 10
        config.training.report_to = "none"  # Disable W&B for debug
        config.training.run_name = f"debug_{config.experiment_name}"
        print("[DEBUG MODE] 1 epoch, frequent logging")

    # Setup logging
    logger = setup_logger(
        name="train", log_file=os.path.join(config.training.output_dir, "training.log")
    )

    logger.info("=" * 60)
    logger.info("MedVQA Training Pipeline")
    logger.info("=" * 60)

    # Set seed for reproducibility
    set_seed(config.training.seed)

    # Print system info
    print_system_info()

    # Initialize W&B
    if config.training.report_to == "wandb":
        wandb.init(
            project=config.project_name,
            name=config.training.run_name or config.experiment_name,
            config=config,
        )

    # Setup data
    train_dataset, val_dataset, test_dataset, text_preprocessor = setup_data(config, logger)

    # Setup model
    medvqa_model = setup_model(config, logger)

    # Setup training
    trainer = setup_training(
        config, medvqa_model, train_dataset, val_dataset, text_preprocessor, logger
    )

    # Train
    logger.info("Starting training...")
    _ = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save final model
    logger.info("Saving final model...")
    trainer.save_model(os.path.join(config.training.output_dir, "final_model"))
    logger.info(f"Model saved to {config.training.output_dir}/final_model")

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
    logger.info(f"Test metrics: {test_metrics}")

    # Close W&B
    if config.training.report_to == "wandb":
        wandb.finish()

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
