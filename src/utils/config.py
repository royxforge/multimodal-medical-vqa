"""Configuration dataclasses for the MedVQA system.

All hyperparameters are centralized here with clear documentation.
This mirrors the approach used in the CURA project where a single config
object controlled the entire pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""

    # Paths
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    augmented_dir: str = "data/augmented"

    # Dataset
    dataset_name: str = "vqa-rad"  # "vqa-rad" or "pathvqa"
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"

    # Image preprocessing
    image_size: int = 224  # BioViL-T uses 224x224; CLIP ViT-L/14 uses 336x336
    image_mean: tuple = (0.485, 0.456, 0.406)
    image_std: tuple = (0.229, 0.224, 0.225)

    # Text preprocessing
    max_question_length: int = 128
    max_answer_length: int = 64
    tokenizer_name: str = "mistralai/Mistral-7B-Instruct-v0.3"

    # Augmentation
    use_augmentation: bool = True
    rotation_degrees: float = 10.0
    brightness_range: tuple = (0.9, 1.1)
    contrast_range: tuple = (0.9, 1.1)
    min_crop_retention: float = 0.85
    gaussian_noise_sigma: float = 0.03

    # Dataloader
    batch_size: int = 4
    num_workers: int = 2
    prefetch_factor: int = 2
    pin_memory: bool = True

    def __post_init__(self):
        """Normalize types after init (handles YAML loading lists into tuple fields)."""
        if isinstance(self.brightness_range, list):
            self.brightness_range = tuple(self.brightness_range)
        if isinstance(self.contrast_range, list):
            self.contrast_range = tuple(self.contrast_range)


@dataclass
class ModelConfig:
    """Configuration for model architecture."""

    # Vision encoder
    vision_encoder_name: str = "openai/clip-vit-large-patch14"
    vision_encoder_fallback: str = "openai/clip-vit-large-patch14"
    vision_hidden_size: int = 1024
    projection_dim: int = 4096
    freeze_vision_encoder: bool = True
    unfreeze_top_k_layers: int = 4

    # Language model
    lm_model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"
    lm_hidden_size: int = 4096
    lm_max_length: int = 512
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True

    # QLoRA
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple = ("q_proj", "v_proj", "k_proj", "o_proj")

    def __post_init__(self):
        """Normalize types after init (handles YAML loading lists into tuple fields)."""
        if isinstance(self.lora_target_modules, list):
            self.lora_target_modules = tuple(self.lora_target_modules)

    # Fusion
    fusion_num_heads: int = 4
    fusion_dropout: float = 0.1
    fusion_use_residual: bool = True

    # Generation
    num_beams: int = 4
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1


@dataclass
class APIConfig:
    """Configuration for API-based LLM inference.

    When enabled, the API client is used instead of the local Mistral-7B model.
    This allows the system to run without a GPU by sending images + questions
    to a cloud LLM provider.

    Supported providers:
      - openai   (requires OPENAI_API_KEY)
      - anthropic (requires ANTHROPIC_API_KEY)
      - gemini   (requires GEMINI_API_KEY)
      - ollama   (no key needed, runs locally)
    """

    enabled: bool = False
    provider: str = "openai"
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 256
    api_key: Optional[str] = None
    ollama_base_url: str = "http://localhost:11434"


@dataclass
class TrainingConfig:
    """Configuration for training."""

    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_epochs: int = 10
    max_steps: int = -1
    label_smoothing: float = 0.1
    gradient_clipping: float = 1.0
    dropout: float = 0.1
    fp16: bool = False
    bf16: bool = True
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 3
    eval_strategy: str = "steps"
    eval_steps: int = 500
    logging_steps: int = 50
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.01
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_8bit"
    closed_ended_alpha: float = 0.5
    use_contrastive_loss: bool = False
    contrastive_loss_weight: float = 0.1
    report_to: str = "wandb"
    run_name: Optional[str] = None
    seed: int = 42
    output_dir: str = "experiments/medvqa_run"


@dataclass
class InferenceConfig:
    """Configuration for inference."""

    mode: str = "local"  # "local" or "api"
    checkpoint_path: str = "experiments/medvqa_run/best_model"
    device: str = "cuda"
    use_mc_dropout: bool = True
    mc_dropout_samples: int = 20
    uncertainty_threshold: float = 0.3
    temperature_scaling: bool = True
    temperature_scaler_path: Optional[str] = None
    cache_vision_outputs: bool = True
    max_new_tokens: int = 64


@dataclass
class MedVQAConfig:
    """Top-level configuration combining all sub-configs."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    api: APIConfig = field(default_factory=APIConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    project_name: str = "medvqa"
    experiment_name: str = "medvqa_biovil_mistral_qlora"

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "MedVQAConfig":
        """Load configuration from a YAML file."""
        import yaml

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        config = cls()
        if "data" in data:
            config.data = DataConfig(**data["data"])
        if "model" in data:
            config.model = ModelConfig(**data["model"])
        if "api" in data:
            config.api = APIConfig(**data["api"])
        if "training" in data:
            config.training = TrainingConfig(**data["training"])
        if "inference" in data:
            config.inference = InferenceConfig(**data["inference"])
        if "project_name" in data:
            config.project_name = data["project_name"]
        if "experiment_name" in data:
            config.experiment_name = data["experiment_name"]

        return config
