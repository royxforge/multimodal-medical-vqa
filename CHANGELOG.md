# Changelog

All notable changes to the MedVQA (Multimodal Medical Visual Question Answering) project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

- GPT-4o baseline benchmarking script evaluating 180 VQA-RAD samples with structured output parsing, cost tracking (approximately $1.50 for the full evaluation), calibration visualization, and zero-shot accuracy reporting. The evaluation revealed a 56.36 percent yes/no accuracy with a 41.5 percent abstention rate, establishing the zero-shot performance ceiling that the fine-tuned model must surpass.
- BERTScore integration for open-ended answer evaluation, providing semantic similarity measurement that remains reliable even when reference answers and model predictions differ substantially in length. This addressed a critical metric gap: conventional BLEU and ROUGE scores penalize GPT-4o's paragraph-length outputs against VQA-RAD's 1 to 5 word reference answers, whereas BERTScore captured the 65 percent semantic alignment that those surface-form metrics missed.
- Automated calibration plotting and worst-case error analysis within the evaluator module.

### Changed

- **README.md**: Substantially revised on 2026-07-14 with updated GPT-4o evaluation figures including the 56.36 percent accuracy finding, expanded architectural diagrams, refined installation instructions, and additional configuration reference documentation. Further updated alongside the evaluate script addition to document the GPT-4o baseline methodology and results.
- **scripts/evaluate_api.py**: Introduced to support structured API-based evaluation of cloud LLMs on VQA-RAD, encompassing prompt construction, response parsing, metric computation, and cost logging.

---

## [0.2.0] - 2026-07-20

### Added

- **Community health files**: Added `CODE_OF_CONDUCT.md` (Contributor Covenant v2.1), `CONTRIBUTING.md` (contribution guidelines), `SECURITY.md` (vulnerability reporting policy), and `CITATION.cff` (citation metadata). These files establish project governance, community participation guidelines, and academic attribution framework.

---

## [0.1.0]

### Fixed

- Merge conflict resolution between two independent initial commits. The repository history shows two initialization paths with distinct author email addresses (royxlead@proton.me and royxforge@proton.me); one path contributed the LICENSE file while the other contributed the full source tree. These were reconciled in a merge commit. This resolution is inferred from the available commit ancestry; the merge message itself did not specify the conflict details. - 2026-06-15

### Added

- Initial project scaffolding with pyproject.toml, setup.py, and requirements.txt pinned to Python 3.11 and CUDA 12.1. Python tooling configuration includes Black for deterministic code formatting at 100-character line length, Ruff for fast linting with pycodestyle, pyflakes, isort, pep8-naming, pyupgrade, and flake8-simplify rule sets, mypy for static type checking with relaxed strictness, and pytest with deprecation warning filtering and custom test markers.
- Pre-commit hooks configuration for automated code quality enforcement.
- Environment configuration via .env.example with API key placeholders for OpenAI, Anthropic, Gemini, and optional Hugging Face tokens.
- MIT license file.

#### Data Pipeline

- MedicalImagePreprocessor with letterbox resize preserving aspect ratio, DICOM support with rescale slope and intercept handling via pydicom, NIfTI middle-slice extraction from volumetric data via nibabel, and configurable normalization parameters.
- TextPreprocessor wrapping the Mistral-7B tokenizer with question truncation at 128 tokens and answer generation limits.
- VQARADDataset with lazy image loading, stratified 80/10/10 train/validation/test splits preserving yes/no label proportion, and filter modes for targeted training on closed-ended or open-ended questions (`filter_type` parameter supporting `'yesno'` or `'open'`).
- Conservative medical augmentation pipeline via albumentations: safe rotation within plus or minus 10 degrees, brightness jitter from 0.9 to 1.1, contrast jitter from 0.9 to 1.1, random resized crop with 85 percent minimum retention, and Gaussian noise at sigma 0.03.

#### Models

- BioViL-T vision encoder with automatic CLIP ViT-L/14 fallback. Includes a projection head comprising LayerNorm, Linear(1024 to 4096), GELU activation, and Dropout. Progressive unfreezing of the top K transformer layers for domain adaptation. Grad-CAM hooks registered on the final transformer layer.
- CrossAttentionFusion layer: each question token attends to all visual patches through four attention heads with a learned visual type embedding and a residual connection for training stability.
- MistralQLoRA class wrapping Mistral-7B-Instruct-v0.3 with 4-bit NF4 quantization, double quantization (saving approximately 0.4 GB), and LoRA adapters (rank 16, alpha 32, dropout 0.05) on q_proj, k_proj, v_proj, and o_proj projections. Gradient checkpointing enabled for 15 GB VRAM compatibility on T4 GPUs.
- MedVQAModel orchestrating vision encoding, cross-attention fusion, language model generation, and a closed-ended yes/no classification head. Supports vision output caching to avoid redundant encoding during training.
- ConfidenceEstimator with Monte Carlo Dropout (20 stochastic forward passes aggregated through predictive entropy, mutual information, and maximum softmax probability) and post-hoc TemperatureScaler via L-BFGS optimization for calibrated uncertainty quantification.

#### Training

- MedVQATrainer with Paged AdamW 8-bit optimizer, cosine learning rate schedule with 5 percent warmup ratio, gradient accumulation for effective batch size of 16, gradient clipping at norm 1.0, and early stopping with patience 3 and threshold 0.01.
- Three-component composite loss function (MedVQALoss): binary cross-entropy with label smoothing at 0.1 for yes/no classification, causal language model cross-entropy for open-ended answer generation, and an optional CLIP-style contrastive loss for visual-text embedding alignment (disabled by default, weight 0.1 when enabled).
- Logging and experiment tracking callbacks with Weights and Biases integration.
- Full training script (train.py) supporting debug mode (single epoch, frequent logging, no W&B), checkpoint resumption, and YAML configuration override.

#### Inference

- Dual inference pipeline (MedVQAPipeline): local GPU mode executing the full MedVQAModel with Grad-CAM heatmap generation, and API mode leveraging cloud providers via base64 image encoding with no GPU requirement for the language component.
- API LLM client (APILLMClient) supporting OpenAI GPT-4o, Anthropic Claude, Google Gemini, and local Ollama deployments with configurable temperature, max tokens, and provider-specific model names.

#### Evaluation

- GradCAM module with gradient-weighted heatmap generation, hooks registered on the final vision encoder layer, ReLU-weighted aggregation across patches, configurable colormap and opacity overlay, and batch report construction.
- Evaluator with comprehensive metric computation: accuracy for closed-ended classification, BLEU-1/4 and ROUGE-L F1 for open-ended response evaluation, Expected Calibration Error over 15 bins, Brier Score, and OOD AUROC for out-of-distribution detection.
- Automated calibration plot generation and worst-case error analysis within the evaluator.

#### API and Demo

- FastAPI REST API with three endpoints: POST /predict accepting multipart form data (medical image, clinical question, optional conversation history, patient context, and region of interest), GET /health returning model status and device information, and GET /metrics returning cached evaluation results.
- Gradio web demo supporting both local and API inference modes selected via a command-line flag.
- Pydantic request and response schemas with typed fields for answer text, confidence score, uncertainty flag, heatmap path, latency in milliseconds, predictive entropy, and follow-up questions.

#### Frontend

- Next.js 16 clinical dashboard with two pages: a home page for feature overview, architecture statistics, and a how-it-works timeline, and a diagnose page featuring drag-and-drop image upload, conversation threading with SVG confidence rings color-coded by confidence tier, region of interest canvas, patient context form, findings board with auto-tagging for 15 finding patterns (nodule, opacity, effusion, fracture, edema, pneumothorax, atelectasis, cardiomegaly, and others), follow-up question chips, Grad-CAM heatmap display, Markdown report export, and dark mode with persistent theme toggle.
- Keyboard shortcuts: Ctrl+K for upload, Ctrl+L for clear conversation, Enter to send, and Shift+Enter for newline.

#### Notebooks

- Four Jupyter notebooks: exploratory data analysis (01_eda.ipynb) with VQA-RAD dataset statistics, CLIP zero-shot baseline (02_baseline.ipynb), full training loop (03_training.ipynb), and ablation analysis (04_analysis.ipynb).

#### Testing

- 49 unit tests across four modules: data loading and preprocessing (23 tests), model architecture and forward pass correctness (7 tests), inference and uncertainty quantification (6 tests), and end-to-end training pipeline with mock data and synthetic gradients (13 tests). Slow and integration test markers for selective execution.

#### Utilities

- Configuration system using YAML with pydantic-validated schema (MedVQAConfig, DataConfig, ModelConfig, TrainingConfig, APIConfig, InferenceConfig) covering all model architecture, training hyperparameter, API provider, and inference settings.
- Deterministic reproducibility utilities: set_seed for Python, NumPy, PyTorch, and CUDA random number generators, cuDNN deterministic mode, and comprehensive system information logging (CUDA version, GPU count, VRAM, disk space).
- Data download script (scripts/download_vqa_rad.py) for VQA-RAD with automatic dataset registration, stratified split generation, and local caching.

---

The format of this changelog and the project's versioning approach follow the recommendations of [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and [Semantic Versioning](https://semver.org/spec/v2.0.0.html), respectively.
