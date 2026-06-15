# 🏥 MedVQA — Multimodal Medical Visual Question Answering

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11-blue" alt="Python 3.11">
  <img src="https://img.shields.io/badge/pytorch-2.1+-ee4c2c" alt="PyTorch 2.1+">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License">
  <img src="https://img.shields.io/badge/PRs-welcome-brightgreen" alt="PRs Welcome">
</p>

<p align="center">
  <strong>A research-grade system that answers clinical questions about medical images</strong><br>
  (X-rays, MRIs, CT scans, pathology slides) with calibrated confidence scores,<br>
  Grad-CAM attention heatmaps, and an interactive Next.js frontend.
</p>

---

## 📋 Table of Contents

- [Features Overview](#-features-overview)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Inference Modes](#-inference-modes)
- [API Reference](#-api-reference)
- [Data Pipeline](#-data-pipeline)
- [Model Architecture](#-model-architecture)
- [Training](#-training)
- [Evaluation & Metrics](#-evaluation--metrics)
- [Confidence & Uncertainty](#-confidence--uncertainty)
- [Frontend Dashboard](#-frontend-dashboard)
- [Gradio Demo](#-gradio-demo)
- [Testing](#-testing)
- [Configuration Reference](#-configuration-reference)
- [Research Context](#-research-context)
- [Tech Stack](#-tech-stack)
- [Contributing](#-contributing)
- [License](#-license)
- [Citation](#-citation)

---

## ✨ Features Overview

### 🧠 Core Intelligence

| Feature | Description |
|---------|-------------|
| **Multimodal Fusion** | Cross-attention between visual patches and question tokens for fine-grained visual-textual alignment |
| **QLoRA Fine-Tuning** | 4-bit NF4 quantization with LoRA adapters (~40M trainable params out of 7B) |
| **Calibrated Confidence** | Temperature scaling + Monte Carlo Dropout (20 samples) for reliable uncertainty quantification |
| **Grad-CAM Heatmaps** | Attention heatmaps overlaid on input images highlighting diagnostically relevant regions |
| **Uncertainty Flagging** | Flags predictions below configurable confidence threshold (default: 0.3) |
| **Vision Cache** | Per-image vision encoder output caching eliminates redundant forward passes during follow-up questions |
| **Finding Auto-Tagging** | Automatic extraction of 15+ medical finding patterns (nodule, opacity, effusion, fracture, etc.) |

### 💬 Conversational Features

| Feature | Description |
|---------|-------------|
| **Threaded Chat** | Continuous follow-up questions with full conversation history remembered |
| **Follow-up Suggestions** | AI-generated 3-4 follow-up questions after each answer |
| **Patient Context** | Optional age, sex, history, symptoms injected into every question |
| **ROI Drawing** | Drag-select a region of interest on the image to focus analysis |
| **Conversation Persistence** | Conversations survive page refreshes via localStorage |

### 🌐 Deployment Options

| Feature | Description |
|---------|-------------|
| **Local Mode** | Full MedVQAModel (vision encoder + Mistral-7B 4-bit). Requires 15GB+ VRAM GPU |
| **API Mode** | Uses cloud LLMs (OpenAI, Anthropic, Gemini, Ollama) — no GPU required |
| **REST API** | FastAPI server with CORS for cross-origin frontend access |
| **Gradio Demo** | Standalone web UI with image upload and answer visualization |
| **Next.js Frontend** | Full-featured dashboard with dark mode, keyboard shortcuts, report export |

### 📊 Evaluation & Analysis

| Feature | Description |
|---------|-------------|
| **BLEU-1/4** | N-gram overlap for open-ended answers |
| **ROUGE-L** | Longest common subsequence matching |
| **BERTScore** | Semantic similarity using BERT embeddings |
| **ECE** | Expected Calibration Error (15 bins) |
| **Brier Score** | Mean squared error between confidence and correctness |
| **OOD AUROC** | Out-of-distribution detection performance |
| **Calibration Plots** | Reliability diagrams with confidence histograms |
| **Error Analysis** | Automated worst-prediction review with Grad-CAM |

---

## 🏗️ Architecture

```
                    ┌─────────────────┐
 Medical Image ────►│  BioViL-T/CLIP  │───► Visual Patches (B, V, D)
                    └─────────────────┘              │
                                                     │
                    ┌─────────────────┐              │
 Clinical Question─►│   Tokenizer     │───► Question Embeddings (B, T, D)
                    └─────────────────┘              │
                                                     ▼
                                          ┌──────────────────────┐
                                          │ Cross-Attention      │
                                          │ Fusion (4 heads)     │
                                          │ Q: text, KV: visual  │
                                          └──────────────────────┘
                                                     │
                                                     ▼
                                          ┌──────────────────────┐
                                          │   Fused Embeddings   │
                                          │   (B, T+1, D)        │
                                          └──────────────────────┘
                                                     │
                                                     ▼
                                          ┌──────────────────────┐
                                          │ Mistral-7B (QLoRA)   │
                                          │ 4-bit NF4 quantized  │
                                          └──────────────────────┘
                                                     │
                                           ┌─────────┴──────────┐
                                           ▼                    ▼
                                     ┌──────────┐       ┌──────────────┐
                                     │  Answer  │       │  Confidence  │
                                     │  Text    │       │  Score +     │
                                     │          │       │  Entropy     │
                                     └──────────┘       └──────────────┘
                                           │
                                           ▼
                                     ┌──────────────┐
                                     │  Grad-CAM    │
                                     │  Heatmap     │
                                     └──────────────┘
```

### Two Inference Modes

The system supports two mutually exclusive inference paths:

**Local Mode** — Runs the full MedVQAModel locally:
1. Vision encoder (BioViL-T or CLIP ViT-L/14) processes the image
2. Cross-attention fusion aligns visual patches with question tokens
3. Mistral-7B (4-bit QLoRA) generates the answer
4. Monte Carlo Dropout (20 samples) estimates confidence
5. Grad-CAM produces attention heatmaps
6. Requires GPU with 15GB+ VRAM

**API Mode** — Uses a cloud LLM for the language component:
1. Vision encoder runs locally for preprocessing only
2. Image is base64-encoded and sent to the API provider
3. Cloud LLM (GPT-4o, Claude, Gemini, or local Ollama) generates the answer
4. No GPU required for the LLM
5. Grad-CAM is not available (requires the full local model)

---

## 📁 Project Structure

```
medvqa/
│
├── api/                              # REST API & Demo servers
│   ├── main.py                       # FastAPI application (endpoints: /predict, /health, /metrics)
│   ├── demo.py                       # Gradio web UI for interactive testing
│   └── schemas.py                    # Pydantic request/response models
│
├── configs/                          # YAML configuration files
│   └── default_config.yaml           # Default hyperparameters (optimized for T4 15GB)
│
├── frontend/                         # Next.js 16 dashboard
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx            # Root layout: nav, footer, theme, inline dark-mode script
│   │   │   ├── page.tsx              # Home page: feature showcase, stats, scroll animations
│   │   │   ├── diagnose/
│   │   │   │   └── page.tsx          # Diagnose page: chat, ROI, context, exports, findings board
│   │   │   └── globals.css           # Tailwind CSS v4 with custom animations, glass morphism, gradients
│   │   └── components/
│   │       ├── ThemeProvider.tsx      # Dark/light theme context with localStorage persistence
│   │       └── ThemeToggle.tsx        # Sun/moon toggle button with animated transition
│   ├── next.config.ts                # API proxy rewrites (localhost:8000)
│   ├── package.json                  # Next.js 16, React 19, Tailwind CSS v4
│   ├── tsconfig.json                 # TypeScript configuration
│   ├── postcss.config.mjs            # PostCSS with @tailwindcss/postcss
│   ├── eslint.config.mjs             # ESLint with Next.js core-web-vitals + typescript configs
│   └── public/
│       └── favicon.svg              # Custom medical cross favicon with gradient
│
├── scripts/                          # Utility scripts
│   └── download_vqa_rad.py           # Download VQA-RAD from HF, stratified 80/10/10 split, image dedup
│
├── src/                              # Python source code
│   ├── __init__.py                   # Version: 0.1.0
│   │
│   ├── data/                         # Data pipeline
│   │   ├── preprocessor.py           # MedicalImagePreprocessor (letterbox, anatomy-preserving),
│   │   │                             # TextPreprocessor (Mistral chat template, yes/no detection),
│   │   │                             # load_medical_image (PIL, DICOM, NIfTI)
│   │   ├── loader.py                 # VQARADDataset, PathVQADataset (lazy loading, stratified splits),
│   │   │                             # collate_fn (variable-length padding), create_dataloaders
│   │   └── augmentation.py           # MedicalAugmentationPipeline (conservative: 10° rotation,
│   │                                 # brightness/contrast 0.9-1.1, 85% crop, Gaussian noise)
│   │
│   ├── models/                       # Model architecture
│   │   ├── vision_encoder.py         # BioViLTEncoder (BioViL-T / CLIP fallback, projection head,
│   │   │                             # Grad-CAM hooks, progressive unfreezing)
│   │   ├── language_model.py         # MistralQLoRA (4-bit NF4, LoRA adapters on q/k/v/o_proj,
│   │   │                             # gradient checkpointing, merge_and_unload)
│   │   ├── fusion.py                 # CrossAttentionFusion (multi-head, residual, learned visual
│   │   │                             # type embedding, 4096 dim, 4 heads)
│   │   ├── medvqa_model.py           # MedVQAModel (full pipeline: vision → fusion → LM → answer,
│   │   │                             # yes/no classification head, vision caching, param summary)
│   │   └── confidence.py             # ConfidenceEstimator (re-exports MonteCarloDropout +
│   │                                 # TemperatureScaler, ConfidenceResult dataclass)
│   │
│   ├── training/                     # Training infrastructure
│   │   ├── trainer.py                # MedVQATrainer (custom HF Trainer: combined loss, VQA metrics,
│   │   │                             # sample prediction logging, W&B integration)
│   │   ├── losses.py                 # closed_ended_loss (BCE + label smoothing), open_ended_loss
│   │   │                             # (causal LM CE), contrastive_loss (CLIP-style alignment),
│   │   │                             # MedVQALoss (combined with tunable alpha)
│   │   └── callbacks.py              # EarlyStoppingCallback, LoggingCallback, get_training_callbacks
│   │
│   ├── inference/                    # Inference pipeline
│   │   ├── pipeline.py               # MedVQAPipeline (dual local/API mode, PredictionResult,
│   │   │                             # vision caching, batch prediction)
│   │   ├── api_llm.py                # APILLMClient + APILLMConfig (OpenAI, Anthropic, Gemini,
│   │   │                             # Ollama, base64 encoding, .env key resolution)
│   │   └── uncertainty.py            # MonteCarloDropout (enable dropout, N=20 samples),
│   │                                 # TemperatureScaler (L-BFGS fitting, calibration)
│   │
│   ├── evaluation/                   # Evaluation & explainability
│   │   ├── metrics.py                # compute_bleu, compute_rouge_l, compute_bertscore,
│   │   │                             # compute_ece (15 bins), compute_brier_score,
│   │   │                             # compute_ood_detection_auroc, compute_all_metrics
│   │   ├── gradcam.py                # GradCAM (forward/backward hooks, ReLU weighting, overlay,
│   │   │                             # side-by-side comparison, batch report generation)
│   │   └── evaluator.py              # Evaluator (full split eval, calibration_plot, baseline
│   │                                 # comparison, error_analysis with Grad-CAM)
│   │
│   └── utils/                        # Utilities
│       ├── config.py                 # MedVQAConfig, DataConfig, ModelConfig, APIConfig,
│       │                             # TrainingConfig, InferenceConfig (dataclass + YAML)
│       ├── logger.py                 # setup_logger (console + file, structured format)
│       └── reproducibility.py        # set_seed (all seeds), enable_determinism (cuDNN),
│                                     # print_system_info (platform, CUDA, VRAM)
│
├── tests/                            # Unit tests (49 tests total)
│   ├── test_data.py                  # MedicalImagePreprocessor, TextPreprocessor,
│   │                                 # MedicalAugmentation, CollateFn, VQARADDataset (23 tests)
│   ├── test_inference.py             # PredictionResult, GradCAM overlay, TemperatureScaler (6 tests)
│   ├── test_model.py                 # BioViLTEncoder, CrossAttentionFusion, Confidence (7 tests)
│   └── test_training_pipeline.py     # End-to-end: data → model → forward → backward →
│                                     # loss → generation → checkpointing (13 tests)
│
├── notebooks/                        # Jupyter notebooks
│   ├── 01_eda.ipynb                  # Exploratory data analysis
│   ├── 02_baseline.ipynb             # Baseline models
│   ├── 03_training.ipynb             # Training walkthrough
│   └── 04_analysis.ipynb             # Results analysis
│
├── .env.example                      # API key template (OpenAI, Anthropic, Gemini, Ollama)
├── .gitignore                        # Python, PyTorch, Jupyter, IDE, data ignores
├── .pre-commit-config.yaml           # ruff (linter + formatter), mypy, pre-commit hooks
├── pyproject.toml                    # Build config, ruff rules, mypy config, pytest settings
├── requirements.txt                  # Core ML: torch, transformers, peft, bitsandbytes, etc.
├── setup.py                          # Editable pip install (pip install -e .)
├── train.py                          # Main training script with CLI args (--config, --debug)
└── LICENSE                           # MIT License
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- CUDA 12.1+ (for local mode; optional for API mode)
- Node.js 20+ (for frontend)

### Backend Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/medvqa.git
cd medvqa

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
# Or for an editable development install:
# pip install -e .

# Download VQA-RAD dataset (optional, for training)
python scripts/download_vqa_rad.py

# Set up API keys (for API mode)
cp .env.example .env
# Edit .env with your API keys
```

### Run the Gradio Demo (Local Mode)

```bash
python api/demo.py
# Opens at http://localhost:7860
```

### Run the Gradio Demo (API Mode — No GPU Needed)

```bash
python api/demo.py --mode api
# Uses OpenAI/Anthropic/Gemini from your .env config
```

### Run the FastAPI Server

```bash
# Local mode
uvicorn api.main:app --host 0.0.0.0 --port 8000

# API mode
python api/main.py --mode api
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
# Opens at http://localhost:3000
```

The frontend proxies API calls to `http://localhost:8000` via Next.js rewrites (configured in `next.config.ts`).

---

## 🔄 Inference Modes

### Local Mode

Uses the full MedVQAModel with vision encoder + Mistral-7B (4-bit). Requires GPU.

```
                        ┌─────────────────────────────────────┐
                        │         MedVQAPipeline              │
                        │                                     │
Image ──► Preprocess ──► Vision Encoder ──► Cross-Attn Fusion │
                        │                                     │
Question ─► Tokenize ──► LLM Embeddings ────┘                │
                        │                                     │
                        │  ┌─► MC Dropout (20x) ──► Confidence│
                        │  └─► Generate ──► Answer            │
                        │                                     │
                        │  Grad-CAM ──► Heatmap Overlay       │
                        └─────────────────────────────────────┘
```

### API Mode

Uses a cloud LLM (no GPU needed). The vision encoder runs locally only for preprocessing.

```
                        ┌─────────────────────────────────────┐
                        │         MedVQAPipeline              │
                        │                                     │
Image ──► Preprocess ──►│                                     │
                        │  base64 encode ──► Cloud LLM API    │
Question ─► Format ─────┘   (GPT-4o / Claude / Gemini / Ollama)
                        │                                     │
                        │  └─► Answer Text                    │
                        │                                     │
                        │  Grad-CAM: Skipped (no local model) │
                        └─────────────────────────────────────┘
```

---

## 📡 API Reference

### `POST /predict`
Predict answer for a medical image and clinical question.

**Request** (multipart/form-data):
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | file | ✓ | Medical image (JPEG, PNG, DICOM, NIfTI) |
| `question` | string | ✓ | Clinical question |
| `conversation` | string | ✗ | JSON array of `{"question", "answer"}` pairs for context |
| `patient_context` | string | ✗ | JSON with `{"age", "sex", "history", "symptoms"}` |
| `roi` | string | ✗ | JSON `{"x", "y", "w", "h"}` normalized 0–1 |

**Response:**
```json
{
  "answer": "No evidence of lung nodule. The lung fields are clear bilaterally.",
  "confidence": 0.92,
  "uncertainty_flag": false,
  "heatmap_path": "experiments/predictions/heatmap_upload.png",
  "latency_ms": 1245.3,
  "predictive_entropy": 0.34,
  "follow_up_questions": [
    "Are there any signs of infection?",
    "Is the cardiac silhouette normal?",
    "Describe the appearance of the mediastinum."
  ]
}
```

### `POST /suggest-questions`
Generate 4-6 suggested clinical questions for an uploaded image.

**Request** (multipart/form-data):
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | file | ✓ | Medical image |

**Response:**
```json
{
  "questions": [
    "What abnormalities are visible in this image?",
    "Is the anatomy normal?",
    "Are there any signs of pathology?",
    "Describe the key findings."
  ]
}
```

### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "model": "MedVQA (API mode)",
  "vram_gb": null,
  "device": "cpu"
}
```

### `GET /metrics`
Get cached evaluation metrics from `experiments/metrics.json`.

### `GET /heatmap/{image_name}`
Serve a saved Grad-CAM heatmap image.

### `GET /predict/?image_path=...&question=...`
Predict with a file path (for testing, no multipart upload).

---

## 📊 Data Pipeline

### MedicalImagePreprocessor
- **Letterbox resize**: Preserves aspect ratio — critical for medical images to avoid anatomical distortion
- **No random flips**: Medical images have defined anatomical orientation (left vs. right)
- **Conservative normalization**: ImageNet stats (compatible with BioViL-T initialization)
- **DICOM support**: Rescale slope/intercept, windowing via pydicom
- **NIfTI support**: Multi-slice volume handling, middle-slice extraction via nibabel

### TextPreprocessor
- **Mistral chat template**: Uses `[INST]`/`[/INST]` tokens for instruction formatting
- **Label masking**: Question tokens masked with -100 in labels (loss computed only on answer tokens)
- **Yes/no detection**: Heuristic detection of binary questions via starting patterns ("Is there", "Does the", etc.)
- **Left padding**: For generation, uses right padding by default

### MedicalAugmentationPipeline
Conservative augmentations safe for medical imaging:
| Augmentation | Range | Rationale |
|--------------|-------|-----------|
| SafeRotate | ±10° | Patient positioning variation |
| Brightness | 0.9–1.1 | X-ray exposure differences |
| Contrast | 0.9–1.1 | Image acquisition variation |
| RandomResizedCrop | 85%+ retention | Field-of-view differences |
| GaussNoise | σ=0.03 | Sensor noise modeling |

### VQARADDataset
- **Lazy loading**: Images loaded on demand, not cached in memory
- **Stratified splits**: 80/10/10 preserving yes/no proportion
- **Filter modes**: `filter_type='yesno'` or `'open'` for targeted training
- **Multi-format JSON**: Handles list and dict annotation formats with fallback keys

---

## 🧠 Model Architecture

### BioViLTEncoder
- **Dual model support**: Loads BioViL-T with automatic CLIP ViT-L/14 fallback
- **Projection head**: LayerNorm → Linear(1024→4096) → GELU → Dropout
- **Grad-CAM hooks**: Forward and backward hooks on the last transformer layer
- **Progressive unfreezing**: Unfreeze top K layers for domain adaptation
- **Output**: `cls_embedding` (B, 4096), `patch_embeddings` (B, 257, 4096), `last_hidden_state`

### MistralQLoRA
- **4-bit NF4 quantization**: Double quantization for memory efficiency (~8GB for 7B model)
- **LoRA adapters**: r=16, α=32 on q_proj, k_proj, v_proj, o_proj (~40M trainable params)
- **Gradient checkpointing**: Enables fitting on 15GB VRAM
- **inputs_embeds support**: Accepts fused visual+text embeddings for multimodal generation
- **merge_and_unload**: Merge LoRA weights for deployment

### CrossAttentionFusion
- **Design**: Each question token attends to all visual patches (Q=text, K/V=visual)
- **4 heads**, d_model=4096, learned visual type embedding
- **Residual connection**: Fused = Attn(text, visual) + text for training stability
- **Output**: Fused text embeddings with visual context injected (B, T, 4096)

### MedVQAModel
The complete end-to-end model:
1. Vision encoder → patch embeddings (B, V, D) + CLS (B, D)
2. LLM embedding layer → question embeddings (B, T, D)
3. Cross-attention fusion → fused embeddings (B, T, D)
4. Prepend visual CLS token → (B, T+1, D)
5. LLM forward → logits (B, seq, vocab)
6. Yes/no classification head → (B, 2) logits

---

## 🎯 Training

```bash
# Full training
python train.py --config configs/default_config.yaml

# Debug mode (1 epoch, frequent logging, no W&B)
python train.py --config configs/default_config.yaml --debug

# Resume from checkpoint
python train.py --config configs/default_config.yaml --resume_from_checkpoint experiments/medvqa_run/checkpoint-1000
```

### Training Configurations

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Quantization** | NF4 4-bit + double quant | Memory-efficient LLM loading |
| **LoRA rank** | 16 | Rank of low-rank adapters |
| **LoRA alpha** | 32 | LoRA scaling factor |
| **LoRA dropout** | 0.05 | Dropout on LoRA layers |
| **LoRA targets** | q_proj, k_proj, v_proj, o_proj | All attention projections |
| **Learning rate** | 2e-4 | Standard QLoRA learning rate |
| **Effective batch size** | 16 | 4 per GPU × 4 gradient accumulation |
| **Precision** | bfloat16 | Mixed precision training |
| **Optimizer** | Paged AdamW 8-bit | Memory-efficient optimizer |
| **Warmup** | 5% of steps | Cosine decay schedule |
| **Gradient clipping** | 1.0 | Prevents gradient explosion |
| **Label smoothing** | 0.1 | Improves confidence calibration |
| **Closed-ended α** | 0.5 | Weight for yes/no BCE loss |
| **Max epochs** | 10 | With early stopping (patience=3) |
| **GPU** | T4 15GB | Single GPU training |
| **Duration** | 4–6 hours | Full training run |

### Loss Functions

The **MedVQALoss** combines three components:

1. **Closed-ended loss** (BCE + label smoothing): Applied to yes/no questions using the model's classification head. Label smoothing (0.1) prevents overconfidence.

2. **Open-ended loss** (causal LM cross-entropy): Standard next-token prediction loss on answer tokens. Question tokens are masked with -100 in the labels.

3. **Contrastive loss** (optional, CLIP-style): Aligns visual CLS embeddings with answer text embeddings in a shared space. Enabled via `use_contrastive_loss: true` in config.

### Monitoring

Training logs to Weights & Biases by default:
- Loss components (total, closed, open, contrastive)
- Learning rate schedule
- Sample predictions with images every 3 evaluations
- System metrics (GPU memory, throughput)

### Callbacks

| Callback | Description |
|----------|-------------|
| **EarlyStopping** | Stops training if eval_loss doesn't improve for 3 evaluations |
| **LoggingCallback** | Formatted console output of loss and learning rate |

---

## 📈 Evaluation & Metrics

### Metrics Suite

| Metric | Type | Description |
|--------|------|-------------|
| **Accuracy** | Closed-ended | Yes/no binary classification accuracy |
| **BLEU-1/4** | Open-ended | N-gram overlap (smoothed) |
| **ROUGE-L** | Open-ended | Longest common subsequence F1 |
| **BERTScore** | Open-ended | Semantic similarity via BERT |
| **ECE** | Calibration | Expected Calibration Error (15 bins) |
| **MCE** | Calibration | Maximum Calibration Error |
| **Brier Score** | Calibration | MSE of confidence vs. correctness |
| **OOD AUROC** | Calibration | Out-of-distribution detection |

### Evaluator

The `Evaluator` class orchestrates complete model evaluation:

```python
from src.evaluation.evaluator import Evaluator

evaluator = Evaluator(model, text_preprocessor, gradcam, device)

# Run evaluation
results = evaluator.evaluate_split(dataloader, split_name="test", use_mc_dropout=True)

# Generate calibration plot
evaluator.calibration_plot(results["confidences"], correctness, "experiments/calib.png")

# Compare against baselines
evaluator.compare_with_baselines(results, {"random": random_results, "clip": clip_results})

# Analyze errors with Grad-CAM
evaluator.error_analysis(results, n_worst=50)
```

### Grad-CAM Explainability

Grad-CAM highlights diagnostically relevant regions in medical images:

1. **Forward pass**: Image passes through vision encoder; activations captured at last transformer block
2. **Backward pass**: Gradients of answer logit flow back to the feature maps
3. **Weighting**: Global average pooling of gradients → importance weights for each feature map
4. **ReLU activation**: Only positive contributions (features that *increase* answer score)
5. **Overlay**: Heatmap upsampled to 224×224, overlaid with α=0.4 transparency

The vision encoder supports:
- **Grad-CAM hook registration**: `vision_encoder.register_gradcam_hooks()`
- **Batch report generation**: `gradcam.generate_gradcam_report(dataset, n_samples=20)`
- **Side-by-side comparison**: Original | Heatmap | Overlay

---

## 🔒 Confidence & Uncertainty

### Monte Carlo Dropout

Runs N=20 forward passes with dropout layers active (stochastic inference):

```
For each of 20 samples:
  1. Enable all dropout layers
  2. Forward pass → logits
  3. Softmax → probabilities

Compute:
  - Mean prediction:       ŷ = (1/N) Σ p(y|x, ω_n)
  - Predictive entropy:    H(ŷ) = -Σ ŷ_c log ŷ_c
  - Mutual information:    I = H(ŷ) - (1/N) Σ H(p(y|x, ω_n))
  - Confidence:            max_c ŷ_c
```

### Temperature Scaling

Post-hoc calibration learned on the validation set:

```python
from src.inference.uncertainty import TemperatureScaler

scaler = TemperatureScaler()
scaler.fit(val_logits, val_labels)
# T = 1.5 → more uniform predictions (less confident)
# T = 0.8 → more peaked predictions (more confident)
```

The temperature parameter is optimized via L-BFGS to minimize negative log-likelihood on the validation set. Applied at inference: `p = softmax(logits / T)`.

### Uncertainty Flagging

Predictions with confidence below the threshold (default: 0.3) are flagged:
```json
{
  "answer": "Possibly a nodule in the upper left lobe...",
  "confidence": 0.22,
  "uncertainty_flag": true
}
```

---

## 🎨 Frontend Dashboard

The Next.js 16 frontend (`frontend/`) provides a full-featured clinical analysis dashboard.

### Home Page (`/`)
- **Hero section**: Animated gradient text, keyboard shortcut badges, CTA buttons
- **Stats bar**: 7B parameters, 15 finding patterns, 4-bit quantization
- **Core features grid**: 4 cards with hover animations (Conversational Workflow, Confidence Estimation, Grad-CAM Heatmaps, Finding Auto-Tagging)
- **UX features grid**: 6 compact cards (Smart Follow-ups, Patient Context, ROI Drawing, Report Generation, Keyboard Shortcuts, Copy & Persist)
- **How it works**: 3-step vertical timeline with staggered scroll animations
- **Feature showcases**: Callout cards with statistics
- **CTA section**: Glowing button with gradient background
- **Scroll animations**: Intersection Observer-based fade-in-up effects on all sections

### Diagnose Page (`/diagnose`)
- **Upload zone**: Drag-and-drop with visual feedback, file browser fallback
- **Image banner**: Thumbnail with filename/size, action buttons (New Question, Context, ROI, Change)
- **Patient Context panel**: Age input, Sex dropdown, History text, Symptoms text with clear button
- **ROI Canvas**: Interactive canvas overlay with mouse drag selection, coordinate normalization 0-1, clear button
- **Suggested questions**: Auto-generated on upload, displayed as clickable chips with loading spinner
- **Message thread**:
  - User messages: Blue rounded bubbles aligned right
  - Assistant messages: White cards with confidence rings (SVG circles with color coding)
  - Confidence labels: High (≥0.8) / Moderate (≥0.6) / Low (≥0.4) / Very low (<0.4)
  - Uncertainty badges: Amber warning for low-confidence answers
  - Finding tags: Extracted medical findings displayed as colored tags below each answer
  - Confidence bar: Animated gradient bar below each answer card
  - Follow-up suggestions: Clickable chips with hover effects
  - Copy button: One-click answer copy to clipboard
  - Latency display: Inference time in ms/s
  - Grad-CAM display: Heatmap images with icons
- **Findings board**: Accumulated findings across all messages displayed as colored tags
- **Input area**: Textarea with auto-resize, Enter to send, Shift+Enter for newline
- **Export**: Full conversation as Markdown
- **Report generation**: Structured clinical report (Findings + Detailed Q&A + Disclaimer)
- **Toast notifications**: Animated success/info toasts
- **Dark mode**: Full theme support with smooth transitions

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+K` | Open file upload dialog |
| `Ctrl+L` | Clear conversation |
| `Esc` | Focus upload button (no image) |
| `Enter` | Send message |
| `Shift+Enter` | New line in textarea |

### Found Pattern Detection (15 patterns)

```
Nodule, Opacity, Infiltrate, Effusion, Consolidation,
Fracture, Edema, Pneumothorax, Atelectasis, Cardiomegaly,
Calcification, Mass, Cyst, Fibrosis, Emphysema
```

---

## 🎛️ Gradio Demo

The Gradio demo (`api/demo.py`) provides a standalone web UI:

```bash
# Local mode
python api/demo.py

# API mode (no GPU needed)
python api/demo.py --mode api
```

Features:
- Image upload with preview
- Clinical question text input
- Example question buttons (5 pre-configured VQA-RAD samples)
- Answer display with confidence slider
- Grad-CAM heatmap visualization
- Status bar with latency and entropy
- Responsive layout (side-by-side columns)

---

## 🧪 Testing

The test suite contains **49 tests** across 4 modules:

```bash
# Run all tests
pytest tests/ -v

# Run specific modules
pytest tests/test_data.py -v              # 23 tests: data pipeline
pytest tests/test_model.py -v             # 7 tests: model architecture
pytest tests/test_inference.py -v         # 6 tests: inference & uncertainty
pytest tests/test_training_pipeline.py -v # 13 tests: end-to-end training

# Run with live logging
pytest tests/ -v -s
```

### Test Coverage

| Module | Tests | What's Tested |
|--------|-------|---------------|
| `test_data.py` | 23 | Image preprocessing, text tokenization, augmentation, collation, VQARADDataset (lazy loading, shapes, dtypes, keys, masking, filtering, edge cases) |
| `test_inference.py` | 6 | PredictionResult dataclass, GradCAM overlay/transparency, TemperatureScaler forward/calibrate |
| `test_model.py` | 7 | Vision encoder forward shape, freezing, cross-attention forward/mask/gradient flow, MC Dropout, temperature scaling |
| `test_training_pipeline.py` | 13 | Data pipeline, forward pass, loss components, backward pass (gradient flow), generation, training step, checkpoint save/load |

Tests that require the VQA-RAD dataset (`TestVQARADDataset` class in `test_data.py`) are automatically skipped if the data hasn't been downloaded.

---

## ⚙️ Configuration Reference

The system uses a hierarchical dataclass configuration system loaded from YAML:

```yaml
# configs/default_config.yaml

project_name: "medvqa"
experiment_name: "medvqa_biovil_mistral_qlora"

data:
  dataset_name: "vqa-rad"           # vqa-rad or pathvqa
  image_size: 224                    # Input image size
  max_question_length: 128           # Question truncation
  max_answer_length: 64              # Answer truncation
  batch_size: 4                      # Per GPU batch size
  num_workers: 2                     # DataLoader workers
  use_augmentation: true
  rotation_degrees: 10.0
  brightness_range: [0.9, 1.1]
  contrast_range: [0.9, 1.1]
  min_crop_retention: 0.85
  gaussian_noise_sigma: 0.03

model:
  vision_encoder_name: "openai/clip-vit-large-patch14"
  vision_hidden_size: 1024
  projection_dim: 4096
  freeze_vision_encoder: true
  unfreeze_top_k_layers: 4
  lm_model_name: "mistralai/Mistral-7B-Instruct-v0.3"
  load_in_4bit: true
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  lora_target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
  fusion_num_heads: 4
  fusion_use_residual: true
  num_beams: 4

api:
  provider: "openai"                 # openai | anthropic | gemini | ollama
  model: "gpt-4o"
  temperature: 0.7
  max_tokens: 256
  ollama_base_url: "http://localhost:11434"

training:
  learning_rate: 2.0e-4
  weight_decay: 0.01
  gradient_accumulation_steps: 4
  max_epochs: 10
  warmup_ratio: 0.05
  lr_scheduler_type: "cosine"
  label_smoothing: 0.1
  gradient_clipping: 1.0
  bf16: true
  optim: "paged_adamw_8bit"
  closed_ended_alpha: 0.5
  use_contrastive_loss: false
  seed: 42
  output_dir: "experiments/medvqa_run"
```

---

## 📚 Research Context

This project extends two prior research directions into the multimodal domain:

- **[CURA](https://github.com/...)**: A RAG-based medical QA system for text-only question answering. MedVQA extends this into the multimodal domain by adding vision-language fusion with BioViL-T and cross-attention mechanisms.

- **Self-Diagnosing Neural Models**: A framework for calibrated confidence estimation in neural networks. MedVQA ports the temperature scaling and Monte Carlo Dropout techniques from text QA to multimodal medical VQA.

### Related Work

| Work | Contribution | Relation to MedVQA |
|------|--------------|-------------------|
| **BioViL-T** (Bannur et al., 2023) | Medical vision transformer pretrained on chest X-rays | Primary vision encoder |
| **QLoRA** (Dettmers et al., 2023) | 4-bit quantized LLM fine-tuning | Memory-efficient LM adaptation |
| **Grad-CAM** (Selvaraju et al., 2017) | Gradient-weighted class activation mapping | Visual explanations |
| **Med-Flamingo** | Medical adaptation of Flamingo architecture | Alternative approach |
| **LLaVA-Med** | Biomedical instruction-tuned LLaVA | Alternative approach |

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Vision Encoder** | [BioViL-T](https://huggingface.co/microsoft/BioViL-T) / [CLIP ViT-L/14](https://huggingface.co/openai/clip-vit-large-patch14) | Medical/General image feature extraction |
| **Language Model** | [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) | Answer generation |
| **Parameter-Efficient FT** | [PEFT](https://huggingface.co/docs/peft/) + [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) | QLoRA 4-bit quantized fine-tuning |
| **Deep Learning** | [PyTorch 2.1+](https://pytorch.org/) + [Transformers](https://huggingface.co/docs/transformers/) | Core ML framework |
| **Backend API** | [FastAPI](https://fastapi.tiangolo.com/) | REST API server |
| **Frontend** | [Next.js 16](https://nextjs.org/) + [React 19](https://react.dev/) | Web dashboard |
| **Styling** | [Tailwind CSS v4](https://tailwindcss.com/) | Utility-first CSS |
| **Demo UI** | [Gradio](https://www.gradio.app/) | Standalone web interface |
| **Evaluation** | [NLTK](https://www.nltk.org/) + [rouge-score](https://github.com/google-research/google-research/tree/master/rouge) + [BERTScore](https://github.com/Tiiiger/bert_score) | BLEU, ROUGE-L, BERTScore |
| **Data** | [Datasets](https://huggingface.co/docs/datasets/) | Dataset loading |
| **Tracking** | [Weights & Biases](https://wandb.ai/) | Experiment logging |
| **Medical Image I/O** | [pydicom](https://pydicom.github.io/) + [nibabel](https://nipy.org/nibabel/) | DICOM/NIfTI support |
| **Augmentation** | [Albumentations](https://albumentations.ai/) | Medical image augmentation |
| **Config** | [PyYAML](https://pyyaml.org/) | YAML configuration |
| **Linting** | [Ruff](https://docs.astral.sh/ruff/) + [pre-commit](https://pre-commit.com/) | Code quality |

---

## 🤝 Contributing

Contributions are welcome! Please check the [open issues](https://github.com/yourusername/medvqa/issues) for ideas.

### Before Submitting a PR

1. **Lint**: `pre-commit run --all-files` (ruff checks + formatting)
2. **Type check**: `mypy src/` (optional, configured for manual runs)
3. **Test**: `pytest tests/ -v` (all 49 tests must pass)
4. **Document**: Add docstrings and type hints for new functions

### Development Setup

```bash
# Install pre-commit hooks
pre-commit install

# Run linter
ruff check src/ tests/
ruff format src/ tests/
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 📖 Citation

```bibtex
@software{medvqa2025,
  title = {MedVQA: Multimodal Medical Visual Question Answering},
  author = {MedVQA Research},
  year = {2025},
  url = {https://github.com/yourusername/medvqa}
}
```

---

<p align="center">
  <strong>⚠️ Research prototype — Not for clinical use.</strong>
</p>
