# MedVQA - Multimodal Medical Visual Question Answering
### Vision-Language Fusion for Clinical Image Understanding

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Mistral-7B-6366f1?style=flat-square" />
  <img src="https://img.shields.io/badge/Dataset-VQA--RAD-14b8a6?style=flat-square" />
  <img src="https://img.shields.io/badge/Frontend-Next.js%2016-000000?style=flat-square&logo=nextdotjs&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-6366f1?style=flat-square" />
</p>

> A research-grade system that answers clinical questions about medical images (X-rays, MRIs, CT scans, pathology slides) using cross-attention vision-language fusion, QLoRA fine-tuned Mistral-7B, Monte Carlo Dropout confidence estimation, and Grad-CAM explainability. Evaluated on VQA-RAD with GPT-4o baseline benchmarking.

> **⚠️ Research prototype - not validated for clinical use. Do not use for medical decision-making.**

---

## Table of Contents

- [The Problem](#the-problem)
- [What This Does](#what-this-does)
- [Evaluation Results](#evaluation-results)
- [Research: GPT-4o on VQA-RAD](#research-gpt-4o-on-vqa-rad)
- [Architecture](#architecture)
- [Model Design](#model-design)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Inference Modes](#inference-modes)
- [Confidence and Uncertainty](#confidence-and-uncertainty)
- [Data Pipeline](#data-pipeline)
- [Training](#training)
- [API Reference](#api-reference)
- [Dashboard](#dashboard)
- [Evaluation](#evaluation)
- [Configuration Reference](#configuration-reference)
- [Related Work](#related-work)
- [Citation](#citation)

---

## The Problem

Clinical image interpretation is a high-stakes, expert-constrained bottleneck in healthcare:

- Radiologist shortages create 11-day average report turnaround times in some systems
- Visual question answering requires simultaneous image grounding and clinical reasoning
- Standard LLMs cannot localize findings spatially - they answer from text, not pixels
- Model confidence is uncalibrated by default - a wrong answer delivered at 95% confidence is clinically dangerous

Existing Medical VQA benchmarks show a consistent gap: general-purpose vision-language models score near random on radiology-specific questions requiring anatomical grounding. **Domain-adapted fusion architectures with calibrated uncertainty are essential for meaningful clinical assistance.**

---

## What This Does

MedVQA provides:

1. **Cross-attention vision-language fusion** - BioViL-T visual patches attend to question tokens for fine-grained image-text alignment, with QLoRA fine-tuned Mistral-7B (~40M trainable params) generating the answer
2. **Calibrated confidence** - Monte Carlo Dropout (20 samples) for epistemic uncertainty, post-hoc temperature scaling via L-BFGS, and automatic uncertainty flagging below configurable thresholds
3. **Grad-CAM explainability** - Gradient-weighted heatmaps overlaid on input images, localizing the image regions that drove each answer (e.g., pleural margin for pneumothorax questions, not the cardiac silhouette)
4. **Dual inference modes** - Local GPU mode (full model, Grad-CAM available) and API mode (GPT-4o / Claude / Gemini, no GPU required), with a FastAPI backend and Next.js 16 clinical dashboard

---

## Evaluation Results

### GPT-4o Baseline: Yes/No Accuracy

Evaluated on 94 yes/no questions from VQA-RAD. 180 total samples processed (0 errors, ~$1.50 API cost).

| Metric | Value |
|---|---|
| Correct / Answered | 31 / 55 |
| Accuracy (answered) | 56.36% |
| Abstained | 39 / 94 (41.5%) |
| Total yes/no questions | 94 |

GPT-4o abstained on 41% of binary questions - flagging cases where the image modality alone (e.g., chest X-ray without CT) was insufficient for confident determination. Of the questions it answered, 56% were correct.

### GPT-4o Baseline: Open-Ended Metrics

| Metric | Value | Interpretation |
|---|---|---|
| BLEU-1 | 0.0088 | Expected to be low - GPT-4o produces paragraph answers; VQA-RAD references are 1-5 words |
| BLEU-4 | 0.0003 | N-gram overlap collapses at 4-gram level across length-mismatched pairs |
| ROUGE-L F1 | 0.0583 | Low for the same length mismatch reason |
| BERTScore F1 | **0.6520** | Semantic similarity is 65% - GPT-4o captures the right medical concepts despite verbosity |
| Keyword Recall | 14.8% | Key medical terms from references recovered in predictions |

**BLEU and ROUGE are misleading here.** These metrics compare n-gram overlap between texts of similar length. GPT-4o generates clinical paragraphs; VQA-RAD reference answers are 1-5 words. BERTScore (65% F1) is the appropriate metric - it measures semantic alignment at the embedding level and confirms that GPT-4o produces conceptually correct answers even when the surface form differs.

### Comparison Against Baselines

The most striking number in this table is that GPT-4o at 56.36% falls *below* CLIP zero-shot at ~58% - a general-purpose vision-language model with a massive context window barely trailing a pure image-text similarity model on radiology questions. This confirms that VQA-RAD tests genuine visual grounding, not knowledge retrieval: even a strong multimodal LLM struggles without domain-specific spatial understanding.

| Model | Yes/No Accuracy | Notes |
|---|---|---|
| Random | 50% | Baseline |
| CLIP Zero-Shot | ~58% | From literature |
| Mistral Text-Only | ~65% | Text-only, no image input |
| GPT-4o (this evaluation) | 56.36% | 41% abstention - being medically cautious |
| MedVQA (fine-tuned local) | TBD | Requires GPU training run |

The 56% figure is below GPT-4o's known medical QA performance (~70-80% on standard benchmarks). Two factors explain this: the high abstention rate (41% of questions are skipped and excluded from the accuracy denominator) and the radiology-specific nature of VQA-RAD questions that require spatial image grounding rather than knowledge retrieval.

### Execution

| Metric | Value |
|---|---|
| Samples processed | 180 / 180 |
| Errors | 0 |
| API cost | ~$1.50 (180 GPT-4o image calls) |
| Duration | 479s (~8 min) |

---

## Research: GPT-4o on VQA-RAD

VQA-RAD is constructed from real clinical questions posed by radiologists during image review sessions - not a knowledge test, but a visual grounding test. A model cannot answer "Is the endotracheal tube correctly positioned?" from training data; it must localize the tube in the image. GPT-4o's 41% abstention rate reflects this correctly: the model declines to guess when the imaging modality is insufficient (e.g., a question requiring CT resolution from a chest X-ray). The relevant accuracy figure is 56% over the questions it *does* answer, not over the full set. That is the zero-shot ceiling a fine-tuned local model needs to beat - and it needs to beat it with calibrated confidence, since a correct answer hedged at 22% confidence has different clinical value than the same answer at 92%.

```python
from src.evaluation.evaluator import Evaluator

evaluator = Evaluator(model, text_preprocessor, gradcam, device)
results = evaluator.evaluate_split(dataloader, split_name="test", use_mc_dropout=True)
evaluator.calibration_plot(results["confidences"], results["correctness"], "experiments/calib.png")
```

---

## Architecture

```
Medical Image
      |
      v
+------------------+
|   BioViL-T       |   Visual patches (B, V, 4096)
|   Vision Encoder |   CLS embedding (B, 4096)
+--------+---------+
         |
         |          Clinical Question
         |                |
         |          +-----v------+
         |          | Tokenizer  |   Question embeddings (B, T, 4096)
         |          +-----+------+
         |                |
         +-------+--------+
                 |
                 v
      +---------------------+
      |  Cross-Attention    |   Q: text tokens, K/V: visual patches
      |  Fusion (4 heads)   |   Residual connection for training stability
      +----------+----------+
                 |
                 v
      +---------------------+
      |   Fused Embeddings  |   (B, T+1, D) - visual CLS prepended
      +----------+----------+
                 |
                 v
      +---------------------+
      |  Mistral-7B QLoRA   |   4-bit NF4, LoRA on q/k/v/o_proj
      +----------+----------+
                 |
         +-------+--------+
         v                v
   +-----------+   +------------------+
   |  Answer   |   |  Confidence      |
   |  Text     |   |  Score + Entropy |
   +-----------+   |  (MC Dropout)    |
         |         +------------------+
         v
   +-----------+
   |  Grad-CAM |
   |  Heatmap  |
   +-----------+
```

### Two Inference Paths

**Local Mode** requires a GPU with 15GB+ VRAM and runs the full pipeline: vision encoder → cross-attention fusion → Mistral-7B generation → Monte Carlo Dropout confidence → Grad-CAM heatmap.

**API Mode** encodes the image as base64 and sends it to a cloud provider (GPT-4o, Claude, Gemini, or local Ollama). No GPU required. Grad-CAM is unavailable in this path since it requires the local model's gradient flow.

---

## Model Design

### BioViL-T Vision Encoder
- Dual model support: loads BioViL-T with automatic CLIP ViT-L/14 fallback
- Projection head: LayerNorm → Linear(1024→4096) → GELU → Dropout
- Grad-CAM hooks registered on the last transformer layer
- Progressive unfreezing: top K layers for domain adaptation

### CrossAttentionFusion
- Each question token attends to all visual patches (Q=text, K/V=visual)
- 4 heads, d_model=4096, learned visual type embedding
- Residual connection: Fused = Attn(text, visual) + text

### MistralQLoRA
- 4-bit NF4 quantization with double quantization (~8GB for 7B)
- LoRA adapters: r=16, α=32 on q_proj, k_proj, v_proj, o_proj (~40M trainable params)
- Gradient checkpointing for 15GB VRAM compatibility

### Loss Functions

Three components combined via MedVQALoss:

**Closed-ended loss** (BCE + label smoothing 0.1): Applied to yes/no questions via a classification head. Label smoothing prevents overconfidence on binary predictions.

**Open-ended loss** (causal LM cross-entropy): Standard next-token prediction on answer tokens only. Question tokens are masked with -100.

**Contrastive loss** (optional, CLIP-style): Aligns visual CLS and answer text embeddings. Enabled via `use_contrastive_loss: true`.

---

## Repository Structure

```
multimodal-medical-vqa/
|
+-- api/                              # REST API and demo servers
|   +-- main.py                       # FastAPI application (/predict, /health, /metrics)
|   +-- demo.py                       # Gradio web UI for interactive testing
|   +-- schemas.py                    # Pydantic request/response models
|
+-- configs/
|   +-- default_config.yaml           # Default hyperparameters (optimized for T4 15GB)
|
+-- frontend/                         # Next.js 16 clinical dashboard
|   +-- src/app/
|   |   +-- page.tsx                  # Home page
|   |   +-- diagnose/page.tsx         # Chat, ROI, context, findings, export
|   +-- src/components/
|       +-- ThemeProvider.tsx
|       +-- ThemeToggle.tsx
|
+-- scripts/
|   +-- download_vqa_rad.py           # VQA-RAD download, stratified 80/10/10 split
|
+-- src/
|   +-- data/
|   |   +-- preprocessor.py           # MedicalImagePreprocessor, TextPreprocessor
|   |   +-- loader.py                 # VQARADDataset, PathVQADataset, create_dataloaders
|   |   +-- augmentation.py           # Conservative medical augmentation pipeline
|   |
|   +-- models/
|   |   +-- vision_encoder.py         # BioViLTEncoder (BioViL-T / CLIP, Grad-CAM hooks)
|   |   +-- language_model.py         # MistralQLoRA (4-bit NF4, LoRA adapters)
|   |   +-- fusion.py                 # CrossAttentionFusion (4 heads, residual)
|   |   +-- medvqa_model.py           # MedVQAModel (full pipeline, yes/no head, vision cache)
|   |   +-- confidence.py             # ConfidenceEstimator, MonteCarloDropout, TemperatureScaler
|   |
|   +-- training/
|   |   +-- trainer.py                # MedVQATrainer (combined loss, W&B integration)
|   |   +-- losses.py                 # closed_ended_loss, open_ended_loss, contrastive_loss
|   |   +-- callbacks.py              # EarlyStopping, LoggingCallback
|   |
|   +-- inference/
|   |   +-- pipeline.py               # MedVQAPipeline (local/API, vision caching, batch)
|   |   +-- api_llm.py                # APILLMClient (OpenAI, Anthropic, Gemini, Ollama)
|   |   +-- uncertainty.py            # MonteCarloDropout, TemperatureScaler
|   |
|   +-- evaluation/
|   |   +-- metrics.py                # BLEU, ROUGE-L, BERTScore, ECE, Brier, OOD AUROC
|   |   +-- gradcam.py                # GradCAM (hooks, ReLU weighting, overlay, batch report)
|   |   +-- evaluator.py              # Evaluator (full split eval, calibration plot, error analysis)
|   |
|   +-- utils/
|       +-- config.py                 # MedVQAConfig, DataConfig, ModelConfig, TrainingConfig
|       +-- logger.py                 # setup_logger
|       +-- reproducibility.py        # set_seed, enable_determinism, print_system_info
|
+-- tests/                            # 49 unit tests across 4 modules
+-- notebooks/                        # EDA, baseline, training, analysis
+-- train.py                          # Main training script (--config, --debug, --resume)
+-- pyproject.toml
+-- requirements.txt
+-- .env.example
```

---

## Installation

```bash
git clone https://github.com/royxforge/multimodal-medical-vqa.git
cd multimodal-medical-vqa

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
# Or editable install:
pip install -e .

# Download VQA-RAD (optional, for training and local evaluation)
python scripts/download_vqa_rad.py

# Configure API keys (for API mode)
cp .env.example .env
```

**Requirements:** Python 3.11+ · CUDA 12.1+ (local mode) · Node.js 20+ (frontend) · 15GB+ VRAM (local mode)

---

## Quick Start

**Gradio demo - local mode (GPU required):**

```bash
python api/demo.py
# Opens at http://localhost:7860
```

**Gradio demo - API mode (no GPU):**

```bash
python api/demo.py --mode api
# Uses OpenAI/Anthropic/Gemini from .env
```

**Python API:**

```python
from src.inference.pipeline import MedVQAPipeline

pipeline = MedVQAPipeline(mode="api")  # or mode="local"

result = pipeline.predict(
    image_path="chest_xray.jpg",
    question="Is there evidence of pneumothorax?"
)

print(f"Answer: {result.answer}")
print(f"Confidence: {result.confidence:.3f}")
print(f"Uncertainty flag: {result.uncertainty_flag}")
print(f"Entropy: {result.predictive_entropy:.3f}")
```

**FastAPI server:**

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

**Frontend:**

```bash
cd frontend
npm install
npm run dev
# Opens at http://localhost:3000
```

---

## Inference Modes

### Local Mode

Runs the full MedVQAModel. Requires GPU with 15GB+ VRAM.

```python
from src.inference.pipeline import MedVQAPipeline
from src.inference.uncertainty import MonteCarloDropout

pipeline = MedVQAPipeline(mode="local", checkpoint="experiments/medvqa_run/checkpoint-best")

# Monte Carlo Dropout confidence (20 samples)
with MonteCarloDropout(pipeline.model, n_samples=20) as mc:
    result = pipeline.predict(image_path, question)
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Heatmap: {result.heatmap_path}")
```

### API Mode

Uses a cloud LLM. No GPU required for the language component. Vision encoder preprocessing runs locally.

```python
pipeline = MedVQAPipeline(
    mode="api",
    api_config={"provider": "openai", "model": "gpt-4o"}
)

result = pipeline.predict(image_path, question)
# Note: result.heatmap_path is None in API mode (no gradient flow)
```

Supported providers: `openai`, `anthropic`, `gemini`, `ollama`.

---

## Confidence and Uncertainty

### Monte Carlo Dropout

20 stochastic forward passes with dropout active:

```
For each of 20 samples:
  Enable all dropout layers
  Forward pass → logits → softmax → p(y|x, ω_n)

Aggregate:
  Mean prediction:      ŷ = (1/20) Σ p(y|x, ω_n)
  Predictive entropy:   H(ŷ) = -Σ ŷ_c log ŷ_c
  Mutual information:   I = H(ŷ) - (1/20) Σ H(p(y|x, ω_n))
  Confidence:           max_c ŷ_c
```

### Temperature Scaling

Post-hoc calibration fitted on the validation set via L-BFGS:

```python
from src.inference.uncertainty import TemperatureScaler

scaler = TemperatureScaler()
scaler.fit(val_logits, val_labels)
# T > 1.0 → flattens predictions (less confident)
# T < 1.0 → sharpens predictions (more confident)

calibrated_probs = scaler(logits)
```

### Uncertainty Flagging

Predictions below the confidence threshold (default 0.3) are flagged:

```json
{
  "answer": "Possible opacity in the right lower lobe, though findings are subtle.",
  "confidence": 0.22,
  "uncertainty_flag": true,
  "predictive_entropy": 0.81
}
```

This connects directly to the confidence monitoring work in [Production Drift Detection](https://github.com/royxforge/production-drift-detection) - the same entropy and margin signals used there for population-level monitoring apply here at the individual prediction level.

---

## Data Pipeline

### MedicalImagePreprocessor
- **Letterbox resize**: Preserves aspect ratio - avoids anatomical distortion from naive squash resize
- **No random flips**: Medical images have defined orientation (cardiac apex is always left on a PA chest film)
- **DICOM support**: Rescale slope/intercept, windowing via pydicom
- **NIfTI support**: Middle-slice extraction from volumetric data via nibabel

### Medical Augmentation
Conservative augmentations that preserve diagnostic validity:

| Augmentation | Range | Rationale |
|---|---|---|
| SafeRotate | ±10° | Patient positioning variation |
| Brightness | 0.9-1.1 | X-ray exposure differences |
| Contrast | 0.9-1.1 | Image acquisition variation |
| RandomResizedCrop | 85%+ retention | Field-of-view variation |
| GaussNoise | σ=0.03 | Sensor noise |

### VQARADDataset
- Lazy loading: images loaded on demand
- Stratified 80/10/10 splits preserving yes/no proportion
- Filter modes: `filter_type='yesno'` or `'open'` for targeted training

---

## Training

```bash
# Full training run
python train.py --config configs/default_config.yaml

# Debug mode (1 epoch, frequent logging, no W&B)
python train.py --config configs/default_config.yaml --debug

# Resume from checkpoint
python train.py --config configs/default_config.yaml \
  --resume_from_checkpoint experiments/medvqa_run/checkpoint-1000
```

### Key Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| Quantization | NF4 4-bit + double quant | ~8GB for 7B model |
| LoRA rank | 16 | ~40M trainable params |
| LoRA alpha | 32 | Scaling factor |
| LoRA targets | q/k/v/o_proj | All attention projections |
| Learning rate | 2e-4 | Standard QLoRA rate |
| Effective batch size | 16 | 4 per GPU × 4 gradient accumulation |
| Optimizer | Paged AdamW 8-bit | Memory-efficient |
| Warmup | 5% of steps | Cosine decay |
| Label smoothing | 0.1 | Improves calibration |
| Closed-ended α | 0.5 | Weight of yes/no BCE loss |
| Max epochs | 10 | Early stopping patience=3 |
| Target GPU | T4 15GB | Single GPU |
| Est. duration | 4-6 hours | Full run |

Metrics, loss components, and sample predictions log to Weights & Biases by default.

---

## API Reference

### POST /predict

**Request** (multipart/form-data):

| Parameter | Type | Required | Description |
|---|---|---|---|
| `file` | file | Yes | Medical image (JPEG, PNG, DICOM, NIfTI) |
| `question` | string | Yes | Clinical question |
| `conversation` | string | No | JSON array of prior `{"question", "answer"}` pairs |
| `patient_context` | string | No | JSON with `{"age", "sex", "history", "symptoms"}` |
| `roi` | string | No | JSON `{"x", "y", "w", "h"}` normalized 0-1 |

**Response:**

```json
{
  "answer": "No evidence of pneumothorax. Lung fields are clear bilaterally.",
  "confidence": 0.92,
  "uncertainty_flag": false,
  "heatmap_path": "experiments/predictions/heatmap_upload.png",
  "latency_ms": 1245.3,
  "predictive_entropy": 0.34,
  "follow_up_questions": [
    "Are there signs of pleural effusion?",
    "Is the cardiac silhouette within normal limits?",
    "Describe the mediastinal contour."
  ]
}
```

### GET /health

```json
{ "status": "ok", "model": "MedVQA (API mode)", "device": "cpu" }
```

### GET /metrics

Returns cached evaluation metrics from `experiments/metrics.json`.

---

## Dashboard

```bash
# Start backend
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Start frontend
cd frontend && npm run dev
# Opens at http://localhost:3000
```

**Pages:**
1. Home - feature overview, architecture stats, how-it-works timeline
2. Diagnose - drag-and-drop upload, conversation thread, ROI canvas, patient context, findings board, report export

**Diagnose page features:** Confidence rings on each answer (SVG, color-coded by confidence tier), uncertainty badges, finding auto-tags (15 patterns: nodule, opacity, effusion, fracture, edema, pneumothorax, atelectasis, cardiomegaly, and more), follow-up question chips, Grad-CAM display, Markdown export, dark mode.

**Keyboard shortcuts:** `Ctrl+K` (upload), `Ctrl+L` (clear), `Enter` (send), `Shift+Enter` (newline).

---

## Evaluation

```bash
# Run GPT-4o baseline on VQA-RAD
python scripts/evaluate_gpt4o.py

# Run full model evaluation
python -c "
from src.evaluation.evaluator import Evaluator
evaluator = Evaluator(model, preprocessor, gradcam, device)
results = evaluator.evaluate_split(test_loader, split_name='test', use_mc_dropout=True)
evaluator.calibration_plot(results['confidences'], results['correctness'], 'calib.png')
evaluator.error_analysis(results, n_worst=50)
"

# Run tests
pytest tests/ -v                        # All 49 tests
pytest tests/test_data.py -v            # Data pipeline (23 tests)
pytest tests/test_model.py -v           # Model architecture (7 tests)
pytest tests/test_inference.py -v       # Inference and uncertainty (6 tests)
pytest tests/test_training_pipeline.py  # End-to-end training (13 tests)
```

### Metrics Suite

| Metric | Type | Description |
|---|---|---|
| Accuracy | Closed-ended | Yes/no binary classification |
| BLEU-1/4 | Open-ended | N-gram overlap (use cautiously on length-mismatched pairs) |
| ROUGE-L | Open-ended | Longest common subsequence F1 |
| BERTScore | Open-ended | Semantic similarity via BERT embeddings |
| ECE | Calibration | Expected Calibration Error (15 bins) |
| Brier Score | Calibration | MSE between confidence and correctness |
| OOD AUROC | Calibration | Out-of-distribution detection performance |

---

## Configuration Reference

```yaml
# configs/default_config.yaml

model:
  vision_encoder_name: "openai/clip-vit-large-patch14"  # swap for BioViL-T if available
  lm_model_name: "mistralai/Mistral-7B-Instruct-v0.3"
  load_in_4bit: true
  lora_r: 16                    # increase to 32 for more capacity, at memory cost
  lora_alpha: 32
  unfreeze_top_k_layers: 4      # vision encoder layers to unfreeze for domain adaptation
  fusion_num_heads: 4
  fusion_use_residual: true

api:
  provider: "openai"            # openai | anthropic | gemini | ollama
  model: "gpt-4o"

training:
  learning_rate: 2.0e-4
  gradient_accumulation_steps: 4
  label_smoothing: 0.1          # critical for confidence calibration on yes/no head
  closed_ended_alpha: 0.5       # weight of BCE loss vs causal LM loss
  use_contrastive_loss: false   # CLIP-style visual-text alignment, off by default
  seed: 42
```

---

## Related Work

- [Production Drift Detection](https://github.com/royxforge/production-drift-detection) - The confidence monitoring and entropy tracking in MedVQA's ConfidenceEstimator shares methodology with the production drift monitoring system. Temperature scaling and MC Dropout are directly ported from the confidence research there.
- [Loss Landscape Analysis](https://github.com/royxforge/loss-landscape-analysis) - MedVQA's closed-ended loss uses BCE with label smoothing, not MSE. The gradient saturation analysis in that work is the direct motivation for this choice on the yes/no classification head.

---

## Citation

```bibtex
@software{roy2026medvqa,
  author = {Roy, Sourav},
  title  = {MedVQA: Multimodal Medical Visual Question Answering},
  year   = {2026},
  url    = {https://github.com/royxforge/multimodal-medical-vqa}
}
```

---

<p align="center">
  <sub>Built by <a href="https://github.com/royxforge">Sourav Roy</a> · Artificial Intelligence Engineer · Accure Inc.</sub>
</p>
