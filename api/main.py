"""FastAPI application for MedVQA.

Provides REST API endpoints for inference, health checks, and metrics.
Supports two modes:
  1. Local mode (default): Uses the full MedVQAModel (vision encoder + Mistral-7B).
     Requires a GPU for Mistral-7B.
  2. API mode (--mode api): Uses a cloud LLM (OpenAI, Anthropic, Gemini, Ollama).
     No GPU needed.

Usage:
    uvicorn api.main:app --host 0.0.0.0 --port 8000              (local mode)
    python api/main.py --mode api                                 (API mode)
"""

import json
import os
import re
import sys
import time
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.schemas import (
    HealthResponse,
    MetricsResponse,
    PredictResponse,
    SuggestQuestionsResponse,
)

# ── CLI argument: inference mode ────────────────────────────────────────────
_INFERENCE_MODE = "local"  # default; overridden by --mode flag
for i, arg in enumerate(sys.argv):
    if arg == "--mode" and i + 1 < len(sys.argv):
        _INFERENCE_MODE = sys.argv[i + 1]
    elif arg.startswith("--mode="):
        _INFERENCE_MODE = arg.split("=", 1)[1]

app = FastAPI(
    title="MedVQA API",
    description="Multimodal Medical Visual Question Answering API",
    version="0.1.0",
)

# CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline (lazy initialized)
_pipeline = None
_mode = _INFERENCE_MODE  # "local" or "api"


def _normalize_extension(filename: str) -> str:
    """Normalize file extension to handle multi-suffix formats like .nii.gz."""
    lower = filename.lower()
    if lower.endswith(".nii.gz"):
        return ".nii.gz"
    return Path(filename).suffix.lower()


def get_pipeline():
    """Lazy initialize the inference pipeline.

    Returns the singleton MedVQAPipeline instance.
    The mode ("local" or "api") is detected at startup from the --mode flag.
    """
    global _pipeline

    if _pipeline is not None:
        return _pipeline

    from src.data.preprocessor import MedicalImagePreprocessor, TextPreprocessor
    from src.evaluation.gradcam import GradCAM
    from src.inference.pipeline import MedVQAPipeline
    from src.utils.config import MedVQAConfig
    from src.utils.reproducibility import set_seed

    config = MedVQAConfig.from_yaml("configs/default_config.yaml")
    set_seed(config.training.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_api = _mode == "api"

    # Preprocessors (needed in both modes)
    image_preprocessor = MedicalImagePreprocessor(image_size=config.data.image_size)
    text_preprocessor = TextPreprocessor(
        model_name=config.data.tokenizer_name,
        max_question_length=config.data.max_question_length,
        max_answer_length=config.data.max_answer_length,
    )

    if use_api:
        # ── API mode: lightweight, no local Mistral-7B ──
        print(f"[INIT] MedVQA pipeline in API mode on {device}...")

        from src.inference.api_llm import APILLMClient, APILLMConfig

        api_config = APILLMConfig(
            provider=config.api.provider,
            model=config.api.model,
            temperature=config.api.temperature,
            max_tokens=config.api.max_tokens,
            ollama_base_url=config.api.ollama_base_url,
        )
        api_client = APILLMClient(api_config)

        # Grad-CAM requires the full MedVQAModel (not available in API mode)
        # The pipeline gracefully skips heatmap generation when no gradcam is provided

        _pipeline = MedVQAPipeline(
            image_preprocessor=image_preprocessor,
            text_preprocessor=text_preprocessor,
            gradcam=None,
            device=device,
            api_llm_client=api_client,
        )

        print(f"[OK] API pipeline ready (provider={config.api.provider}, model={config.api.model})")
        return _pipeline

    # ── Local mode: full MedVQAModel (vision encoder + Mistral-7B) ──
    print(f"[INIT] MedVQA pipeline in LOCAL mode on {device}...")

    from src.models.fusion import CrossAttentionFusion
    from src.models.language_model import MistralQLoRA
    from src.models.medvqa_model import MedVQAModel
    from src.models.vision_encoder import BioViLTEncoder

    vision_encoder = BioViLTEncoder(
        model_name=config.model.vision_encoder_name,
        fallback_model_name=config.model.vision_encoder_fallback,
        hidden_size=config.model.vision_hidden_size,
        projection_dim=config.model.projection_dim,
        freeze_encoder=True,
    )

    language_model = MistralQLoRA(
        model_name=config.model.lm_model_name,
        load_in_4bit=config.model.load_in_4bit,
        lora_r=config.model.lora_r,
        lora_alpha=config.model.lora_alpha,
        lora_dropout=config.model.lora_dropout,
        lora_target_modules=list(config.model.lora_target_modules),
        gradient_checkpointing=False,
    )

    fusion = CrossAttentionFusion(
        d_model=config.model.projection_dim,
        n_heads=config.model.fusion_num_heads,
        dropout=0.0,
        use_residual=config.model.fusion_use_residual,
    )

    medvqa_model = MedVQAModel(
        vision_encoder=vision_encoder,
        language_model=language_model,
        fusion=fusion,
        freeze_vision=True,
        num_beams=config.model.num_beams,
    )

    # Load checkpoint if available
    checkpoint_path = config.inference.checkpoint_path
    if os.path.exists(checkpoint_path):
        print(f"[LOAD] Loading checkpoint from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        medvqa_model.load_state_dict(state_dict, strict=False)
        print("[OK] Checkpoint loaded")

    medvqa_model.to(device)
    medvqa_model.eval()

    gradcam = GradCAM(model=medvqa_model, image_size=config.data.image_size)

    _pipeline = MedVQAPipeline(
        model=medvqa_model,
        text_preprocessor=text_preprocessor,
        image_preprocessor=image_preprocessor,
        gradcam=gradcam,
        device=device,
        cache_vision=config.inference.cache_vision_outputs,
        uncertainty_threshold=config.inference.uncertainty_threshold,
        temperature_scaler_path=config.inference.temperature_scaler_path,
    )

    print("[OK] Pipeline ready!")
    return _pipeline


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    vram_gb = None
    if torch.cuda.is_available():
        vram_gb = torch.cuda.memory_allocated() / 1e9

    mode_str = "API" if _mode == "api" else "LOCAL"
    model_name = f"MedVQA ({mode_str} mode)"

    return HealthResponse(
        status="ok",
        model=model_name,
        vram_gb=vram_gb,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    question: str = Form(""),
    conversation: str = Form("[]"),
    patient_context: str = Form("{}"),
    roi: str = Form(""),
):
    """Predict answer for a medical image and clinical question.

    Accepts a multipart upload with an image file, a question string,
    optional conversation history (JSON array of previous Q&A pairs),
    optional patient context (JSON with age, sex, history, symptoms),
    and optional ROI region (JSON with x, y, w, h normalized 0-1).
    """
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question is required")

    if not file.filename:
        raise HTTPException(status_code=400, detail="Image file is required")

    # Validate file type
    allowed_types = {".jpg", ".jpeg", ".png", ".dcm", ".nii", ".nii.gz"}
    ext = _normalize_extension(file.filename)
    if ext not in allowed_types:
        raise HTTPException(
            status_code=400, detail=f"Unsupported file type: {ext}. Allowed: {allowed_types}"
        )

    # Parse conversation history
    try:
        conv_history = json.loads(conversation) if conversation.strip() else []
        if not isinstance(conv_history, list):
            conv_history = []
    except (json.JSONDecodeError, TypeError):
        conv_history = []

    # Parse patient context
    try:
        pctx = (
            json.loads(patient_context)
            if patient_context.strip() and patient_context != "{}"
            else {}
        )
        if not isinstance(pctx, dict):
            pctx = {}
    except (json.JSONDecodeError, TypeError):
        pctx = {}

    # Parse ROI
    try:
        roi_region = json.loads(roi) if roi.strip() else None
        if not isinstance(roi_region, dict):
            roi_region = None
    except (json.JSONDecodeError, TypeError):
        roi_region = None

    # Save uploaded file temporarily
    temp_dir = Path("data/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"upload_{int(time.time())}{ext}"

    try:
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)

        pipeline = get_pipeline()

        # Build enhanced prompt
        context_parts = []

        # Patient context prefix
        if pctx:
            ctx_lines = ["Patient context:"]
            for key, label in [
                ("age", "Age"),
                ("sex", "Sex"),
                ("history", "History"),
                ("symptoms", "Symptoms"),
            ]:
                val = pctx.get(key, "")
                if val:
                    ctx_lines.append(f"  {label}: {val}")
            if len(ctx_lines) > 1:
                context_parts.append("\n".join(ctx_lines))

        # ROI annotation
        if roi_region:
            rx, ry, rw, rh = (
                roi_region.get("x", 0),
                roi_region.get("y", 0),
                roi_region.get("w", 1),
                roi_region.get("h", 1),
            )
            context_parts.append(
                f"The user has drawn a region of interest covering approximately ({rx * 100:.0f}%, {ry * 100:.0f}%) to ({(rx + rw) * 100:.0f}%, {(ry + rh) * 100:.0f}%) of the image. Focus your analysis on this region."
            )

        # Conversation history
        if conv_history:
            context_parts.append("Conversation history:")
            for turn in conv_history[-6:]:
                q = turn.get("question", "")
                a = turn.get("answer", "")
                if q:
                    context_parts.append(f"  Q: {q}")
                if a:
                    context_parts.append(f"  A: {a}")

        context_parts.append(f"New question: {question}")
        enhanced_question = "\n".join(context_parts)

        result = pipeline.predict(
            image_path=str(temp_path),
            question=enhanced_question,
            generate_heatmap=True,
            use_mc_dropout=True,
        )

        # Generate follow-up suggestions if in API mode
        follow_ups: list[str] = []
        if pipeline._is_api_mode and pipeline.api_llm_client is not None:
            try:
                fu_prompt = (
                    f"Based on the medical image and this conversation:\n"
                    f"Latest question: {question}\n"
                    f"Latest answer: {result.answer}\n\n"
                    f"Suggest 3-4 follow-up questions a radiologist might ask next. "
                    f"Return ONLY a JSON array of strings."
                )
                fu_raw = pipeline.api_llm_client.predict(
                    image_path=str(temp_path), question=fu_prompt
                )
                fu_match = re.search(r"\[.*?\]", fu_raw, re.DOTALL)
                if fu_match:
                    try:
                        follow_ups = json.loads(fu_match.group(0))
                        if not isinstance(follow_ups, list):
                            follow_ups = []
                    except json.JSONDecodeError:
                        follow_ups = []
                follow_ups = [
                    str(qq) for qq in follow_ups if isinstance(qq, str) and len(str(qq)) > 5
                ][:4]
            except Exception:
                follow_ups = []

        # Build response
        response = PredictResponse(
            answer=result.answer,
            confidence=result.confidence,
            uncertainty_flag=result.uncertainty_flag,
            heatmap_path=result.heatmap_path,
            latency_ms=result.latency_ms,
            predictive_entropy=result.predictive_entropy,
            follow_up_questions=follow_ups,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file
        if temp_path.exists():
            temp_path.unlink()

    return response


@app.post("/suggest-questions", response_model=SuggestQuestionsResponse)
async def suggest_questions(file: UploadFile = File(...)):
    """Generate suggested clinical questions for a medical image.

    Uses the LLM to propose 4-6 relevant questions based on the image content.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Image file is required")

    # Validate file type
    allowed_types = {".jpg", ".jpeg", ".png", ".dcm", ".nii", ".nii.gz"}
    ext = _normalize_extension(file.filename)
    if ext not in allowed_types:
        raise HTTPException(
            status_code=400, detail=f"Unsupported file type: {ext}. Allowed: {allowed_types}"
        )

    temp_dir = Path("data/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"suggest_{int(time.time())}{ext}"

    try:
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)

        pipeline = get_pipeline()

        if pipeline._is_api_mode and pipeline.api_llm_client is not None:
            # Use the LLM to generate image-specific questions
            prompt = (
                "You are a medical imaging expert. Look at this medical image and "
                "suggest 4 to 6 relevant clinical questions that a radiologist "
                "might ask about it. Return ONLY a JSON array of strings, each "
                "string being one clinical question. Do not include any other text."
            )
            raw = pipeline.api_llm_client.predict(image_path=str(temp_path), question=prompt)
            # Parse the JSON array from the response
            match = re.search(r"\[.*?\]", raw, re.DOTALL)
            if match:
                try:
                    questions = json.loads(match.group(0))
                except json.JSONDecodeError:
                    questions = []
            else:
                # Fallback: split by newlines
                lines = [
                    ln.strip().strip("\"'1234567890. )(").strip()
                    for ln in raw.split("\n")
                    if ln.strip()
                ]
                questions = [ln for ln in lines if len(ln) > 10][:6]

            if not questions:
                questions = [
                    "What abnormalities are visible in this image?",
                    "Is the anatomy normal?",
                    "Are there any signs of pathology?",
                    "Describe the key findings.",
                ]
        else:
            # Local mode fallback
            questions = [
                "What abnormalities are visible in this image?",
                "Is the anatomy normal?",
                "Are there any signs of pathology?",
                "Describe the key findings.",
            ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path.exists():
            temp_path.unlink()

    return SuggestQuestionsResponse(questions=questions)


@app.get("/predict/")
async def predict_get(image_path: str = "", question: str = ""):
    """Predict with file path instead of upload (for testing)."""
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question is required")
    if not image_path or not Path(image_path).exists():
        raise HTTPException(status_code=400, detail="Valid image path is required")

    pipeline = get_pipeline()
    result = pipeline.predict(
        image_path=image_path, question=question, generate_heatmap=True, use_mc_dropout=True
    )

    return PredictResponse(
        answer=result.answer,
        confidence=result.confidence,
        uncertainty_flag=result.uncertainty_flag,
        heatmap_path=result.heatmap_path,
        latency_ms=result.latency_ms,
        predictive_entropy=result.predictive_entropy,
    )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get cached evaluation metrics."""
    candidates = [
        Path("experiments/metrics.json"),
        Path("experiments/eval_metrics.json"),
        Path("experiments/medvqa_run/metrics.json"),
    ]

    for path in candidates:
        if path.exists():
            try:
                import json

                with open(path) as f:
                    payload = json.load(f)

                return MetricsResponse(
                    accuracy=payload.get("yesno_accuracy") or payload.get("accuracy"),
                    bleu_4=payload.get("bleu_4"),
                    rouge_l_f1=payload.get("rouge_l_f1"),
                    ece=payload.get("ece"),
                    brier_score=payload.get("brier_score"),
                    total_samples=int(payload.get("total_samples", 0)),
                )
            except Exception:
                break

    return MetricsResponse(total_samples=0)


@app.get("/heatmap/{image_name}")
async def get_heatmap(image_name: str):
    """Serve a generated heatmap image."""
    heatmap_path = Path("experiments/predictions") / f"heatmap_{image_name}"
    if heatmap_path.exists():
        return FileResponse(str(heatmap_path), media_type="image/png")
    return {"error": "Heatmap not found"}


if __name__ == "__main__":
    import uvicorn

    mode = "api" if _mode == "api" else "local"
    print(f"[INIT] Starting MedVQA API in {mode.upper()} mode")
    print("[INIT]   python api/main.py --mode api   (API mode)")
    print("[INIT]   python api/main.py --mode local (local model mode)")
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
