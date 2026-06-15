"""Gradio demo interface for MedVQA.

Provides a clean web UI for testing the model with image uploads,
question input, and visualization of answers, confidence, and Grad-CAM.

Supports two modes:
  1. Local mode (default): Uses the full MedVQAModel (vision encoder + Mistral-7B).
     Requires a GPU for Mistral-7B.
  2. API mode (--mode api): Uses a cloud LLM (OpenAI, Anthropic, Gemini, Ollama).
     No GPU needed.

Usage:
    python api/demo.py                               (local mode)
    python api/demo.py --mode api                    (API mode)
"""

import os
import sys
from pathlib import Path

import gradio as gr
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── CLI argument: inference mode ────────────────────────────────────────────
_INFERENCE_MODE = "local"  # default; overridden by --mode flag
for i, arg in enumerate(sys.argv):
    if arg == "--mode" and i + 1 < len(sys.argv):
        _INFERENCE_MODE = sys.argv[i + 1]
    elif arg.startswith("--mode="):
        _INFERENCE_MODE = arg.split("=", 1)[1]

_mode = _INFERENCE_MODE  # "local" or "api"

# Global pipeline
_pipeline = None


def load_pipeline():
    """Load or reload the inference pipeline.

    The mode is determined by the --mode CLI argument passed at startup.
    In API mode, only the vision encoder + API client are loaded (no Mistral-7B).
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
        print(f"[INIT] MedVQA demo in API mode on {device}...")

        from src.inference.api_llm import APILLMClient, APILLMConfig

        api_config = APILLMConfig(
            provider=config.api.provider,
            model=config.api.model,
            temperature=config.api.temperature,
            max_tokens=config.api.max_tokens,
            ollama_base_url=config.api.ollama_base_url,
        )
        api_client = APILLMClient(api_config)

        # Grad-CAM requires the full MedVQAModel (needs language model for
        # forward/backward hooks). Not available in API mode.
        # The pipeline handles this gracefully with a warning.
        _pipeline = MedVQAPipeline(
            image_preprocessor=image_preprocessor,
            text_preprocessor=text_preprocessor,
            gradcam=None,
            device=device,
            api_llm_client=api_client,
        )

        print(f"[OK] API demo ready (provider={config.api.provider}, model={config.api.model})")
        return _pipeline

    # ── Local mode: full MedVQAModel (vision encoder + Mistral-7B) ──
    print(f"[INIT] MedVQA demo in LOCAL mode on {device}...")

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

    # Load checkpoint
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

    print("[OK] MedVQA demo ready!")
    return _pipeline


def predict(image, question):
    """Run inference and return formatted outputs for Gradio.

    Args:
        image: PIL Image from Gradio upload.
        question: Question string from text input.

    Returns:
        Tuple of (answer, confidence_display, heatmap, status).
    """
    if image is None:
        return "", 0.0, None, "[WARN] Please upload an image"

    if not question.strip():
        return "", 0.0, None, "[WARN] Please enter a question"

    try:
        # Save uploaded image temporarily
        temp_dir = Path("data/temp")
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / "gradio_upload.png"
        image.save(str(temp_path))

        # Run inference
        pipeline = load_pipeline()
        result = pipeline.predict(
            image_path=str(temp_path), question=question, generate_heatmap=True, use_mc_dropout=True
        )

        # Format confidence
        confidence_pct = f"{result.confidence * 100:.1f}%"
        uncertainty_warning = ""
        if result.uncertainty_flag:
            uncertainty_warning = " [LOW CONFIDENCE]"

        # Load heatmap
        heatmap_image = None
        if result.heatmap_path and Path(result.heatmap_path).exists():
            from PIL import Image as PILImage

            heatmap_image = PILImage.open(result.heatmap_path)

        answer_text = f"{result.answer}{uncertainty_warning}"

        mode_tag = "[API]" if _mode == "api" else "[LOCAL]"
        status = f"{mode_tag} Done in {result.latency_ms:.0f}ms | Confidence: {confidence_pct}"
        if result.predictive_entropy is not None:
            status += f" | Entropy: {result.predictive_entropy:.3f}"

        # Cleanup
        if temp_path.exists():
            temp_path.unlink()

        return answer_text, result.confidence, heatmap_image, status

    except Exception as e:
        return "", 0.0, None, f"[ERROR] {e!s}"


# Example VQA-RAD samples
EXAMPLE_SAMPLES = [
    {"question": "Is there a lung nodule in the upper left lobe?", "label": "Yes/no"},
    {"question": "What is the size of the cardiac silhouette?", "label": "Open-ended"},
    {"question": "Is the endotracheal tube placement correct?", "label": "Yes/no"},
    {"question": "Describe the findings in the right lower zone.", "label": "Open-ended"},
    {"question": "Are there signs of pneumothorax?", "label": "Yes/no"},
]

# Build Gradio interface
css_style = """
    .gradio-container { max-width: 1100px; margin: auto; }
    h1 { text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
         -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .confidence-bar { height: 20px; border-radius: 10px; }
"""
with gr.Blocks(
    title="MedVQA - Medical Visual Question Answering",
) as demo:
    mode_tag = "API" if _mode == "api" else "LOCAL"
    gr.Markdown(
        f"""
        # MedVQA - Multimodal Medical Visual Question Answering

        **Mode: {mode_tag}** | Upload a medical image (X-ray, MRI, CT, pathology) and ask a clinical question.
        The model will answer with a **confidence score** and **Grad-CAM attention heatmap**.
        """
    )

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            # Image upload
            image_input = gr.Image(type="pil", label="Medical Image", height=350)

            # Pre-loaded examples
            gr.Markdown("### Example Questions")
            example_btns = []
            for ex in EXAMPLE_SAMPLES:
                btn = gr.Button(f"{ex['label']}: {ex['question'][:50]}...", size="sm")
                example_btns.append((btn, ex["question"]))

        with gr.Column(scale=1):
            # Question input
            question_input = gr.Textbox(
                label="Clinical Question",
                placeholder="e.g., Is there a lung nodule in the upper left lobe?",
                lines=2,
            )

            submit_btn = gr.Button("Analyze", variant="primary", size="lg")

            # Outputs
            answer_output = gr.Textbox(label="Answer", lines=2, interactive=False)

            confidence_output = gr.Slider(
                label="Confidence Score", minimum=0, maximum=1, value=0, interactive=False
            )

            heatmap_output = gr.Image(type="pil", label="Grad-CAM Attention Heatmap", height=300)

            status_output = gr.Textbox(label="Status", lines=1, interactive=False)

    # Wire up events
    submit_btn.click(
        fn=predict,
        inputs=[image_input, question_input],
        outputs=[answer_output, confidence_output, heatmap_output, status_output],
    )

    question_input.submit(
        fn=predict,
        inputs=[image_input, question_input],
        outputs=[answer_output, confidence_output, heatmap_output, status_output],
    )

    for btn, question in example_btns:
        btn.click(fn=lambda q=question: q, outputs=[question_input])

    gr.Markdown(
        """
        ---
        ### How It Works
        1. **Vision Encoder**: CLIP ViT-L/14 processes the medical image into visual patches
        2. **Fusion**: Cross-attention between visual features and question tokens
        3. **LLM**: Mistral-7B (QLoRA) or cloud API generates the answer
        4. **Confidence**: MC Dropout (20 samples) + Temperature Scaling (local mode)
        5. **Explainability**: Grad-CAM highlights diagnostically relevant regions

        **Disclaimer**: This is a research prototype. Not for clinical use.
        """
    )


def main():
    print("=" * 50)
    mode_tag = "API" if _mode == "api" else "LOCAL"
    print(f"  MedVQA - Gradio Demo ({mode_tag} mode)")
    print("=" * 50)
    print("  python api/demo.py --mode api   (API mode)")
    print("  python api/demo.py --mode local (local model mode)")
    print("=" * 50)

    # Pre-load pipeline
    load_pipeline()

    # Launch demo (theme and css go here in Gradio 6.0+)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=False,
        theme=gr.themes.Soft(),
        css=css_style,
    )


if __name__ == "__main__":
    main()
