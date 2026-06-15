"""End-to-end inference pipeline for MedVQA.

Supports two modes:
  - **Local** (default): Uses the local MedVQAModel (vision encoder + Mistral-7B).
    Requires a GPU for Mistral-7B (15GB+ VRAM).
  - **API**: Uses a cloud LLM (OpenAI, Anthropic, Gemini, Ollama) instead of
    the local model. The vision encoder still runs locally for Grad-CAM.
    No GPU needed.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


@dataclass
class PredictionResult:
    """Structured output from the inference pipeline.

    Attributes:
        answer: Generated natural language answer.
        confidence: Calibrated confidence score in [0, 1].
        uncertainty_flag: True if model is not confident (< 0.3 threshold).
        heatmap_path: Path to saved Grad-CAM overlay (if generated).
        latency_ms: Inference time in milliseconds.
        predictive_entropy: Entropy of the predictive distribution.
        question: Original question text.
    """

    answer: str
    confidence: float
    uncertainty_flag: bool
    heatmap_path: Optional[str] = None
    latency_ms: float = 0.0
    predictive_entropy: Optional[float] = None
    question: str = ""


class MedVQAPipeline:
    """End-to-end inference pipeline for medical VQA.

    Supports two modes:
      - **Local** (default): Uses the local MedVQAModel (vision encoder + Mistral-7B).
        Requires a GPU for Mistral-7B (15GB+ VRAM).
      - **API**: Uses a cloud LLM (OpenAI, Anthropic, Gemini, Ollama) instead of
        the local model. The vision encoder still runs locally for Grad-CAM.
        No GPU needed.

    The mode is determined by whether ``api_llm_client`` is passed.

    Handles the complete inference workflow: image loading -> preprocessing ->
    tokenization -> model inference -> confidence estimation -> Grad-CAM -> output.

    Args:
        model: MedVQAModel instance (not needed in API mode).
        text_preprocessor: TextPreprocessor instance.
        image_preprocessor: MedicalImagePreprocessor instance.
        gradcam: Optional GradCAM instance for heatmap generation.
        device: Device to run inference on.
        cache_vision: Whether to cache vision encoder outputs per image.
        uncertainty_threshold: Confidence below this flags uncertainty.
        temperature_scaler_path: Path to saved temperature scaler.
        api_llm_client: If provided, use this API client instead of the local model.
    """

    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        text_preprocessor: Optional[object] = None,
        image_preprocessor: Optional[object] = None,
        gradcam: Optional[object] = None,
        device: str = "cuda",
        cache_vision: bool = True,
        uncertainty_threshold: float = 0.3,
        temperature_scaler_path: Optional[str] = None,
        api_llm_client: Optional[object] = None,
    ):
        self.model = model
        self.text_preprocessor = text_preprocessor
        self.image_preprocessor = image_preprocessor
        self.gradcam = gradcam
        self.device = device
        self.cache_vision = cache_vision
        self.uncertainty_threshold = uncertainty_threshold
        self.api_llm_client = api_llm_client
        self._is_api_mode = api_llm_client is not None

        # Vision cache: {image_path: visual_features}
        self._vision_cache: dict[str, torch.Tensor] = {}

        self.temperature_scaler = None
        if temperature_scaler_path and not self._is_api_mode:
            scaler_path = Path(temperature_scaler_path)
            if scaler_path.exists():
                from .uncertainty import TemperatureScaler

                state = torch.load(scaler_path, map_location="cpu")
                scaler = TemperatureScaler()
                if isinstance(state, dict) and "temperature" in state:
                    value = float(state["temperature"])
                elif isinstance(state, int | float):
                    value = float(state)
                elif torch.is_tensor(state):
                    value = float(state.item())
                else:
                    value = None

                if value is not None:
                    scaler.temperature = torch.nn.Parameter(torch.tensor(value))
                    scaler._fitted = True
                self.temperature_scaler = scaler

        if model is not None:
            self.model.eval()

    def _get_vision_outputs(
        self, image_tensor: torch.Tensor, image_path: str
    ) -> dict[str, torch.Tensor]:
        """Get or compute cached vision encoder outputs for an image."""
        if not self.cache_vision:
            return self.model.vision_encoder(image_tensor)

        cache_key = str(image_path)
        if cache_key in self._vision_cache:
            return self._vision_cache[cache_key]

        outputs = self.model.vision_encoder(image_tensor)
        cached = {
            "patch_embeddings": outputs["patch_embeddings"],
            "cls_embedding": outputs["cls_embedding"],
        }
        self._vision_cache[cache_key] = cached
        return cached

    # ── Local model inference ────────────────────────────────────────────────

    def _predict_local(
        self,
        image_path: str,
        image_tensor: torch.Tensor,
        question: str,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_mc_dropout: bool,
    ) -> PredictionResult:
        """Run inference using the local MedVQAModel (vision encoder + Mistral-7B)."""
        start_time = time.time()

        vision_outputs = self._get_vision_outputs(image_tensor, image_path)

        with torch.no_grad():
            generated_ids = self.model.generate(
                images=image_tensor,
                input_ids=input_ids,
                attention_mask=attention_mask,
                vision_outputs=vision_outputs,
            )

        answer = self.text_preprocessor.decode_answer(generated_ids[0])

        # Confidence estimation
        confidence = 0.5
        predictive_entropy = None
        uncertainty_flag = False

        if use_mc_dropout:
            from .uncertainty import MonteCarloDropout

            mc_dropout = MonteCarloDropout(self.model, num_samples=20)
            mc_samples = mc_dropout.sample(
                image_tensor, input_ids, attention_mask, vision_outputs=vision_outputs
            )
            uncertainty = mc_dropout.compute_uncertainty(
                mc_samples["predictions"],
                mc_samples["logits"],
                temperature_scaler=self.temperature_scaler,
            )
            confidence = uncertainty["confidence"]
            predictive_entropy = uncertainty["predictive_entropy"]
            uncertainty_flag = confidence < self.uncertainty_threshold
        else:
            with torch.no_grad():
                outputs = self.model(
                    images=image_tensor,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    vision_outputs=vision_outputs,
                )
                last_logits = outputs["logits"][:, -1, :]
                if self.temperature_scaler is not None and self.temperature_scaler._fitted:
                    last_logits = last_logits / float(self.temperature_scaler.temperature.item())
                probs = torch.softmax(last_logits, dim=-1)
                confidence = float(probs.max(dim=-1)[0].item())
                predictive_entropy = float((-probs * torch.log(probs + 1e-10)).sum(dim=-1).item())
                uncertainty_flag = confidence < self.uncertainty_threshold

        latency_ms = (time.time() - start_time) * 1000

        return PredictionResult(
            answer=answer,
            confidence=confidence,
            uncertainty_flag=uncertainty_flag,
            latency_ms=latency_ms,
            predictive_entropy=predictive_entropy,
            question=question,
        )

    # ── API-based inference ──────────────────────────────────────────────────

    def _predict_api(self, image_path: str, question: str) -> PredictionResult:
        """Run inference using the API LLM client (no local model needed)."""
        start_time = time.time()

        answer = self.api_llm_client.predict(image_path=image_path, question=question)

        latency_ms = (time.time() - start_time) * 1000

        return PredictionResult(
            answer=answer,
            confidence=0.9,  # Nominal — not calibrated (request logprobs for production)
            uncertainty_flag=False,
            latency_ms=latency_ms,
            predictive_entropy=None,
            question=question,
        )

    # ── Public predict API ───────────────────────────────────────────────────

    def predict(
        self,
        image_path: str,
        question: str,
        generate_heatmap: bool = True,
        use_mc_dropout: bool = True,
        save_heatmap: bool = True,
        output_dir: str = "experiments/predictions",
    ) -> PredictionResult:
        """Run full inference on an image-question pair.

        Automatically selects the inference path based on whether ``api_llm_client``
        was provided at construction time.

        Args:
            image_path: Path to the medical image file.
            question: Clinical question string.
            generate_heatmap: Whether to generate Grad-CAM heatmap.
            use_mc_dropout: Whether to use MC Dropout for confidence (local mode only).
            save_heatmap: Whether to save the heatmap to disk.
            output_dir: Directory to save heatmap overlays.

        Returns:
            PredictionResult with answer, confidence, and heatmap.
        """
        start_time = time.time()

        # ── API mode: skip local model, send image+question to cloud LLM ──
        if self._is_api_mode:
            result = self._predict_api(image_path, question)

            # Grad-CAM is skipped in API mode because the GradCAM hooks expect
            # the full MedVQAModel (vision encoder + language model) for the
            # forward pass. In API mode, only the vision encoder is loaded.
            # Set generate_heatmap=False explicitly to avoid confusion.
            if generate_heatmap:
                print(
                    "[WARN] Grad-CAM is not available in API mode. "
                    "Switch to local mode for heatmap generation."
                )

            result.heatmap_path = None
            result.uncertainty_flag = False
            return result

        # ── Local mode: use the full MedVQA model ──
        from src.data.preprocessor import load_medical_image

        image = load_medical_image(image_path)
        image_tensor = self.image_preprocessor(image).unsqueeze(0).to(self.device)

        # Tokenize question
        question_enc = self.text_preprocessor.tokenize_question(question)
        input_ids = question_enc["input_ids"].unsqueeze(0).to(self.device)
        attention_mask = question_enc["attention_mask"].unsqueeze(0).to(self.device)

        # Run local inference
        result = self._predict_local(
            image_path, image_tensor, question, input_ids, attention_mask, use_mc_dropout
        )

        # Generate Grad-CAM heatmap
        heatmap_path = None
        if generate_heatmap and self.gradcam is not None:
            try:
                heatmap_np = self.gradcam.compute_heatmap(image_tensor, input_ids, attention_mask)
                if save_heatmap:
                    Path(output_dir).mkdir(parents=True, exist_ok=True)
                    output_name = f"heatmap_{Path(image_path).stem}.png"
                    heatmap_path = str(Path(output_dir) / output_name)
                    overlay = self.gradcam.overlay_on_image(image, heatmap_np)
                    overlay.save(heatmap_path)
            except Exception as e:
                print(f"[WARN] Grad-CAM generation failed: {e}")

        result.heatmap_path = heatmap_path
        result.latency_ms = (time.time() - start_time) * 1000
        return result

    @torch.no_grad()
    def predict_batch(self, items: list, generate_heatmap: bool = False) -> list:
        """Run inference on multiple image-question pairs efficiently.

        Args:
            items: List of dicts with 'image_path' and 'question'.
            generate_heatmap: Whether to generate Grad-CAM for each.

        Returns:
            List of PredictionResult objects.
        """
        results = []
        for item in items:
            result = self.predict(
                image_path=item["image_path"],
                question=item["question"],
                generate_heatmap=generate_heatmap,
            )
            results.append(result)

        return results
