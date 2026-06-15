"""Pydantic schemas for the MedVQA API.

Defines request and response models with validation for the FastAPI endpoints.
"""

from typing import Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Request model for the /predict endpoint.

    Attributes:
        question: Clinical question about the image.
    """

    question: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Clinical question about the medical image",
        example="Is there a lung nodule in the upper left lobe?",
    )


class PredictResponse(BaseModel):
    """Response model for the /predict endpoint.

    Attributes:
        answer: Generated natural language answer.
        confidence: Calibrated confidence score (0-1).
        uncertainty_flag: Whether the model is uncertain.
        heatmap_path: URL to Grad-CAM overlay if generated.
        latency_ms: Inference time in milliseconds.
    """

    answer: str = Field(..., description="Generated answer to the clinical question")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Calibrated confidence score")
    uncertainty_flag: bool = Field(
        ..., description="Whether the model is uncertain about its answer"
    )
    heatmap_path: Optional[str] = Field(None, description="Path to Grad-CAM visualization overlay")
    latency_ms: float = Field(..., description="Inference time in milliseconds")
    predictive_entropy: Optional[float] = Field(
        None, description="Predictive entropy of the answer distribution"
    )
    follow_up_questions: list[str] = Field(
        default_factory=list,
        description="Suggested follow-up questions based on the conversation context",
    )


class HealthResponse(BaseModel):
    """Response model for the /health endpoint.

    Attributes:
        status: Service status.
        model: Model configuration info.
        vram_gb: Current GPU memory usage.
        device: Active device.
    """

    status: str = Field("ok", description="Service health status")
    model: str = Field(..., description="Model name and configuration")
    vram_gb: Optional[float] = Field(None, description="Current GPU memory usage in GB")
    device: str = Field(..., description="Active inference device")


class SuggestQuestionsResponse(BaseModel):
    """Response model for the /suggest-questions endpoint.

    Attributes:
        questions: List of suggested clinical questions relevant to the uploaded image.
    """

    questions: list[str] = Field(
        ..., description="List of suggested questions relevant to the uploaded image"
    )


class MetricsResponse(BaseModel):
    """Response model for the /metrics endpoint.

    Attributes:
        accuracy: Overall accuracy.
        bleu_4: BLEU-4 score.
        rouge_l_f1: ROUGE-L F1 score.
        ece: Expected Calibration Error.
        brier_score: Brier score for calibration.
        total_samples: Number of evaluation samples.
    """

    accuracy: Optional[float] = Field(None, description="Closed-ended accuracy")
    bleu_4: Optional[float] = Field(None, description="BLEU-4 score")
    rouge_l_f1: Optional[float] = Field(None, description="ROUGE-L F1 score")
    ece: Optional[float] = Field(None, description="Expected Calibration Error")
    brier_score: Optional[float] = Field(None, description="Brier score")
    total_samples: int = Field(0, description="Number of evaluation samples")
