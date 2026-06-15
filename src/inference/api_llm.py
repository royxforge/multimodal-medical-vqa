"""API-based LLM client for MedVQA inference.

Provides an alternative to running Mistral-7B locally. Supports four providers:

1. **OpenAI** (gpt-4o, gpt-4o-mini)  — set OPENAI_API_KEY
2. **Anthropic** (claude-3-5-sonnet)  — set ANTHROPIC_API_KEY
3. **Google Gemini** (gemini-1.5-pro) — set GEMINI_API_KEY
4. **Ollama** (llava, bakllava)       — run locally, no key needed

All providers support multimodal (image + text) input, so medical images are
sent directly to the API. This avoids needing a GPU for the LLM while keeping
the vision encoder (for Grad-CAM) and preprocessing on your machine.

Usage:
    from src.inference.api_llm import APILLMClient

    client = APILLMClient(provider="openai", model="gpt-4o")
    answer = client.predict("path/to/xray.png", "Is there a lung nodule?")
"""

import base64
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Optional


@dataclass
class APILLMConfig:
    """Configuration for the API-based LLM client.

    Attributes:
        provider: One of "openai", "anthropic", "gemini", "ollama".
        model: Model name for the provider (e.g. "gpt-4o", "claude-3-5-sonnet-20241022",
               "gemini-1.5-pro", "llava").
        temperature: Sampling temperature (0.0 to 2.0).
        max_tokens: Maximum tokens in the response.
        api_key: API key (if None, read from environment variable).
        ollama_base_url: Base URL for Ollama (default: http://localhost:11434).
    """

    provider: str = "openai"
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 256
    api_key: Optional[str] = None
    ollama_base_url: str = "http://localhost:11434"

    _ENV_MAP: ClassVar[dict] = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }

    def resolve_api_key(self) -> Optional[str]:
        """Resolve API key from explicit value, env var, or .env file."""
        if self.api_key:
            return self.api_key

        env_var = self._ENV_MAP.get(self.provider)
        if env_var and env_var in os.environ:
            return os.environ[env_var]

        # Try loading from .env file
        env_path = Path(".env")
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith(f"{env_var}="):
                        return line.split("=", 1)[1].strip("\"'")

        return None

    def validate(self) -> tuple[bool, str]:
        """Validate that the configuration is usable.

        Returns:
            Tuple of (is_valid, error_message).
        """
        if self.provider not in ("openai", "anthropic", "gemini", "ollama"):
            return (
                False,
                f"Unsupported provider: {self.provider}. Choose: openai, anthropic, gemini, ollama",
            )

        if self.provider == "ollama":
            return True, ""

        api_key = self.resolve_api_key()
        if not api_key:
            env_var = self._ENV_MAP.get(self.provider, "")
            return False, (
                f"No API key found for {self.provider}. "
                f"Set the {env_var} environment variable or add it to .env"
            )

        return True, ""


class APILLMClient:
    """Client for API-based LLM inference.

    Supports OpenAI, Anthropic, Gemini, and local Ollama. The medical image
    is base64-encoded and sent alongside the clinical question so multimodal
    models can reason over it directly.
    """

    def __init__(self, config: Optional[APILLMConfig] = None):
        self.config = config or APILLMConfig()
        self._session = None  # Lazy-init per provider

        # Validate configuration on init to fail fast
        is_valid, err_msg = self.config.validate()
        if not is_valid:
            raise ValueError(err_msg)

    # ── Image encoding ───────────────────────────────────────────────────────

    @staticmethod
    def _encode_image(image_path: str) -> str:
        """Read an image file and return a base64 data URI string."""
        with open(image_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")

        ext = Path(image_path).suffix.lower()
        mime = {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "webp": "image/webp",
            "dcm": "application/dicom",
        }.get(ext.lstrip("."), "image/png")
        return f"data:{mime};base64,{data}"

    # ── Provider-specific implementations ────────────────────────────────────

    def _predict_openai(self, image_path: str, question: str) -> str:
        """Call OpenAI GPT-4o / GPT-4 Vision."""
        import openai  # type: ignore

        client = openai.OpenAI(api_key=self.config.resolve_api_key())
        b64 = self._encode_image(image_path)

        resp = client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are a medical imaging expert. Answer the "
                                "following clinical question based on the medical "
                                "image provided.\n\n"
                                f"Clinical question: {question}\n\n"
                                "Answer concisely and directly."
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": b64}},
                    ],
                }
            ],
        )
        return resp.choices[0].message.content.strip()

    def _predict_anthropic(self, image_path: str, question: str) -> str:
        """Call Anthropic Claude 3.5 Sonnet / Opus / Haiku."""
        import anthropic  # type: ignore

        client = anthropic.Anthropic(api_key=self.config.resolve_api_key())
        b64 = self._encode_image(image_path)

        # Anthropic uses a separate media_type + base64 format
        ext = Path(image_path).suffix.lower()
        media_type = {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "webp": "image/webp",
        }.get(ext.lstrip("."), "image/png")
        # Strip the data:...;base64, prefix
        raw_b64 = b64.split(",", 1)[1] if "," in b64 else b64

        resp = client.messages.create(
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are a medical imaging expert. Answer the "
                                "following clinical question based on the medical "
                                "image provided.\n\n"
                                f"Clinical question: {question}\n\n"
                                "Answer concisely and directly."
                            ),
                        },
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": media_type, "data": raw_b64},
                        },
                    ],
                }
            ],
        )
        return resp.content[0].text.strip()

    def _predict_gemini(self, image_path: str, question: str) -> str:
        """Call Google Gemini 1.5 Pro / Flash."""
        import google.generativeai as genai  # type: ignore

        genai.configure(api_key=self.config.resolve_api_key())
        model = genai.GenerativeModel(self.config.model)

        uploaded = genai.upload_file(path=image_path, display_name=Path(image_path).name)

        resp = model.generate_content(
            [
                uploaded,
                (
                    "You are a medical imaging expert. Answer the following "
                    "clinical question based on the medical image provided.\n\n"
                    f"Clinical question: {question}\n\n"
                    "Answer concisely and directly."
                ),
            ],
            generation_config=genai.types.GenerationConfig(
                temperature=self.config.temperature, max_output_tokens=self.config.max_tokens
            ),
        )
        return resp.text.strip()

    def _predict_ollama(self, image_path: str, question: str) -> str:
        """Call a locally-running Ollama model (e.g. llava, bakllava)."""
        import requests  # type: ignore

        b64_data = self._encode_image(image_path)
        payload = {
            "model": self.config.model,
            "prompt": (
                f"You are a medical imaging expert. Answer the following "
                f"clinical question based on the medical image provided.\n\n"
                f"Clinical question: {question}\n\n"
                f"Answer concisely and directly."
            ),
            "images": [b64_data.split(",", 1)[1]],
            "options": {"temperature": self.config.temperature},
            "stream": False,
        }

        base_url = self.config.ollama_base_url.rstrip("/")
        resp = requests.post(f"{base_url}/api/generate", json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["response"].strip()

    # ── Public API ───────────────────────────────────────────────────────────

    def predict(self, image_path: str, question: str) -> str:
        """Send a medical image + clinical question to the API and return the answer.

        Args:
            image_path: Path to the medical image file.
            question: Clinical question string.

        Returns:
            Answer text from the LLM.
        """
        start = time.time()
        provider = self.config.provider

        if provider == "openai":
            answer = self._predict_openai(image_path, question)
        elif provider == "anthropic":
            answer = self._predict_anthropic(image_path, question)
        elif provider == "gemini":
            answer = self._predict_gemini(image_path, question)
        elif provider == "ollama":
            answer = self._predict_ollama(image_path, question)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        elapsed = (time.time() - start) * 1000
        print(f"[API] {provider}/{self.config.model} answered in {elapsed:.0f}ms")
        return answer

    def predict_batch(self, items: list[dict[str, str]]) -> list[str]:
        """Run inference on multiple image-question pairs.

        Args:
            items: List of dicts with 'image_path' and 'question'.

        Returns:
            List of answer strings.
        """
        return [self.predict(item["image_path"], item["question"]) for item in items]
