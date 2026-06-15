"""Image and text preprocessing for medical VQA.

Medical images require special handling: we must NOT distort anatomy
by using non-uniform resizing. Letterboxing preserves aspect ratio.
"""

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from transformers import AutoTokenizer


class MedicalImagePreprocessor:
    """Preprocess medical images with anatomy-preserving transforms.

    Key design decisions:
    - Letterbox resize: preserves aspect ratio (critical for medical images)
    - No random flips: flipping would change anatomical left/right orientation
    - Conservative normalization: uses ImageNet stats (BioViL-T was trained on
      ImageNet-initialized weights)

    Args:
        image_size: Target size (default: 224 for BioViL-T).
        mean: Normalization mean (ImageNet stats).
        std: Normalization standard deviation.
    """

    def __init__(
        self,
        image_size: int = 224,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        self.image_size = image_size
        self.mean = mean
        self.std = std

        # Base transform: resize and normalize only
        self.base_transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

        # Transform without normalization (for Grad-CAM visualization)
        self.unnormalized_transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
        ])

    def __call__(self, image: Image.Image) -> torch.Tensor:
        """Preprocess a medical image.

        Args:
            image: PIL Image in RGB mode.

        Returns:
            Normalized image tensor of shape (3, H, W).
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
        return self.base_transform(image)

    def letterbox_resize(self, image: Image.Image) -> Image.Image:
        """Resize with aspect ratio preservation via letterboxing.

        This is the preferred method for medical images because it avoids
        distorting anatomical structures.

        Args:
            image: PIL Image.

        Returns:
            Letterboxed PIL Image.
        """
        w, h = image.size
        target_size = self.image_size

        # Calculate scale to fit within target_size
        scale = min(target_size / w, target_size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize preserving aspect ratio
        resized = TF.resize(image, (new_h, new_w), interpolation=T.InterpolationMode.BICUBIC)

        # Create letterboxed image (padded to square)
        result = Image.new("RGB", (target_size, target_size), (128, 128, 128))
        result.paste(resized, ((target_size - new_w) // 2, (target_size - new_h) // 2))

        return result

    def get_transform_with_letterbox(self) -> T.Compose:
        """Get a transform that uses letterbox resize instead of crop.

        Useful for evaluation when you want to preserve the full image.
        """
        return T.Compose([
            T.Lambda(lambda x: self.letterbox_resize(x)),
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std),
        ])


def _normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    """Normalize a numpy array to uint8 [0, 255]."""
    array = array.astype(np.float32)
    array = array - np.min(array)
    max_val = np.max(array)
    if max_val > 0:
        array = array / max_val
    array = (array * 255.0).clip(0, 255).astype(np.uint8)
    return array


def load_medical_image(image_path: str) -> Image.Image:
    """Load a medical image from common formats into a PIL Image.

    Supports standard image formats via PIL, and optionally DICOM/NIfTI
    if the relevant dependencies are installed.

    Args:
        image_path: Path to the image file.

    Returns:
        PIL Image in RGB mode.
    """
    path_lower = image_path.lower()
    if path_lower.endswith(".dcm"):
        try:
            import pydicom  # type: ignore
        except ImportError as exc:
            raise ImportError("pydicom is required to load DICOM images") from exc

        dataset = pydicom.dcmread(image_path)
        pixel_array = dataset.pixel_array.astype(np.float32)

        # Apply rescale if present
        slope = float(getattr(dataset, "RescaleSlope", 1.0))
        intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
        pixel_array = pixel_array * slope + intercept

        normalized = _normalize_to_uint8(pixel_array)
        image = Image.fromarray(normalized)
        return image.convert("RGB")

    if path_lower.endswith(".nii") or path_lower.endswith(".nii.gz"):
        try:
            import nibabel as nib  # type: ignore
        except ImportError as exc:
            raise ImportError("nibabel is required to load NIfTI images") from exc

        nifti = nib.load(image_path)
        volume = nifti.get_fdata()

        # Handle 4D data by taking the first volume
        if volume.ndim == 4:
            volume = volume[..., 0]

        if volume.ndim == 2:
            slice_data = volume
        else:
            # Take middle slice along the last axis
            mid_index = volume.shape[-1] // 2
            slice_data = volume[..., mid_index]

        normalized = _normalize_to_uint8(slice_data)
        image = Image.fromarray(normalized)
        return image.convert("RGB")

    # Default: standard image formats
    image = Image.open(image_path)
    return image.convert("RGB")


class TextPreprocessor:
    """Preprocess medical questions and answers for Mistral-7B.

    Handles tokenization with Mistral's chat template, which uses
    [INST] and [/INST] tokens for instruction formatting.

    Args:
        model_name: HuggingFace model name for the tokenizer.
        max_question_length: Max tokens for questions (truncation).
        max_answer_length: Max tokens for answers (truncation).
    """

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
        max_question_length: int = 128,
        max_answer_length: int = 64,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_question_length = max_question_length
        self.max_answer_length = max_answer_length

        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Mistral uses left padding for generation
        self.tokenizer.padding_side = "right"

    def tokenize_question(self, question: str) -> dict:
        """Tokenize a clinical question with Mistral's instruction template.

        Format: <s> [INST] {question} [/INST]

        Args:
            question: Clinical question string.

        Returns:
            Dict with input_ids, attention_mask.
        """
        # Apply Mistral chat template
        messages = [{"role": "user", "content": question}]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        encoded = self.tokenizer(
            formatted,
            max_length=self.max_question_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }

    def tokenize_qa_pair(self, question: str, answer: str) -> dict:
        """Tokenize a full question-answer pair for causal LM training.

        The question prompt is masked out in the labels so loss is computed
        only on the answer tokens.

        Args:
            question: Clinical question string.
            answer: Ground-truth answer string.

        Returns:
            Dict with input_ids, attention_mask, and labels tensors.
        """
        messages = [{"role": "user", "content": question}]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        prompt_ids = self.tokenizer(
            prompt,
            max_length=self.max_question_length,
            truncation=True,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        answer_text = answer.strip()
        if self.tokenizer.eos_token:
            answer_text = f"{answer_text}{self.tokenizer.eos_token}"

        answer_ids = self.tokenizer(
            answer_text,
            max_length=self.max_answer_length,
            truncation=True,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        input_ids = torch.cat([prompt_ids, answer_ids], dim=0)
        attention_mask = torch.ones_like(input_ids)
        labels = torch.cat([torch.full_like(prompt_ids, -100), answer_ids], dim=0)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    def tokenize_answer(self, answer: str) -> dict:
        """Tokenize a ground-truth answer.

        Args:
            answer: Answer string.

        Returns:
            Dict with input_ids, attention_mask for the answer.
        """
        encoded = self.tokenizer(
            answer,
            max_length=self.max_answer_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }

    def decode_answer(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs back to an answer string.

        Args:
            token_ids: Tensor of token IDs.

        Returns:
            Decoded answer string.
        """
        # Skip special tokens and strip whitespace
        return self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()

    def is_yes_no_question(self, question: str) -> bool:
        """Determine if a question is yes/no type.

        Args:
            question: Clinical question string.

        Returns:
            True if the question expects a yes/no answer.
        """
        question_lower = question.lower().strip()
        yes_no_starters = [
            "is ",
            "are ",
            "was ",
            "were ",
            "do ",
            "does ",
            "did ",
            "has ",
            "have ",
            "had ",
            "can ",
            "could ",
            "will ",
            "would ",
            "shall ",
            "should ",
            "may ",
            "might ",
            "does the ",
            "is there ",
            "are there ",
            "does this ",
        ]
        return any(question_lower.startswith(starter) for starter in yes_no_starters)

    @property
    def vocab_size(self) -> int:
        """Get the tokenizer vocabulary size."""
        return len(self.tokenizer)
