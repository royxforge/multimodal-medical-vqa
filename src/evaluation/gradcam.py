"""Grad-CAM heatmap generation for the vision encoder.

Implements gradient-weighted class activation mapping to highlight regions
of the medical image that are most relevant to the model's answer.

Math behind Grad-CAM:
====================
For a target class c, the Grad-CAM heatmap is:
    alpha_k = (1/Z) * Sum_i Sum_j dy^c / dA^k_{ij}   (importance of feature map k)
    L^c = ReLU(Sum_k alpha_k * A^k)                   (weighted combination)

Where A^k is the k-th feature map from the last convolutional/attention layer,
and y^c is the score for class c. The ReLU ensures we only show positive
contributions (features that *increase* the class score).

For medical images, Grad-CAM highlights diagnostically relevant regions —
e.g., a lung nodule location for "Is there a nodule?" questions.
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """Grad-CAM for medical image attention visualization.

    Generates heatmaps showing which parts of the medical image the model
    focuses on when answering a clinical question.

    Args:
        model: MedVQAModel instance (must have vision_encoder with hooks).
        target_layer: Transformer layer index for hooking (None = last).
        image_size: Input image size.
    """

    def __init__(
        self, model: torch.nn.Module, target_layer: Optional[int] = None, image_size: int = 224
    ):
        self.model = model
        self.image_size = image_size
        self.activations = None
        self.gradients = None
        self._tokenizer = None

        # Register hooks on the vision encoder
        self._register_hooks(target_layer)

    def _get_tokenizer(self):
        """Lazy-load a tokenizer for target-answer guided Grad-CAM."""
        if self._tokenizer is not None:
            return self._tokenizer

        model_name = None
        if hasattr(self.model, "language_model") and hasattr(
            self.model.language_model, "model_name"
        ):
            model_name = self.model.language_model.model_name

        if model_name is None:
            raise RuntimeError("Cannot resolve tokenizer model name for Grad-CAM")

        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        return self._tokenizer

    def _register_hooks(self, target_layer: Optional[int] = None):
        """Register forward and backward hooks on vision encoder.

        Args:
            target_layer: Which transformer layer to hook.
        """
        vision_encoder = self.model.vision_encoder.encoder

        # Find the last transformer layer
        if hasattr(vision_encoder, "vision_model"):
            layers = vision_encoder.vision_model.encoder.layers
        elif hasattr(vision_encoder, "encoder"):
            layers = vision_encoder.encoder.layer
        else:
            raise AttributeError("Cannot find transformer layers for Grad-CAM")

        if target_layer is None:
            target_layer = len(layers) - 1

        target_module = layers[target_layer]

        def forward_hook(module, input, output):
            # Extract the hidden states (activations before the next layer)
            if isinstance(output, tuple):
                self.activations = output[0].detach()
            else:
                self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            # Extract gradients
            self.gradients = grad_output[0].detach()

        self.forward_handle = target_module.register_forward_hook(forward_hook)
        self.backward_handle = target_module.register_full_backward_hook(backward_hook)

    def remove_hooks(self):
        """Remove registered hooks."""
        if hasattr(self, "forward_handle"):
            self.forward_handle.remove()
        if hasattr(self, "backward_handle"):
            self.backward_handle.remove()

    def compute_heatmap(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_answer: Optional[str] = None,
    ) -> np.ndarray:
        """Compute Grad-CAM heatmap for an image-question pair.

        Args:
            image: Preprocessed image tensor (1, 3, H, W).
            input_ids: Tokenized question (1, T).
            attention_mask: Attention mask (1, T).
            target_answer: Optional target answer for guided Grad-CAM.
                          If None, uses the predicted answer.

        Returns:
            Heatmap array (H, W) in range [0, 1] as float32.
        """
        self.model.eval()
        image = image.to(self.model.vision_encoder.encoder.device)
        input_ids = input_ids.to(self.model.vision_encoder.encoder.device)
        attention_mask = attention_mask.to(self.model.vision_encoder.encoder.device)

        # Forward pass
        outputs = self.model(images=image, input_ids=input_ids, attention_mask=attention_mask)

        # Get the logits and compute gradients
        logits = outputs["logits"]

        # We need to compute gradients w.r.t. the answer logits
        # For simplicity, use the max logit as the target
        if target_answer is not None:
            tokenizer = self._get_tokenizer()
            token_ids = tokenizer(target_answer, add_special_tokens=False, return_tensors="pt")[
                "input_ids"
            ].squeeze(0)

            if token_ids.numel() == 0:
                raise ValueError("Target answer tokenization produced no tokens")

            token_ids = token_ids.to(logits.device)
            target_logits = logits[:, -1, :]  # (1, vocab_size)
            target = target_logits.gather(1, token_ids[:1].unsqueeze(0)).sum()
        else:
            # Use the most likely next token
            target_logits = logits[:, -1, :]  # (1, vocab_size)
            target = target_logits.max(dim=-1)[0].sum()

        # Backward pass
        self.model.zero_grad()
        target.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("No activations/gradients captured. Check hooks.")

        # Global average pooling of gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # (1, 1, H', W')
        cam = F.relu(cam)  # Only positive contributions

        # Resize to image size
        cam = F.interpolate(
            cam, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
        )

        # Normalize to [0, 1]
        heatmap = cam.squeeze().cpu().numpy()
        if heatmap.max() > heatmap.min():
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        else:
            heatmap = np.zeros_like(heatmap)

        return heatmap.astype(np.float32)

    @staticmethod
    def overlay_on_image(
        image: Image.Image,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET,
    ) -> Image.Image:
        """Overlay heatmap on original image.

        Args:
            image: Original PIL Image (will be resized to heatmap size).
            heatmap: Heatmap array (H, W) in [0, 1].
            alpha: Overlay transparency (0 = no overlay, 1 = heatmap only).
            colormap: OpenCV colormap to use.

        Returns:
            PIL Image with heatmap overlay.
        """
        # Resize image to match heatmap
        image_resized = image.resize((heatmap.shape[1], heatmap.shape[0]))

        # Convert image to numpy
        img_np = np.array(image_resized)

        # Apply colormap to heatmap
        heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Overlay
        overlayed = (1 - alpha) * img_np + alpha * heatmap_colored
        overlayed = overlayed.astype(np.uint8)

        return Image.fromarray(overlayed)

    def generate_gradcam_report(
        self,
        dataset: torch.utils.data.Dataset,
        output_dir: str = "experiments/gradcam",
        n_samples: int = 20,
        device: str = "cuda",
    ) -> list[str]:
        """Generate Grad-CAM overlays for a set of samples.

        Args:
            dataset: Dataset to sample from.
            output_dir: Directory to save overlays.
            n_samples: Number of samples to process.
            device: Device to run on.

        Returns:
            List of saved file paths.
        """
        import random

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))

        for idx in indices:
            sample = dataset[idx]

            image = sample["image"].unsqueeze(0).to(device)
            input_ids = sample["input_ids"].unsqueeze(0).to(device)
            attention_mask = sample["attention_mask"].unsqueeze(0).to(device)

            # Compute heatmap
            heatmap = self.compute_heatmap(image, input_ids, attention_mask)

            # Load original image for overlay
            orig_image = Image.open(sample["image_path"]).convert("RGB")

            # Create overlay
            overlay = self.overlay_on_image(orig_image, heatmap)

            # Save
            save_name = f"gradcam_{idx:04d}.png"
            overlay.save(str(output_path / save_name))
            saved_paths.append(str(output_path / save_name))

            # Save side-by-side: original, heatmap, overlay
            side_by_side = self._create_side_by_side(orig_image, heatmap, overlay)
            side_path = output_path / f"comparison_{idx:04d}.png"
            side_by_side.save(str(side_path))

        print(f"✅ Grad-CAM report saved to {output_dir} ({len(saved_paths)} samples)")
        return saved_paths

    @staticmethod
    def _create_side_by_side(
        original: Image.Image, heatmap: np.ndarray, overlay: Image.Image
    ) -> Image.Image:
        """Create a side-by-side comparison image.

        Args:
            original: Original PIL Image.
            heatmap: Heatmap array (H, W).
            overlay: PIL Image with overlay.

        Returns:
            Side-by-side PIL Image (3 panels).
        """
        w, h = original.size
        panel = Image.new("RGB", (w * 3 + 20, h), (255, 255, 255))

        # Resize heatmap to match
        heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8), mode="L")
        heatmap_img = heatmap_img.convert("RGB")
        heatmap_colored = Image.fromarray(
            cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        )

        panel.paste(original, (0, 0))
        panel.paste(heatmap_colored, (w + 10, 0))
        panel.paste(overlay, (2 * w + 20, 0))

        return panel
