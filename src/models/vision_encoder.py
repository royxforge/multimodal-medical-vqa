"""Vision encoder wrapper supporting BioViL-T with CLIP fallback.

Exposes patch embeddings, CLS token, and last hidden states with forward
hook registration for Grad-CAM. Supports progressive unfreezing.

Reference: BioViL-T (Bannur et al., 2023) — medical vision transformer
pretrained on chest X-rays. Falls back to CLIP ViT-L/14 if BioViL-T
is unavailable (e.g., access restrictions).
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel


class BioViLTEncoder(nn.Module):
    """Vision encoder wrapping BioViL-T or CLIP ViT-L/14.

    The encoder projects visual features from the vision transformer's hidden
    dimension to the LLM's hidden dimension (4096 for Mistral-7B).

    Features:
    - Gradient checkpointing for memory efficiency
    - Forward hook registration for Grad-CAM
    - Progressive unfreezing of top-k layers

    Args:
        model_name: HuggingFace model name (default: microsoft/BioViL-T).
        fallback_model_name: Fallback if primary is unavailable.
        hidden_size: Vision encoder's hidden dimension.
        projection_dim: Target dimension for LLM projection.
        freeze_encoder: Whether to freeze the encoder initially.
    """

    def __init__(
        self,
        model_name: str = "microsoft/BioViL-T",
        fallback_model_name: str = "openai/clip-vit-large-patch14",
        hidden_size: int = 768,
        projection_dim: int = 4096,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.hidden_size = hidden_size
        self.projection_dim = projection_dim
        self._gradcam_hooks = []
        self._gradcam_activations = None
        self._gradcam_gradients = None

        # Load vision encoder
        try:
            self.encoder = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,  # BioViL-T requires custom code
            )
            print(f"[OK] Loaded vision encoder from {model_name}")
        except Exception as e:
            print(f"[WARN] BioViL-T load failed ({e}). Falling back to CLIP ViT-L/14.")
            # Use AutoModel instead of CLIPVisionModel directly to avoid
            # CLIPVisionConfig vs CLIPConfig compatibility issues on newer
            # transformers versions. The forward() method handles CLIPModel
            # by using the vision_model submodule directly.
            self.encoder = AutoModel.from_pretrained(fallback_model_name)
            self.hidden_size = 1024  # CLIP ViT-L/14 hidden size
            print(f"[OK] Loaded CLIP ViT-L/14 from {fallback_model_name}")

        # Projection head: vision dim → LLM dim
        self.projection = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, projection_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Freeze encoder if needed
        if freeze_encoder:
            self._freeze_all()
        else:
            self._freeze_all()  # Freeze by default, then unfreeze specific layers
            self.unfreeze_top_k_layers(0)

    def _freeze_all(self):
        """Freeze all encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("[LOCK] Vision encoder frozen")

    def unfreeze_top_k_layers(self, k: int = 4):
        """Unfreeze the last k transformer layers for fine-tuning.

        Args:
            k: Number of top layers to unfreeze (0 = none).
        """
        if k <= 0:
            return

        # Determine number of layers in the encoder
        if hasattr(self.encoder, "vision_model"):
            # CLIP case
            layers = self.encoder.vision_model.encoder.layers
        elif hasattr(self.encoder, "encoder"):
            # BioViL-T case
            layers = self.encoder.encoder.layer
        else:
            print("⚠️ Could not unfreeze layers: unknown encoder structure")
            return

        n_layers = len(layers)
        for i in range(n_layers - k, n_layers):
            for param in layers[i].parameters():
                param.requires_grad = True

        # Always unfreeze the projection head
        for param in self.projection.parameters():
            param.requires_grad = True

        print(f"[UNFROZE] Last {k}/{n_layers} vision encoder layers unfrozen")

    def get_trainable_params(self) -> int:
        """Count trainable parameters.

        Returns:
            Number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, pixel_values: torch.Tensor, output_hidden_states: bool = True) -> dict:
        """Forward pass through the vision encoder.

        Args:
            pixel_values: Image tensor (B, 3, H, W).
            output_hidden_states: Whether to return all hidden states.

        Returns:
            Dict with:
                - cls_embedding: CLS token embedding (B, D)
                - patch_embeddings: All patch token embeddings (B, N, D)
                - last_hidden_state: Full sequence (B, N+1, D)
                - hidden_states: Tuple of hidden states (if requested)
        """
        # Handle different model outputs
        if hasattr(self.encoder, "vision_model"):
            # CLIP model path — use the vision submodel directly (avoids
            # CLIPModel's requirement for input_ids along with pixel_values)
            vision_model = self.encoder.vision_model
            outputs = vision_model(
                pixel_values=pixel_values, output_hidden_states=output_hidden_states
            )
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token
            patch_embeddings = outputs.last_hidden_state[:, 1:, :]  # Patch tokens
            last_hidden_state = outputs.last_hidden_state
            hidden_states = outputs.hidden_states if output_hidden_states else None

        else:
            # BioViL-T / base HF model path
            outputs = self.encoder(
                pixel_values=pixel_values, output_hidden_states=output_hidden_states
            )
            last_hidden_state = outputs.last_hidden_state
            cls_embedding = last_hidden_state[:, 0, :]
            patch_embeddings = last_hidden_state[:, 1:, :]
            hidden_states = outputs.hidden_states if output_hidden_states else None

        # Project to LLM dimension
        projected_cls = self.projection(cls_embedding)
        projected_patches = self.projection(patch_embeddings)

        return {
            "cls_embedding": projected_cls,
            "patch_embeddings": projected_patches,
            "last_hidden_state": last_hidden_state,
            "hidden_states": hidden_states,
        }

    def register_gradcam_hooks(self, target_layer: Optional[int] = None):
        """Register forward and backward hooks for Grad-CAM.

        Hooks capture activations and gradients from the last transformer
        block's output for Grad-CAM heatmap computation.

        Args:
            target_layer: Which transformer layer to hook (None = last).
        """
        self._clear_gradcam_hooks()

        # Find the last transformer layer
        if hasattr(self.encoder, "vision_model"):
            layers = self.encoder.vision_model.encoder.layers
        elif hasattr(self.encoder, "encoder"):
            layers = self.encoder.encoder.layer
        else:
            raise AttributeError("Could not locate transformer layers for Grad-CAM")

        if target_layer is None:
            target_layer = len(layers) - 1

        target_module = layers[target_layer]

        def forward_hook(module, input, output):
            self._gradcam_activations = (
                output[0].detach() if isinstance(output, tuple) else output.detach()
            )

        def backward_hook(module, grad_input, grad_output):
            self._gradcam_gradients = grad_output[0].detach()

        self._gradcam_hooks.append(target_module.register_forward_hook(forward_hook))
        self._gradcam_hooks.append(target_module.register_full_backward_hook(backward_hook))

        print(f"[OK] Grad-CAM hooks registered on layer {target_layer}")

    def _clear_gradcam_hooks(self):
        """Clear all registered Grad-CAM hooks."""
        for hook in self._gradcam_hooks:
            hook.remove()
        self._gradcam_hooks = []
        self._gradcam_activations = None
        self._gradcam_gradients = None

    def get_gradcam_activations(self):
        """Get captured activations and gradients for Grad-CAM.

        Returns:
            Tuple of (activations, gradients) or (None, None).
        """
        return self._gradcam_activations, self._gradcam_gradients
