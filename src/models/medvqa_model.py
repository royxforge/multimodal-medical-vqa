"""End-to-end MedVQA model combining vision encoder, fusion, and LLM.

Architecture:
    Image ──► BioViL-T ──► Visual Tokens ──┐
                                            ├──► Cross-Attn Fusion ──► Mistral-7B (QLoRA) ──► Answer
    Question ──► Tokenizer ──► Embeddings ──┘

All trainable parameters should be under 50M (LoRA adapters only).
"""

from typing import Optional

import torch
import torch.nn as nn

from .fusion import CrossAttentionFusion
from .language_model import MistralQLoRA
from .vision_encoder import BioViLTEncoder


class MedVQAModel(nn.Module):
    """End-to-end Multimodal Medical VQA model.

    Combines a vision encoder, cross-attention fusion, and a language model
    with QLoRA fine-tuning.

    Args:
        vision_encoder: BioViLTEncoder instance.
        language_model: MistralQLoRA instance.
        fusion: CrossAttentionFusion instance.
        freeze_vision: Whether to freeze vision encoder.
        num_beams: Beam search width for generation.
    """

    def __init__(
        self,
        vision_encoder: BioViLTEncoder,
        language_model: MistralQLoRA,
        fusion: CrossAttentionFusion,
        freeze_vision: bool = True,
        num_beams: int = 4,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.language_model = language_model
        self.fusion = fusion
        self.num_beams = num_beams

        # Freeze vision encoder if specified
        if freeze_vision:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

        # Binary classification head for yes/no questions
        # Takes the fused CLS embedding and produces 2 logits (no, yes)
        self.yesno_head = nn.Sequential(
            nn.LayerNorm(self.vision_encoder.projection_dim),
            nn.Linear(self.vision_encoder.projection_dim, 2),
        )

    def get_trainable_params(self) -> dict[str, int]:
        """Get parameter efficiency statistics across all components.

        Returns:
            Dict with detailed trainable parameter counts.
        """
        vision_trainable = sum(
            p.numel() for p in self.vision_encoder.parameters() if p.requires_grad
        )
        lm_stats = self.language_model.get_trainable_params()
        fusion_trainable = sum(p.numel() for p in self.fusion.parameters() if p.requires_grad)

        total_trainable = vision_trainable + lm_stats["trainable_params"] + fusion_trainable
        total_all = (
            sum(p.numel() for p in self.vision_encoder.parameters())
            + lm_stats["total_params"]
            + sum(p.numel() for p in self.fusion.parameters())
        )

        return {
            "vision_trainable": vision_trainable,
            "lm_trainable": lm_stats["trainable_params"],
            "fusion_trainable": fusion_trainable,
            "total_trainable": total_trainable,
            "total_params": total_all,
            "trainable_pct": 100.0 * total_trainable / total_all if total_all > 0 else 0,
        }

    def print_param_summary(self):
        """Print formatted parameter summary."""
        stats = self.get_trainable_params()
        print("=" * 55)
        print("MedVQA Model — Parameter Summary")
        print("=" * 55)
        print("  Component            │ Trainable    │ Total")
        print("  ─────────────────────┼──────────────┼──────────")
        print(
            f"  Vision Encoder       │ {stats['vision_trainable'] / 1e6:>8.2f}M │ {stats['total_params'] / 1e6:.2f}M"
        )
        print(
            f"  Cross-Attn Fusion    │ {stats['fusion_trainable'] / 1e6:>8.2f}M │ {stats['total_params'] / 1e6:.2f}M"
        )
        print(
            f"  Mistral-7B (QLoRA)   │ {stats['lm_trainable'] / 1e6:>8.2f}M │ {stats['total_params'] / 1e6:.2f}M"
        )
        print("  ─────────────────────┼──────────────┼──────────")
        print(
            f"  Total trainable: {stats['total_trainable'] / 1e6:.2f}M / {stats['total_params'] / 1e6:.2f}M "
            f"({stats['trainable_pct']:.2f}%)"
        )
        print("=" * 55)

        assert stats["total_trainable"] < 50e6, (
            f"Trainable params ({stats['total_trainable'] / 1e6:.2f}M) " f"exceeds 50M limit!"
        )
        print("✅ Trainable parameters under 50M threshold")

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        return_visual_features: bool = False,
        return_text_features: bool = False,
        vision_outputs: Optional[dict[str, torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through the full MedVQA model.

        Args:
            images: Batch of images (B, 3, H, W).
            input_ids: Tokenized question IDs (B, T).
            attention_mask: Question attention mask (B, T).
            labels: Target answer token IDs (B, A) for loss computation.
            return_visual_features: Whether to return visual features.
            return_text_features: Whether to return text features.

        Returns:
            Dict with:
                - logits: Answer token logits (B, A, vocab_size)
                - loss: LM loss (if labels provided)
                - visual_features: Projected visual CLS embeddings (if requested)
                - text_features: Text embeddings (if requested)
        """
        # 1. Encode image (or reuse cached vision outputs)
        if vision_outputs is None:
            vision_outputs = self.vision_encoder(images)
        visual_patches = vision_outputs["patch_embeddings"]  # (B, V, D)
        visual_cls = vision_outputs["cls_embedding"]  # (B, D)

        # 2. Get question token embeddings from LLM
        text_embeddings = self.language_model.get_input_embeddings()(input_ids)  # (B, T, D)

        # 3. Cross-attention fusion
        fused_embeddings = self.fusion(
            text_embeddings=text_embeddings, visual_tokens=visual_patches, text_mask=attention_mask
        )

        # 4. Prepend visual CLS token as a special visual token
        # This gives the LLM direct access to global image features
        visual_cls_expanded = visual_cls.unsqueeze(1)  # (B, 1, D)
        fused_with_cls = torch.cat([visual_cls_expanded, fused_embeddings], dim=1)  # (B, T+1, D)

        # Adjust attention mask for the prepended visual token
        expanded_mask = torch.cat(
            [
                torch.ones(
                    attention_mask.shape[0],
                    1,
                    device=attention_mask.device,
                    dtype=attention_mask.dtype,
                ),
                attention_mask,
            ],
            dim=1,
        )

        # 5. Pass through LLM
        # When labels are provided, we shift them to account for the prepended visual token
        if labels is not None:
            # Pad labels with -100 (ignored in loss) for the visual token position
            expanded_labels = torch.cat(
                [
                    torch.full(
                        (labels.shape[0], 1), -100, device=labels.device, dtype=labels.dtype
                    ),
                    labels,
                ],
                dim=1,
            )
        else:
            expanded_labels = None

        lm_outputs = self.language_model(
            inputs_embeds=fused_with_cls, attention_mask=expanded_mask, labels=expanded_labels
        )

        # 6. Binary classification head for yes/no questions
        # The yes/no head uses the fused CLS embedding (visual + attended text)
        yesno_logits = self.yesno_head(visual_cls)  # (B, 2)

        result = {
            "logits": lm_outputs["logits"],
            "loss": lm_outputs.get("loss", None),
            "yesno_logits": yesno_logits,
        }

        if return_visual_features:
            result["visual_features"] = visual_cls

        if return_text_features:
            result["text_features"] = fused_embeddings

        return result

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor,
        max_new_tokens: int = 64,
        num_beams: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        vision_outputs: Optional[dict[str, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.LongTensor:
        """Generate answer for a batch of image-question pairs.

        This method properly passes visual information to the LLM via
        inputs_embeds, ensuring the generated answer conditions on the
        medical image, not just the question text.

        Args:
            images: Batch of images (B, 3, H, W).
            input_ids: Tokenized question IDs (B, T).
            attention_mask: Question attention mask (B, T).
            max_new_tokens: Maximum tokens to generate.
            num_beams: Beam search width (defaults to self.num_beams).
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            **kwargs: Additional generation kwargs.

        Returns:
            Generated token IDs (B, gen_len).
        """
        if num_beams is None:
            num_beams = self.num_beams

        # 1. Encode image (or reuse cached vision outputs)
        if vision_outputs is None:
            vision_outputs = self.vision_encoder(images)
        visual_patches = vision_outputs["patch_embeddings"]
        visual_cls = vision_outputs["cls_embedding"]

        # 2. Get question token embeddings
        text_embeddings = self.language_model.get_input_embeddings()(input_ids)

        # 3. Cross-attention fusion: inject visual info into text embeddings
        fused_embeddings = self.fusion(
            text_embeddings=text_embeddings, visual_tokens=visual_patches, text_mask=attention_mask
        )

        # 4. Prepend visual CLS token as a special visual token
        visual_cls_expanded = visual_cls.unsqueeze(1)  # (B, 1, D)
        fused_with_cls = torch.cat([visual_cls_expanded, fused_embeddings], dim=1)  # (B, T+1, D)

        # Adjust attention mask for the prepended visual token
        expanded_mask = torch.cat(
            [
                torch.ones(
                    attention_mask.shape[0],
                    1,
                    device=attention_mask.device,
                    dtype=attention_mask.dtype,
                ),
                attention_mask,
            ],
            dim=1,
        )

        # 5. Generate tokens using the LLM with custom fused embeddings
        # We pass inputs_embeds so the model uses visual+text fused representations
        return self.language_model.generate(
            inputs_embeds=fused_with_cls,
            attention_mask=expanded_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.language_model.model.gradient_checkpointing_enable()
        print("✅ Gradient checkpointing enabled")
