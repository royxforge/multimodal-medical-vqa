"""Mistral-7B language model with QLoRA fine-tuning.

Uses 4-bit NF4 quantization via bitsandbytes for memory-efficient training
on Colab T4 (15GB VRAM). LoRA adapters enable fine-tuning with < 50M
trainable parameters.

Reference: QLoRA (Dettmers et al., 2023) — Efficient finetuning of quantized LLMs.
"""

from typing import Optional

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


class MistralQLoRA(nn.Module):
    """Mistral-7B wrapped with QLoRA for memory-efficient fine-tuning.

    The model is loaded in 4-bit NF4 quantization and wrapped with LoRA
    adapters targeting the query, key, value, and output projections.

    Args:
        model_name: HuggingFace model name.
        load_in_4bit: Whether to use 4-bit quantization.
        bnb_config: BitsAndBytesConfig dict for quantization.
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha scaling.
        lora_dropout: LoRA dropout rate.
        lora_target_modules: Which modules to apply LoRA to.
        gradient_checkpointing: Enable gradient checkpointing for memory.
    """

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
        load_in_4bit: bool = True,
        bnb_config: Optional[dict] = None,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[list[str]] = None,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha

        if lora_target_modules is None:
            lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

        # Configure 4-bit quantization
        if bnb_config is None:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        # Load model with quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            use_cache=not gradient_checkpointing,
        )

        # Set up for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        self.model.config.use_cache = not gradient_checkpointing

        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Apply LoRA
        self._apply_lora(lora_target_modules, lora_dropout)

        # Store hidden size
        self.hidden_size = self.model.config.hidden_size  # 4096 for Mistral-7B

        print(f"✅ Loaded {model_name} with QLoRA (r={lora_r}, alpha={lora_alpha})")

    def _apply_lora(self, target_modules: list[str], dropout: float):
        """Apply LoRA adapters to specified modules.

        Args:
            target_modules: Module names to apply LoRA to.
            dropout: LoRA dropout rate.
        """
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=target_modules,
            lora_dropout=dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_config)

    def get_trainable_params(self) -> dict[str, int]:
        """Get parameter efficiency statistics.

        Returns:
            Dict with total, trainable, and percentage of trainable params.
        """
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            "total_params": total,
            "trainable_params": trainable,
            "trainable_pct": 100.0 * trainable / total,
        }

    def get_input_embeddings(self) -> nn.Embedding:
        """Get the input embedding layer for cross-attention fusion.

        Returns:
            The model's input embedding layer.
        """
        return self.model.get_input_embeddings()

    def prepare_inputs_for_generation(self, *args, **kwargs):
        """Delegate to the underlying model's prepare_inputs_for_generation."""
        return self.model.prepare_inputs_for_generation(*args, **kwargs)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> dict:
        """Forward pass through the language model.

        Args:
            input_ids: Token IDs (B, seq_len).
            attention_mask: Attention mask (B, seq_len).
            labels: Target labels for LM loss (B, seq_len).
            inputs_embeds: Embedded inputs (alternative to input_ids).
            **kwargs: Additional arguments for the HF model.

        Returns:
            Model output with logits, loss, etc.
        """
        # Ensure we use the same dtype as the quantized model
        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds.to(self.model.dtype)

        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            **kwargs,
        )
        return output

    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        max_new_tokens: int = 64,
        num_beams: int = 4,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        **kwargs,
    ) -> torch.LongTensor:
        """Generate answer tokens from question tokens or fused embeddings.

        Supports both input_ids (text-only) and inputs_embeds (with visual
        information fused in) for multimodal generation.

        Args:
            input_ids: Tokenized question (B, seq_len). Optional if inputs_embeds provided.
            attention_mask: Attention mask (B, seq_len).
            inputs_embeds: Fused visual+text embeddings (B, seq_len, D). If provided,
                          overrides input_ids for the embedding lookup.
            max_new_tokens: Maximum tokens to generate.
            num_beams: Beam search width.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            repetition_penalty: Penalty for repeated tokens.

        Returns:
            Generated token IDs (B, gen_len).
        """
        with torch.no_grad():
            # Build initial inputs for generation
            if inputs_embeds is not None:
                # When using custom embeddings, we still need input_ids as a
                # reference for the model's generate() method.
                # We create a dummy input_ids of the right length.
                batch_size = inputs_embeds.shape[0]
                seq_len = inputs_embeds.shape[1]
                # Use the first token's ID repeated as a placeholder
                placeholder_ids = torch.full(
                    (batch_size, seq_len),
                    self.model.config.eos_token_id or 0,
                    device=inputs_embeds.device,
                    dtype=torch.long,
                )
                input_ids = placeholder_ids

            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.model.config.eos_token_id,
                **kwargs,
            )
        return outputs

    def merge_and_unload(self):
        """Merge LoRA weights and unload the base model for inference.

        Call this after training to get a single model without adapter overhead.
        """
        return self.model.merge_and_unload()

    def print_trainable_params(self):
        """Print a formatted summary of trainable parameters."""
        stats = self.get_trainable_params()
        print("=" * 50)
        print("Parameter Efficiency (Mistral-7B QLoRA)")
        print("=" * 50)
        print(f"  Total parameters:     {stats['total_params'] / 1e6:.2f}M")
        print(f"  Trainable parameters: {stats['trainable_params'] / 1e6:.2f}M")
        print(f"  Trainable percentage: {stats['trainable_pct']:.2f}%")
        print("=" * 50)
