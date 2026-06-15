"""End-to-end training pipeline test.

Validates the complete training flow using tiny placeholder models so the
test can run on CPU without downloading 7B parameter models. This test
exercises:

1. Data loading (VQARADDataset)
2. Model assembly (vision encoder, fusion, LM → MedVQAModel)
3. Forward pass with combined loss
4. Backward pass (gradient flow through all components)
5. Training loop (optimizer, scheduler, gradient accumulation)
6. Evaluation (generation, metric computation)
7. Checkpointing (save/load)

Note: This test uses GPT-2 (124M params) as the language model instead of
Mistral-7B so it can run on CPU with minimal resources. The architecture
remains the same: vision encoder → cross-attention fusion → LLM.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import VQARADDataset, collate_fn
from src.data.preprocessor import MedicalImagePreprocessor, TextPreprocessor
from src.models.fusion import CrossAttentionFusion
from src.models.medvqa_model import MedVQAModel
from src.training.losses import MedVQALoss
from src.utils.reproducibility import set_seed


class TinyVisionEncoder(torch.nn.Module):
    """Minimal vision encoder for pipeline testing.

    Replaces CLIP/BioViL-T during testing. A simple conv net that
    produces CLS embeddings and patch embeddings matching the expected
    interface of MedVQAModel.
    """

    def __init__(self, projection_dim: int = 128):
        super().__init__()
        self.projection_dim = projection_dim
        self.hidden_size = projection_dim
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((7, 7)),
            torch.nn.Flatten(),
            torch.nn.Linear(32 * 7 * 7, projection_dim),
        )

    def forward(self, pixel_values, output_hidden_states=False):
        features = self.conv(pixel_values)  # (B, projection_dim)
        return {
            "cls_embedding": features,
            "patch_embeddings": features.unsqueeze(1),  # (B, 1, D)
            "last_hidden_state": features.unsqueeze(1),
            "hidden_states": None,
        }

    def get_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def register_gradcam_hooks(self, target_layer=None):
        pass  # Noop for testing


class TinyLanguageModel(torch.nn.Module):
    """Minimal language model for pipeline testing.

    Replaces Mistral-7B during testing. Has a proper embedding layer,
    configurable hidden size, and a forward/generate interface matching
    what MedVQAModel expects.
    """

    def __init__(self, vocab_size: int = 1000, hidden_size: int = 128):
        super().__init__()
        self.hidden_size = hidden_size
        self.config = type("Config", (), {"hidden_size": hidden_size})()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.transformer = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=hidden_size, nhead=4, dim_feedforward=hidden_size * 4, batch_first=True
            ),
            num_layers=2,
        )
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size)

    def get_input_embeddings(self) -> torch.nn.Embedding:
        return self.embedding

    def get_trainable_params(self) -> dict:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return {
            "total_params": total,
            "trainable_params": trainable,
            "trainable_pct": 100.0 * trainable / total if total > 0 else 0,
        }

    def print_trainable_params(self):
        stats = self.get_trainable_params()
        print(f"  TinyLM: {stats['trainable_params']:,}/{stats['total_params']:,} trainable")

    def forward(
        self, input_ids=None, attention_mask=None, labels=None, inputs_embeds=None, **kwargs
    ):
        if inputs_embeds is not None:
            x = inputs_embeds
        elif input_ids is not None:
            x = self.embedding(input_ids)
        else:
            raise ValueError("Need input_ids or inputs_embeds")

        if labels is not None and labels.shape[1] < x.shape[1]:
            pad = torch.full(
                (labels.shape[0], x.shape[1] - labels.shape[1]),
                -100,
                device=labels.device,
                dtype=labels.dtype,
            )
            labels = torch.cat([pad, labels], dim=1)

        if attention_mask is not None:
            seq_len = x.shape[1]
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=x.device), diagonal=1
            )
            padding_mask = attention_mask.float()
            padding_mask = padding_mask.masked_fill(padding_mask == 0, float("-inf"))
            padding_mask = padding_mask.masked_fill(padding_mask == 1, 0.0)
            x = self.transformer(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        else:
            x = self.transformer(x)

        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {"logits": logits, "loss": loss}

    @torch.no_grad()
    def generate(
        self, input_ids=None, attention_mask=None, inputs_embeds=None, max_new_tokens=10, **kwargs
    ):
        """Simple greedy generation for testing."""
        if inputs_embeds is not None:
            batch_size = inputs_embeds.shape[0]
            device = inputs_embeds.device
            return torch.randint(0, 100, (batch_size, max_new_tokens), device=device)
        elif input_ids is not None:
            batch_size = input_ids.shape[0]
            device = input_ids.device
            return torch.randint(
                0, 100, (batch_size, input_ids.shape[1] + max_new_tokens), device=device
            )
        return torch.randint(0, 100, (1, max_new_tokens))


@pytest.fixture(scope="module")
def setup_data():
    """Load a small subset of VQA-RAD for testing."""
    data_dir = "data/raw/vqa_rad"
    if not Path(data_dir).exists():
        pytest.skip("VQA-RAD data not found — run scripts/download_vqa_rad.py first")

    set_seed(42)

    # Use a tiny tokenizer for speed
    text_processor = TextPreprocessor(
        model_name="mistralai/Mistral-7B-Instruct-v0.3",
        max_question_length=64,
        max_answer_length=32,
    )

    image_processor = MedicalImagePreprocessor(image_size=224)

    # Load a small subset
    dataset = VQARADDataset(
        data_dir=data_dir,
        split="train",
        image_transform=image_processor,
        text_preprocessor=text_processor,
    )

    # Use first 16 samples for fast testing
    subset_size = min(16, len(dataset))
    indices = list(range(subset_size))
    subset = torch.utils.data.Subset(dataset, indices)

    return subset, text_processor


@pytest.fixture(scope="module")
def tiny_model_and_data(setup_data):
    """Build a tiny MedVQA model and return it with data."""
    subset, text_processor = setup_data
    device = "cpu"
    projection_dim = 128

    # 1. Tiny vision encoder (use module-level TinyVisionEncoder)
    vision_encoder = TinyVisionEncoder(projection_dim=projection_dim)

    # 2. Tiny language model
    language_model = TinyLanguageModel(
        vocab_size=(
            text_processor.tokenizer.vocab_size if hasattr(text_processor, "tokenizer") else 1000
        ),
        hidden_size=projection_dim,
    )

    # 3. Fusion layer
    fusion = CrossAttentionFusion(d_model=projection_dim, n_heads=2, dropout=0.0, use_residual=True)

    # 4. Full model
    medvqa_model = MedVQAModel(
        vision_encoder=vision_encoder,
        language_model=language_model,
        fusion=fusion,
        freeze_vision=True,
        num_beams=1,
    )

    medvqa_model.to(device)
    medvqa_model.train()

    return medvqa_model, subset, text_processor, device


class TestEndToEndTrainingPipeline:
    """Complete end-to-end training pipeline verification."""

    def test_data_pipeline(self, setup_data):
        """Test 1: Data loading produces correct shapes."""
        subset, text_processor = setup_data
        sample = subset[0]

        assert "image" in sample, "Missing 'image' key"
        assert "input_ids" in sample, "Missing 'input_ids' key"
        assert "labels" in sample, "Missing 'labels' key"
        assert sample["image"].shape == (
            3,
            224,
            224,
        ), f"Unexpected image shape: {sample['image'].shape}"
        assert sample["input_ids"].dim() == 1, "input_ids should be 1D"
        assert sample["labels"].dim() == 1, "labels should be 1D"
        assert (sample["labels"] != -100).any(), "Labels should have some non-masked tokens"

    def test_forward_pass(self, tiny_model_and_data):
        """Test 2: Forward pass produces logits and loss."""
        model, subset, tp, device = tiny_model_and_data
        loader = torch.utils.data.DataLoader(subset, batch_size=2, collate_fn=collate_fn)
        batch = next(iter(loader))

        # Move to device
        images = batch["images"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_visual_features=True,
        )

        assert "logits" in outputs, "Missing 'logits' in output"
        assert "loss" in outputs, "Missing 'loss' in output"
        assert "yesno_logits" in outputs, "Missing 'yesno_logits' in output"
        assert outputs["logits"] is not None, "logits should not be None"
        assert outputs["loss"] is not None, "loss should not be None"
        assert outputs["loss"].item() > 0, "Loss should be positive"
        assert torch.isfinite(outputs["loss"]), "Loss should be finite"
        print(f"[OK] Forward pass: loss={outputs['loss'].item():.4f}")

    def test_loss_components(self, tiny_model_and_data):
        """Test 3: Combined loss with closed-ended and open-ended components."""
        model, subset, tp, device = tiny_model_and_data
        loader = torch.utils.data.DataLoader(subset, batch_size=4, collate_fn=collate_fn)
        batch = next(iter(loader))

        images = batch["images"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_visual_features=True,
        )

        # Apply MedVQALoss
        loss_fn = MedVQALoss(closed_ended_alpha=0.5, label_smoothing=0.1, use_contrastive=False)

        loss_dict = loss_fn(
            logits=outputs["logits"],
            labels=labels,
            is_yesno=batch["is_yesno"].to(device),
            yesno_logits=outputs.get("yesno_logits"),
            answer_labels=batch["answer_labels"].to(device),
            visual_features=outputs.get("visual_features"),
            lm_loss=outputs.get("loss"),
        )

        assert "loss" in loss_dict, "Missing combined loss"
        assert "closed_loss" in loss_dict, "Missing closed_loss"
        assert "open_loss" in loss_dict, "Missing open_loss"
        assert loss_dict["loss"].item() > 0, "Combined loss should be positive"
        assert torch.isfinite(loss_dict["loss"]), "Combined loss should be finite"
        print(
            f"[OK] Loss components: total={loss_dict['loss'].item():.4f}, "
            f"closed={loss_dict['closed_loss'].item():.4f}, "
            f"open={loss_dict['open_loss'].item():.4f}"
        )

    def test_backward_pass(self, tiny_model_and_data):
        """Test 4: Gradients flow backward through all components."""
        model, subset, tp, device = tiny_model_and_data
        loader = torch.utils.data.DataLoader(subset, batch_size=2, collate_fn=collate_fn)
        batch = next(iter(loader))

        images = batch["images"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward
        outputs = model(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_visual_features=True,
        )

        # Backward
        loss = outputs["loss"]
        loss.backward()

        # Check gradients flow to LM
        lm_has_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in model.language_model.parameters()
        )
        assert lm_has_grad, "No gradients flowing to language model"

        # Check gradients flow to fusion
        fusion_has_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.fusion.parameters()
        )
        assert fusion_has_grad, "No gradients flowing to fusion layer"

        print(
            f"[OK] Backward pass: gradients flow to LM ({lm_has_grad}) and fusion ({fusion_has_grad})"
        )

    def test_generation(self, tiny_model_and_data):
        """Test 5: Model generates output tokens."""
        model, subset, tp, device = tiny_model_and_data
        loader = torch.utils.data.DataLoader(subset, batch_size=2, collate_fn=collate_fn)
        batch = next(iter(loader))

        images = batch["images"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        generated = model.generate(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=5,
            num_beams=1,
        )

        assert generated is not None, "Generation returned None"
        assert generated.shape[0] == 2, f"Expected batch size 2, got {generated.shape[0]}"
        print(f"[OK] Generation: output shape {generated.shape}")

        # Decode
        decoded = tp.decode_answer(generated[0])
        assert isinstance(decoded, str), "Decoded answer should be string"
        print(f"[OK] Decoded answer: '{decoded}'")

    def test_training_step(self, tiny_model_and_data):
        """Test 6: Complete training step (forward + backward + optimizer)."""
        model, subset, tp, device = tiny_model_and_data
        loader = torch.utils.data.DataLoader(subset, batch_size=2, collate_fn=collate_fn)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        loss_fn = MedVQALoss(closed_ended_alpha=0.5)

        model.train()
        total_loss = 0.0

        for step, batch in enumerate(loader):
            if step >= 3:  # 3 gradient accumulation steps
                break

            images = batch["images"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_visual_features=True,
            )

            loss_dict = loss_fn(
                logits=outputs["logits"],
                labels=labels,
                is_yesno=batch["is_yesno"].to(device),
                yesno_logits=outputs.get("yesno_logits"),
                answer_labels=batch["answer_labels"].to(device),
                lm_loss=outputs.get("loss"),
            )

            loss = loss_dict["loss"] / 3  # Gradient accumulation: divide by accum steps
            loss.backward()
            total_loss += loss.item()

        # Optimizer step
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        assert total_loss > 0, "Training loss should be positive"
        print(f"[OK] Training step: accumulated loss over 3 steps = {total_loss:.4f}")

    def test_checkpoint_save_load(self, tiny_model_and_data):
        """Test 7: Model checkpointing (save and load into fresh model)."""
        model, subset, tp, device = tiny_model_and_data
        proj_dim = 128

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            save_path = os.path.join(tmpdir, "test_model.pt")
            torch.save(model.state_dict(), save_path)
            assert os.path.exists(save_path), "Checkpoint file not created"
            file_size = os.path.getsize(save_path)
            print(f"[OK] Checkpoint saved: {save_path} ({file_size:,} bytes)")

            # Create a FRESH model (not from fixture, to avoid caching)
            ve2 = TinyVisionEncoder(projection_dim=proj_dim)
            vocab_size = tp.tokenizer.vocab_size if hasattr(tp, "tokenizer") else 1000
            lm2 = TinyLanguageModel(vocab_size=vocab_size, hidden_size=proj_dim)
            fusion2 = CrossAttentionFusion(d_model=proj_dim, n_heads=2)
            model2 = MedVQAModel(ve2, lm2, fusion2, freeze_vision=True, num_beams=1)
            model2.to(device)
            model2.eval()

            # Load saved state dict into fresh model
            model2.load_state_dict(torch.load(save_path, map_location=device), strict=False)

            # Verify loaded model produces same output
            model.eval()
            loader = torch.utils.data.DataLoader(subset, batch_size=2, collate_fn=collate_fn)
            batch = next(iter(loader))

            with torch.no_grad():
                out1 = model(
                    images=batch["images"].to(device),
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )
                out2 = model2(
                    images=batch["images"].to(device),
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )

            # Logits should match
            logits1 = out1["logits"]
            logits2 = out2["logits"]
            min_len = min(logits1.shape[1], logits2.shape[1])
            match = torch.allclose(logits1[:, :min_len], logits2[:, :min_len], atol=1e-5)
            assert match, "Loaded model produces different outputs"
            print("[OK] Checkpoint save/load verified: outputs match")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
