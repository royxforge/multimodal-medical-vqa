"""Tests for the data pipeline components.

Verifies dataset loading, preprocessing, augmentation, and collation.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.augmentation import MedicalAugmentationPipeline
from src.data.loader import VQARADDataset, collate_fn
from src.data.preprocessor import MedicalImagePreprocessor, TextPreprocessor


class TestMedicalImagePreprocessor:
    """Tests for medical image preprocessing."""

    def setup_method(self):
        self.preprocessor = MedicalImagePreprocessor(image_size=224)

    def test_preprocess_pil_image(self):
        """Test preprocessing of a synthetic medical image."""
        image = Image.new("RGB", (256, 256), (100, 100, 100))
        tensor = self.preprocessor(image)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 224, 224)
        assert tensor.dtype == torch.float32

    def test_preprocess_grayscale_image(self):
        """Test that grayscale images are converted to RGB."""
        image = Image.new("L", (256, 256), 100)
        tensor = self.preprocessor(image)
        assert tensor.shape == (3, 224, 224)

    def test_letterbox_resize(self):
        """Test aspect ratio preservation."""
        image = Image.new("RGB", (512, 256), (100, 100, 100))
        letterboxed = self.preprocessor.letterbox_resize(image)
        assert letterboxed.size == (224, 224)

    def test_normalization_range(self):
        """Test that normalized values are roughly in [-2, 2]."""
        image = Image.new("RGB", (224, 224), (200, 150, 100))
        tensor = self.preprocessor(image)
        assert tensor.min() >= -3.0
        assert tensor.max() <= 3.0


class TestTextPreprocessor:
    """Tests for text preprocessing."""

    def setup_method(self):
        self.preprocessor = TextPreprocessor(
            model_name="mistralai/Mistral-7B-Instruct-v0.3",
            max_question_length=128,
            max_answer_length=64,
        )

    def test_tokenize_question(self):
        """Test question tokenization."""
        question = "Is there a lung nodule in the upper left lobe?"
        encoded = self.preprocessor.tokenize_question(question)
        assert "input_ids" in encoded
        assert "attention_mask" in encoded
        assert encoded["input_ids"].dim() == 1
        assert encoded["input_ids"].shape[0] <= 128

    def test_tokenize_answer(self):
        """Test answer tokenization."""
        answer = "No evidence of lung nodule."
        encoded = self.preprocessor.tokenize_answer(answer)
        assert "input_ids" in encoded
        assert "attention_mask" in encoded

    def test_decode_answer(self):
        """Test answer decoding."""
        answer = "No evidence of lung nodule."
        encoded = self.preprocessor.tokenize_answer(answer)
        decoded = self.preprocessor.decode_answer(encoded["input_ids"])
        assert len(decoded) > 0

    def test_tokenize_qa_pair(self):
        """Test question-answer pair tokenization with label masking."""
        question = "Is there a lung nodule?"
        answer = "No."
        encoded = self.preprocessor.tokenize_qa_pair(question, answer)

        assert "input_ids" in encoded
        assert "attention_mask" in encoded
        assert "labels" in encoded
        assert (encoded["labels"] == -100).any()
        assert (encoded["labels"] != -100).any()

    def test_yes_no_detection(self):
        """Test yes/no question detection."""
        assert self.preprocessor.is_yes_no_question("Is there a nodule?")
        assert self.preprocessor.is_yes_no_question("Does the lung appear clear?")
        assert not self.preprocessor.is_yes_no_question("Describe the findings.")
        assert not self.preprocessor.is_yes_no_question("What size is the nodule?")


class TestMedicalAugmentation:
    """Tests for medical augmentation pipeline."""

    def setup_method(self):
        self.augmentation = MedicalAugmentationPipeline(image_size=224)

    def test_augment_numpy(self):
        """Test augmentation on numpy array."""
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        augmented = self.augmentation(image)
        assert augmented.shape == (224, 224, 3)

    def test_augment_pil(self):
        """Test augmentation on PIL Image."""
        image = Image.new("RGB", (224, 224), (100, 100, 100))
        augmented = self.augmentation.augment_pil(image)
        assert augmented.size == (224, 224)


class TestCollateFn:
    """Tests for DataLoader collate function."""

    def test_collate_basic(self):
        """Test collation of a batch."""
        batch = [
            {
                "image": torch.randn(3, 224, 224),
                "input_ids": torch.randint(0, 1000, (10,)),
                "attention_mask": torch.ones(10),
                "labels": torch.randint(0, 1000, (5,)),
                "label_attention_mask": torch.ones(5),
                "is_yesno": torch.tensor(1.0),
                "answer_label": torch.tensor(1),  # 1 = yes
                "question": "Test question?",
                "answer": "Yes",
                "image_path": "test.png",
            }
            for _ in range(4)
        ]

        batched = collate_fn(batch)
        assert batched["images"].shape == (4, 3, 224, 224)
        assert batched["input_ids"].shape[0] == 4
        assert batched["labels"].shape[0] == 4
        assert batched["is_yesno"].shape[0] == 4
        assert batched["answer_labels"].shape[0] == 4


@pytest.mark.skipif(
    not Path("data/raw/vqa_rad/VQA_RAD_Dataset.json").exists(),
    reason="VQA-RAD dataset not downloaded — run scripts/download_vqa_rad.py first",
)
class TestVQARADDataset:
    """Integration tests for VQARADDataset with real downloaded data.

    Verifies __len__ returns correct per-split counts and __getitem__
    returns tensors with the expected shapes and dtypes for both yes/no
    and open-ended questions.
    """

    DATA_DIR = "data/raw/vqa_rad"

    @pytest.fixture(autouse=True)
    def setup(self):
        self.image_processor = MedicalImagePreprocessor(image_size=224)
        self.text_processor = TextPreprocessor(
            model_name="mistralai/Mistral-7B-Instruct-v0.3",
            max_question_length=64,
            max_answer_length=32,
        )

    # ── __len__ tests ───────────────────────────────────────────────────

    def test_len_train_split(self):
        """__len__ returns expected train split size."""
        ds = VQARADDataset(
            data_dir=self.DATA_DIR,
            split="train",
            image_transform=self.image_processor,
            text_preprocessor=self.text_processor,
        )
        # VQA-RAD: ~80% of 1793 samples → ~1434 train
        assert len(ds) > 1000, f"Train set too small: {len(ds)}"
        assert len(ds) < 1600, f"Train set too large: {len(ds)}"

    def test_len_val_split(self):
        """__len__ returns expected val split size."""
        ds = VQARADDataset(
            data_dir=self.DATA_DIR,
            split="val",
            image_transform=self.image_processor,
            text_preprocessor=self.text_processor,
        )
        # VQA-RAD: ~10% of 1793 samples → ~179 val
        assert 100 < len(ds) < 300, f"Val set unexpected size: {len(ds)}"

    def test_len_test_split(self):
        """__len__ returns expected test split size."""
        ds = VQARADDataset(
            data_dir=self.DATA_DIR,
            split="test",
            image_transform=self.image_processor,
            text_preprocessor=self.text_processor,
        )
        # VQA-RAD: ~10% of 1793 samples → ~179 test
        assert 100 < len(ds) < 300, f"Test set unexpected size: {len(ds)}"

    def test_stratified_lengths(self):
        """All splits together account for all annotations."""
        train = VQARADDataset(self.DATA_DIR, "train", self.image_processor, self.text_processor)
        val = VQARADDataset(self.DATA_DIR, "val", self.image_processor, self.text_processor)
        test = VQARADDataset(self.DATA_DIR, "test", self.image_processor, self.text_processor)
        total = len(train) + len(val) + len(test)
        # Expect ~1793 total samples
        assert 1700 < total < 1900, f"Total samples unexpected: {total}"

    # ── __getitem__ shape tests ─────────────────────────────────────────

    def test_getitem_keys(self):
        """__getitem__ returns all expected keys."""
        ds = VQARADDataset(self.DATA_DIR, "train", self.image_processor, self.text_processor)
        sample = ds[0]
        expected_keys = {
            "image",
            "input_ids",
            "attention_mask",
            "labels",
            "label_attention_mask",
            "is_yesno",
            "answer_label",
            "question",
            "answer",
            "image_path",
        }
        assert set(sample.keys()) == expected_keys, (
            f"Key mismatch. Extra: {set(sample.keys()) - expected_keys}, "
            f"Missing: {expected_keys - set(sample.keys())}"
        )

    def test_getitem_image_shape(self):
        """Image tensor is (3, 224, 224) float32."""
        ds = VQARADDataset(self.DATA_DIR, "train", self.image_processor, self.text_processor)
        sample = ds[0]
        img = sample["image"]
        assert isinstance(img, torch.Tensor), "Image should be a tensor"
        assert img.shape == (3, 224, 224), f"Expected (3,224,224), got {img.shape}"
        assert img.dtype == torch.float32, f"Expected float32, got {img.dtype}"

    def test_getitem_text_shapes(self):
        """Text tensors are 1-D with correct dtypes."""
        ds = VQARADDataset(self.DATA_DIR, "train", self.image_processor, self.text_processor)
        sample = ds[0]

        assert sample["input_ids"].dim() == 1, "input_ids should be 1-D"
        assert sample["attention_mask"].dim() == 1, "attention_mask should be 1-D"
        assert sample["labels"].dim() == 1, "labels should be 1-D"
        assert sample["label_attention_mask"].dim() == 1, "label_attention_mask should be 1-D"

        assert (
            sample["input_ids"].dtype == torch.long
        ), f"Expected long, got {sample['input_ids'].dtype}"
        assert sample["labels"].dtype == torch.long, f"Expected long, got {sample['labels'].dtype}"

        # input_ids and attention_mask should have same length
        assert (
            sample["input_ids"].shape[0] == sample["attention_mask"].shape[0]
        ), "input_ids and attention_mask length mismatch"

    def test_getitem_label_masking(self):
        """Labels mask question tokens (-100) and keep answer tokens."""
        ds = VQARADDataset(self.DATA_DIR, "train", self.image_processor, self.text_processor)
        sample = ds[0]

        labels = sample["labels"]
        # At least some tokens should be masked (question part)
        assert (labels == -100).any(), "No masked tokens found — question part not masked"
        # At least some tokens should be valid (answer part)
        assert (labels != -100).any(), "All tokens masked — no answer tokens"
        # The label_attention_mask should match
        assert (
            sample["label_attention_mask"] == (labels != -100).long()
        ).all(), "label_attention_mask doesn't match labels"

    def test_getitem_scalar_types(self):
        """Scalar fields have correct types."""
        ds = VQARADDataset(self.DATA_DIR, "train", self.image_processor, self.text_processor)
        sample = ds[0]

        assert isinstance(sample["is_yesno"], torch.Tensor), "is_yesno should be tensor"
        assert sample["is_yesno"].dim() == 0, "is_yesno should be 0-D (scalar)"
        assert (
            sample["is_yesno"].dtype == torch.float
        ), f"Expected float, got {sample['is_yesno'].dtype}"

        assert isinstance(sample["answer_label"], torch.Tensor), "answer_label should be tensor"
        assert sample["answer_label"].dim() == 0, "answer_label should be 0-D (scalar)"
        assert (
            sample["answer_label"].dtype == torch.long
        ), f"Expected long, got {sample['answer_label'].dtype}"

        assert isinstance(sample["question"], str), "question should be string"
        assert isinstance(sample["answer"], str), "answer should be string"
        assert isinstance(sample["image_path"], str), "image_path should be string"

    # ── Yes/No classification tests ─────────────────────────────────────

    def test_yesno_classification_yes(self):
        """Yes/no questions with 'yes' answer get correct label."""
        ds = VQARADDataset(self.DATA_DIR, "train", self.image_processor, self.text_processor)
        for i in range(len(ds)):
            sample = ds[i]
            if sample["is_yesno"] > 0 and sample["answer"].strip().lower().startswith("yes"):
                assert sample["answer_label"].item() == 1, (
                    f"Expected answer_label=1 for 'yes', got {sample['answer_label'].item()}. "
                    f"Question: {sample['question']}, Answer: {sample['answer']}"
                )
                return
        pytest.fail("No yes-answer yes/no question found in dataset")

    def test_yesno_classification_no(self):
        """Yes/no questions with 'no' answer get correct label."""
        ds = VQARADDataset(self.DATA_DIR, "train", self.image_processor, self.text_processor)
        for i in range(len(ds)):
            sample = ds[i]
            if sample["is_yesno"] > 0 and sample["answer"].strip().lower().startswith("no"):
                assert sample["answer_label"].item() == 0, (
                    f"Expected answer_label=0 for 'no', got {sample['answer_label'].item()}. "
                    f"Question: {sample['question']}, Answer: {sample['answer']}"
                )
                return
        pytest.fail("No no-answer yes/no question found in dataset")

    def test_open_ended_answer_label(self):
        """Open-ended questions have answer_label=-1."""
        ds = VQARADDataset(self.DATA_DIR, "train", self.image_processor, self.text_processor)
        for i in range(len(ds)):
            sample = ds[i]
            if sample["is_yesno"] == 0:
                assert sample["answer_label"].item() == -1, (
                    f"Expected answer_label=-1 for open-ended, got {sample['answer_label'].item()}. "
                    f"Question: {sample['question']}, Answer: {sample['answer']}"
                )
                return
        pytest.fail("No open-ended question found in dataset")

    # ── filter_type tests ───────────────────────────────────────────────

    def test_filter_yesno_only(self):
        """filter_type='yesno' returns only yes/no questions."""
        ds = VQARADDataset(
            self.DATA_DIR, "train", self.image_processor, self.text_processor, filter_type="yesno"
        )
        assert len(ds) > 0, "Should have yes/no samples"
        for i in range(min(20, len(ds))):
            sample = ds[i]
            assert sample["is_yesno"] > 0, "Non-yes/no sample in filtered dataset"
            assert sample["answer_label"].item() in (
                0,
                1,
            ), f"Expected binary label, got {sample['answer_label'].item()}"

    def test_filter_open_only(self):
        """filter_type='open' returns only open-ended questions."""
        ds = VQARADDataset(
            self.DATA_DIR, "train", self.image_processor, self.text_processor, filter_type="open"
        )
        assert len(ds) > 0, "Should have open-ended samples"
        for i in range(min(20, len(ds))):
            sample = ds[i]
            assert sample["is_yesno"] == 0, "Yes/no sample in open-only dataset"
            assert (
                sample["answer_label"].item() == -1
            ), f"Expected answer_label=-1, got {sample['answer_label'].item()}"

    # ── Edge cases ──────────────────────────────────────────────────────

    def test_image_loads_from_disk(self):
        """Image files exist and load successfully."""
        ds = VQARADDataset(self.DATA_DIR, "train", self.image_processor, self.text_processor)
        for i in range(min(10, len(ds))):
            sample = ds[i]
            img_path = sample["image_path"]
            assert os.path.exists(img_path), f"Image file not found: {img_path}"
            from PIL import Image

            img = Image.open(img_path)
            img.verify()  # Raises exception if corrupted

    def test_multiple_samples_independent(self):
        """Each call to __getitem__ returns independent samples."""
        ds = VQARADDataset(self.DATA_DIR, "train", self.image_processor, self.text_processor)
        s1 = ds[0]
        s2 = ds[1]
        # Different samples should have different questions (unlikely identical)
        assert s1["question"] != s2["question"], "Two consecutive samples have identical questions"

    def test_consistent_across_epochs(self):
        """Repeated access to same index returns same data."""
        ds = VQARADDataset(self.DATA_DIR, "train", self.image_processor, self.text_processor)
        first = ds[5]
        second = ds[5]
        assert torch.equal(first["image"], second["image"]), "Images differ between accesses"
        assert torch.equal(
            first["input_ids"], second["input_ids"]
        ), "input_ids differ between accesses"
        assert first["question"] == second["question"], "Questions differ between accesses"
        assert first["answer"] == second["answer"], "Answers differ between accesses"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
