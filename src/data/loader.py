"""PyTorch Dataset classes for VQA-RAD and PathVQA datasets.

Implements lazy loading for memory efficiency and supports train/val/test splits
with separate handling of binary (yes/no) and open-ended questions.

VQA-RAD dataset structure:
    data/
    ├── images/              # All images (.png, .jpg)
    └── VQA_RAD_Dataset.json # Annotations with questions, answers, split info
"""

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


class VQARADDataset(Dataset):
    """PyTorch Dataset for VQA-RAD.

    Features:
    - Lazy loading: images loaded on demand, not cached in memory
    - Supports binary (yes/no) and open-ended questions separately
    - Returns preprocessed image tensors and tokenized text

    Args:
        data_dir: Root directory containing 'images/' and the JSON annotation file.
        split: One of 'train', 'val', 'test'.
        image_transform: Callable to preprocess images.
        text_preprocessor: Callable to tokenize questions/answers.
        filter_type: If 'yesno', only return yes/no questions.
                     If 'open', only return open-ended questions.
                     If None, return all.
    """

    def __init__(
        self,
        data_dir: str = "data/raw/vqa_rad",
        split: str = "train",
        image_transform: Optional[Callable] = None,
        text_preprocessor: Optional[Callable] = None,
        filter_type: Optional[str] = None,
        annotation_files: Optional[list[str]] = None,
    ):
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / "images"
        self.split = split
        self.image_transform = image_transform
        self.text_preprocessor = text_preprocessor
        self.filter_type = filter_type

        # Load annotations
        if annotation_files is None:
            annotation_files = ["VQA_RAD_Dataset.json", "vqa_rad_dataset.json"]

        annotation_path = None
        for filename in annotation_files:
            candidate = self.data_dir / filename
            if candidate.exists():
                annotation_path = candidate
                break

        if annotation_path is None:
            raise FileNotFoundError(
                f"No annotation file found in {self.data_dir}. Tried: {annotation_files}"
            )

        with open(annotation_path) as f:
            self.annotations = json.load(f)

        # Parse and filter samples
        self.samples = self._parse_annotations()

        print(
            f"[OK] VQA-RAD {split} split: {len(self.samples)} samples ({filter_type or 'all'} questions)"
        )

    def __len__(self) -> int:
        """Return the number of samples in this dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single training sample.

        Returns:
            Dict with keys:
                - image: preprocessed image tensor (3, H, W)
                - input_ids: tokenized question IDs (seq_len,)
                - attention_mask: question attention mask (seq_len,)
                - labels: tokenized answer IDs (seq_len,) for generation
                - label_mask: binary mask indicating answer tokens
                - is_yesno: 1 if yes/no question, 0 otherwise
        """
        sample = self.samples[idx]

        # Load image (lazy — not cached)
        image_path = sample["image_path"]
        if not os.path.exists(image_path):
            # Try other extensions
            for ext in [".jpg", ".jpeg", ".png"]:
                alt_path = os.path.splitext(image_path)[0] + ext
                if os.path.exists(alt_path):
                    image_path = alt_path
                    break

        image = Image.open(image_path).convert("RGB")

        if self.image_transform:
            image = self.image_transform(image)

        # Tokenize question + answer for causal LM training
        if self.text_preprocessor:
            qa_enc = self.text_preprocessor.tokenize_qa_pair(sample["question"], sample["answer"])
        else:
            qa_enc = {
                "input_ids": torch.zeros(192, dtype=torch.long),
                "attention_mask": torch.zeros(192, dtype=torch.long),
                "labels": torch.full((192,), -100, dtype=torch.long),
            }

        input_ids = qa_enc["input_ids"]
        attention_mask = qa_enc["attention_mask"]
        labels = qa_enc["labels"]
        label_attention_mask = (labels != -100).long()

        # Determine if yes/no
        qtype = sample.get("question_type", "open").lower()
        is_yesno = 1.0 if qtype in ["yes/no", "binary"] else 0.0

        # Binary label for yes/no questions: 0=no, 1=yes, -1=not yes/no
        answer_label = -1
        if is_yesno > 0:
            answer_lower = sample["answer"].strip().lower()
            if answer_lower.startswith("yes"):
                answer_label = 1
            elif answer_lower.startswith("no"):
                answer_label = 0

        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "label_attention_mask": label_attention_mask,
            "is_yesno": torch.tensor(is_yesno, dtype=torch.float),
            "answer_label": torch.tensor(answer_label, dtype=torch.long),
            "question": sample["question"],  # Keep as string for debugging
            "answer": sample["answer"],
            "image_path": sample["image_path"],
        }

    def _parse_annotations(self) -> list[dict]:
        """Parse VQA-RAD JSON annotations and filter by split and type.

        VQA-RAD JSON structure varies. We handle both list and dict formats.
        """
        samples = []

        # Handle different JSON formats
        if isinstance(self.annotations, list):
            entries = self.annotations
        elif isinstance(self.annotations, dict):
            entries = self.annotations.get("data", self.annotations.get("questions", []))
        else:
            raise ValueError(f"Unexpected annotation format: {type(self.annotations)}")

        for entry in entries:
            if isinstance(entry, dict):
                # Extract fields with fallback keys
                split = str(entry.get("split", entry.get("phase", ""))).lower()
                if split != self.split and self.split not in split:
                    continue

                question = entry.get("question", "")
                answer = entry.get("answer", "")

                # Determine image filename
                img_filename = entry.get("filename", entry.get("image_name", entry.get("img", "")))
                if not img_filename:
                    # Some VQA-RAD formats embed image name
                    img_id = entry.get("image_id", entry.get("id", ""))
                    img_filename = f"{img_id}.png"

                # Clean up path separators
                img_filename = os.path.basename(img_filename)

                # Determine question type
                qtype = entry.get("type", entry.get("question_type", "open"))

                sample = {
                    "image_path": str(self.image_dir / img_filename),
                    "question": question,
                    "answer": answer,
                    "question_type": qtype,
                    "image_id": entry.get("image_id", entry.get("id", "")),
                }

                # Apply optional filter
                if self.filter_type == "yesno" and qtype.lower() not in ["yes/no", "binary"]:
                    continue
                if self.filter_type == "open" and qtype.lower() in ["yes/no", "binary"]:
                    continue

                samples.append(sample)

        return samples


class PathVQADataset(VQARADDataset):
    """PyTorch Dataset for PathVQA with flexible annotation file names.

    Inherits __len__ and __getitem__ from VQARADDataset.
    """

    def __init__(
        self,
        data_dir: str = "data/raw/pathvqa",
        split: str = "train",
        image_transform: Optional[Callable] = None,
        text_preprocessor: Optional[Callable] = None,
        filter_type: Optional[str] = None,
        annotation_files: Optional[list[str]] = None,
    ):
        if annotation_files is None:
            annotation_files = [
                "PathVQA_Dataset.json",
                "PathVQA.json",
                "pathvqa.json",
                "pathvqa_dataset.json",
            ]

        super().__init__(
            data_dir=data_dir,
            split=split,
            image_transform=image_transform,
            text_preprocessor=text_preprocessor,
            filter_type=filter_type,
            annotation_files=annotation_files,
        )


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Custom collate function for MedVQA DataLoader.

    Handles variable-length sequences by padding.

    Args:
        batch: List of sample dicts from VQARADDataset.

    Returns:
        Batched dict with padded tensors.
    """
    # Stack images
    images = torch.stack([item["image"] for item in batch])

    # Pad sequences
    input_ids = pad_sequence(
        [item["input_ids"] for item in batch], batch_first=True, padding_value=0
    )
    attention_mask = pad_sequence(
        [item["attention_mask"] for item in batch], batch_first=True, padding_value=0
    )
    labels = pad_sequence([item["labels"] for item in batch], batch_first=True, padding_value=-100)
    label_attention_mask = pad_sequence(
        [item["label_attention_mask"] for item in batch], batch_first=True, padding_value=0
    )

    # Stack scalars
    is_yesno = torch.stack([item["is_yesno"] for item in batch])
    answer_labels = torch.stack([item["answer_label"] for item in batch])

    return {
        "images": images,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "label_attention_mask": label_attention_mask,
        "is_yesno": is_yesno,
        "answer_labels": answer_labels,
        "questions": [item.get("question", "") for item in batch],
        "answers": [item.get("answer", "") for item in batch],
        "image_paths": [item.get("image_path", "") for item in batch],
    }


def create_dataloaders(
    data_dir: str = "data/raw/vqa_rad",
    image_transform: Optional[Callable] = None,
    text_preprocessor: Optional[Callable] = None,
    batch_size: int = 4,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders for VQA-RAD.

    Args:
        data_dir: Path to VQA-RAD data directory.
        image_transform: Image preprocessing transform.
        text_preprocessor: Text tokenization preprocessor.
        batch_size: Batch size per GPU.
        num_workers: DataLoader workers.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    # Create datasets
    train_dataset = VQARADDataset(
        data_dir=data_dir,
        split="train",
        image_transform=image_transform,
        text_preprocessor=text_preprocessor,
    )
    val_dataset = VQARADDataset(
        data_dir=data_dir,
        split="val",
        image_transform=image_transform,
        text_preprocessor=text_preprocessor,
    )
    test_dataset = VQARADDataset(
        data_dir=data_dir,
        split="test",
        image_transform=image_transform,
        text_preprocessor=text_preprocessor,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    print(
        f"[OK] DataLoaders created: train={len(train_loader)} val={len(val_loader)} test={len(test_loader)} batches"
    )
    return train_loader, val_loader, test_loader
