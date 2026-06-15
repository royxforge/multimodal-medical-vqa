#!/usr/bin/env python3
"""Download and prepare the VQA-RAD dataset from Hugging Face.

VQA-RAD is a publicly available medical VQA dataset containing 1,793 QA pairs
over 355 radiology images. It is hosted on Hugging Face as
``flaviagiammarino/vqa-rad``.

The script:
  1. Downloads the dataset via the ``datasets`` library
  2. Creates 80/10/10 train/val/test splits (stratified by yes/no)
  3. Saves images to ``data/raw/vqa_rad/images/`` (deduplicated — only ~355
     unique images saved despite 1,793 Q&A pairs)
  4. Writes ``VQA_RAD_Dataset.json`` in the format expected by ``VQARADDataset``

Usage:
    python scripts/download_vqa_rad.py --output_dir data/raw/vqa_rad

Dataset citation:
    Lau, Jason J., et al. "VQA-RAD: A Dataset for Visual Question Answering
    in Radiology." Proc. of SPIE Medical Imaging, 2018.
"""

import argparse
import json
import random
import sys
from pathlib import Path

# Force UTF-8 for stdout/stderr (handles emoji in datasets library progress bars
# on Windows where the default console encoding is cp1252)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from datasets import load_dataset
from PIL import Image

# ── Image deduplication ─────────────────────────────────────────────────────
# VQA-RAD has ~355 unique images shared across 1,793 Q&A pairs.  We hash each
# image's pixel bytes so the exact same PIL image is saved to disk only once.
_IMAGE_HASHES: dict[str, str] = {}  # pixel hash -> canonical filename
_IMAGE_COUNTER: int = 0


def _image_filename(image: "Image.Image") -> str:
    """Return a stable filename for *image*, reusing canonical names for
    pixel-identical images."""
    global _IMAGE_COUNTER
    ident = str(hash(image.tobytes()))
    existing = _IMAGE_HASHES.get(ident)
    if existing is not None:
        return existing  # already saved — reuse
    _IMAGE_COUNTER += 1
    name = f"synpic{_IMAGE_COUNTER:05d}.png"
    _IMAGE_HASHES[ident] = name
    return name


def _save_image(image: "Image.Image", images_dir: Path) -> str:
    """Save *image* to *images_dir* (once per unique pixel content) and
    return its canonical filename."""
    filename = _image_filename(image)
    dst = images_dir / filename
    if not dst.exists():
        image.save(dst, format="PNG")
    return filename


# ── Helpers ─────────────────────────────────────────────────────────────────


def infer_question_type(answer: str) -> str:
    """Infer whether the question is yes/no or open-ended from the answer."""
    return "yes/no" if answer.strip().lower() in ("yes", "no") else "open"


def create_stratified_split(
    samples: list[dict], train_ratio: float = 0.8, val_ratio: float = 0.1, seed: int = 42
) -> dict[str, list[dict]]:
    """Stratified 80/10/10 split preserving yes/no proportion across splits."""
    rng = random.Random(seed)

    yesno = [s for s in samples if s["question_type"] == "yes/no"]
    open_ = [s for s in samples if s["question_type"] == "open"]
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}

    for group in (yesno, open_):
        rng.shuffle(group)
        n = len(group)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        splits["train"].extend(group[:n_train])
        splits["val"].extend(group[n_train : n_train + n_val])
        splits["test"].extend(group[n_train + n_val :])

    for v in splits.values():
        rng.shuffle(v)
    return splits


# ── Main ────────────────────────────────────────────────────────────────────


def download_vqa_rad(output_dir: str = "data/raw/vqa_rad") -> None:
    """Download VQA-RAD from Hugging Face, split, and save locally."""
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    output_path.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(exist_ok=True)

    print("=" * 56)
    print("Downloading VQA-RAD Dataset (flaviagiammarino/vqa-rad)")
    print("=" * 56)

    # ── 1. Download from Hugging Face ──────────────────────────────────────
    print("\n[1/5] Downloading dataset from Hugging Face ...")
    ds = load_dataset("flaviagiammarino/vqa-rad", split="train")
    print(f"  [OK] Loaded {len(ds)} total samples")

    # ── 2. Build sample list with metadata ─────────────────────────────────
    print("\n[2/5] Building sample metadata ...")
    samples: list[dict] = []

    for row in ds:
        img_filename = _save_image(row["image"], images_dir)
        qtype = infer_question_type(row["answer"])
        samples.append({
            "image_filename": img_filename,
            "question": row["question"],
            "answer": row["answer"],
            "question_type": qtype,
        })
    del ds  # free memory

    # ── 3. Create stratified splits ────────────────────────────────────────
    print("[3/5] Creating stratified 80/10/10 splits ...")
    splits = create_stratified_split(samples, seed=42)
    for split_name, sp in splits.items():
        yn = sum(1 for s in sp if s["question_type"] == "yes/no")
        op = len(sp) - yn
        print(f"       {split_name}: {len(sp):>4} samples ({yn} yes/no, {op} open)")

    # ── 4. Write annotation JSON ───────────────────────────────────────────
    print("[4/5] Writing VQA_RAD_Dataset.json ...")
    annotation_entries: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for split_name, sp in splits.items():
        for s in sp:
            annotation_entries[split_name].append({
                "question": s["question"],
                "answer": s["answer"],
                "filename": s["image_filename"],
                "split": split_name,
                "question_type": s["question_type"],
            })

    all_annotations = (
        annotation_entries["train"] + annotation_entries["val"] + annotation_entries["test"]
    )
    json_path = output_path / "VQA_RAD_Dataset.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_annotations, f, indent=2, ensure_ascii=False)

    # ── 5. Verify ──────────────────────────────────────────────────────────
    print("[5/5] Verifying ...")
    unique_images = len(_IMAGE_HASHES)
    saved_images = len(list(images_dir.glob("*.*")))
    print()
    print("=" * 56)
    print("[OK] Dataset Preparation Complete")
    print("=" * 56)
    print(f"  Annotation file : {json_path}")
    print(f"  Images on disk  : {saved_images} (unique: {unique_images})")
    print(f"  Total samples   : {len(all_annotations)}")
    print(f"    Train         : {len(annotation_entries['train'])}")
    print(f"    Val           : {len(annotation_entries['val'])}")
    print(f"    Test          : {len(annotation_entries['test'])}")
    print(
        f"  Yes/No questions: {sum(1 for s in all_annotations if s['question_type'] == 'yes/no')}"
    )
    print(f"  Open-ended      : {sum(1 for s in all_annotations if s['question_type'] == 'open')}")
    print()
    print("  Next step: train.py  or  notebooks/01_eda.ipynb")


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare VQA-RAD dataset from Hugging Face"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/raw/vqa_rad",
        help="Output directory for the prepared dataset",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val/test split")
    args = parser.parse_args()

    # Reset module-level state so re-runs don't carry over cached hashes
    global _IMAGE_HASHES, _IMAGE_COUNTER
    _IMAGE_HASHES.clear()
    _IMAGE_COUNTER = 0

    download_vqa_rad(args.output_dir)


if __name__ == "__main__":
    main()
