"""Medical-aware image augmentation pipeline.

WARNING: Aggressive augmentation is dangerous for medical imaging.
Random flips change anatomical orientation (left vs right).
Extreme crops remove diagnostic regions.
High-intensity noise obscures pathology.

We use only conservative, anatomy-preserving augmentations.
"""

import albumentations as A
import numpy as np
from PIL import Image


class MedicalAugmentationPipeline:
    """Conservative augmentation pipeline safe for medical images.

    Design rationale for each augmentation:
    - **Rotation (10 deg)**: Small rotations mimic patient positioning variation.
      Larger rotations would create unrealistic anatomical angles.
    - **Brightness/Contrast (0.9-1.1)**: Mimics different X-ray exposure levels.
      Wider ranges would create unrealistic image quality.
    - **Random crop (85% retention)**: Simulates slightly different field-of-view.
      Must retain most of the image to avoid cropping out pathology.
    - **Gaussian noise (sigma=0.03)**: Sensor noise modeling.
      Higher noise levels would obscure subtle findings.
    - **No flips**: Medical images have defined left/right orientation.
      Flipping would reverse anatomy (e.g., heart on the wrong side).

    Args:
        image_size: Target size after augmentation.
        rotation_limit: Max rotation degrees (default: 10).
        brightness_range: Brightness jitter range (default: (0.9, 1.1)).
        contrast_range: Contrast jitter range (default: (0.9, 1.1)).
        min_crop_retention: Minimum image retention after crop (default: 0.85).
        noise_sigma: Gaussian noise sigma (default: 0.03).
        p: Probability of applying each augmentation (default: 0.5).
    """

    def __init__(
        self,
        image_size: int = 224,
        rotation_limit: float = 10.0,
        brightness_range: tuple = (0.9, 1.1),
        contrast_range: tuple = (0.9, 1.1),
        min_crop_retention: float = 0.85,
        noise_sigma: float = 0.03,
        p: float = 0.5,
    ):
        self.image_size = image_size

        # Calculate crop scale from retention
        # If we keep 85% of the image, the crop scale is sqrt(0.85) ≈ 0.92
        crop_scale = min_crop_retention**0.5

        self.transform = A.Compose([
            A.SafeRotate(
                limit=rotation_limit,
                border_mode=0,
                value=0,
                p=p,  # cv2.BORDER_CONSTANT
            ),
            A.RandomBrightnessContrast(
                brightness_limit=(brightness_range[0] - 1, brightness_range[1] - 1),
                contrast_limit=(contrast_range[0] - 1, contrast_range[1] - 1),
                p=p,
            ),
            A.RandomResizedCrop(
                height=image_size,
                width=image_size,
                scale=(crop_scale, 1.0),
                ratio=(0.9, 1.1),  # Keep near-square crops
                p=p,
            ),
            A.GaussNoise(var_limit=(noise_sigma * 255) ** 2, mean=0, p=p),
        ])

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply augmentation pipeline to a medical image.

        Args:
            image: numpy array (H, W, C) in RGB order.

        Returns:
            Augmented image as numpy array (H, W, C).
        """
        augmented = self.transform(image=image)
        return augmented["image"]

    def augment_pil(self, image: Image.Image) -> Image.Image:
        """Apply augmentation to a PIL image.

        Args:
            image: PIL Image.

        Returns:
            Augmented PIL Image.
        """
        np_image = np.array(image)
        augmented = self(np_image)
        return Image.fromarray(augmented)
