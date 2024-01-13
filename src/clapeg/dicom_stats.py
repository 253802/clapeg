from typing import (
    Any,
    Dict,
)

import numpy as np


def extract_image_statistics(image: np.ndarray) -> Dict[str, Any]:
    """
    Extracts statistics from an image.

    Args:
    - image (np.ndarray): Image data.

    Returns:
    - Dict[str, Any]: Dictionary containing image statistics.
    """
    return {"img_min": np.min(image), "img_max": np.max(image), "img_mean": np.mean(image), "img_std": np.std(image)}
