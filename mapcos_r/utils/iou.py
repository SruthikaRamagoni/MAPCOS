"""
IoU (intersection-over-union) utility — used by the Grounding Agent to score
spatial agreement between the CNN's Grad-CAM region and the detected follicles.
"""

import numpy as np


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Both masks are binary (0/1) arrays of the same shape."""
    mask_a = mask_a.astype(bool)
    mask_b = mask_b.astype(bool)

    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()

    if union == 0:
        return 0.0
    return float(intersection) / float(union)
