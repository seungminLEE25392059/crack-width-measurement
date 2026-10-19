"""Shared crack measurement and experimental severity grading.

Method (Task 1 report, Measurement and Severity Grading): every 8-connected
component of a mask is one crack. Each crack is skeletonised; a Euclidean
distance transform of the crack region gives d_i at skeleton point i and the
local width w_i = 2 d_i. Mean and maximum width are taken over the skeleton
points, and length follows the skeleton path. The same function is applied to
predicted and reference masks.

Inputs : bool mask (H, W)
Outputs: per-crack measurements in pixels, label image, severity thresholds
Owner  : Junsen Liang
"""
from __future__ import annotations

import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize

from . import config
from .utils import placeholder

CRACK_KEYS = ("id", "area_px", "length_px", "mean_width_px", "max_width_px")


@placeholder
def measure_cracks(mask: np.ndarray) -> tuple[list[dict], np.ndarray]:
    """Measure every crack (8-connected component) in a mask.

    Returns:
        cracks: one dict per component with the keys in CRACK_KEYS (pixels)
        labels: int array (H, W); 0 = background, k = crack with id k
    TODO(Junsen): replace the rough estimate with the Task 1 method: skeletonize
    each component, scipy.ndimage.distance_transform_edt for d_i, w_i = 2 d_i at
    skeleton pixels, mean and max of w_i, and skeleton length with diagonal steps
    counted as sqrt(2). The placeholder uses length = skeleton pixel count and
    mean width = area / length (max width = mean width).
    """
    labels = label(mask, connectivity=config.CONNECTIVITY)
    cracks = []
    for region in regionprops(labels):
        length = float(skeletonize(region.image).sum())
        width = region.area / max(length, 1.0)
        cracks.append({
            "id": int(region.label),
            "area_px": int(region.area),
            "length_px": length,
            "mean_width_px": float(width),
            "max_width_px": float(width),
        })
    return cracks, labels


@placeholder
def severity_thresholds(train_max_widths: list[float]) -> tuple[float, float]:
    """Fixed pixel thresholds (low, high) for the experimental severity categories.

    TODO(Junsen): confirm the rule. Default from the Task 2 plan: the 1/3 and 2/3
    quantiles of reference maximum widths from the TRAINING split only.
    """
    widths = np.asarray(train_max_widths, float)
    low, high = np.quantile(widths, [1 / 3, 2 / 3])
    return float(low), float(high)


@placeholder
def severity_grade(max_width_px: float, thresholds: tuple[float, float]) -> str:
    """Grade one crack as minor, moderate or severe from its maximum width."""
    low, high = thresholds
    if max_width_px < low:
        return config.SEVERITY_LABELS[0]
    if max_width_px < high:
        return config.SEVERITY_LABELS[1]
    return config.SEVERITY_LABELS[2]
