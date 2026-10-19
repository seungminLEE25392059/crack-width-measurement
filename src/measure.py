"""Shared crack measurement and experimental severity grading.

Purpose: measure each 8-connected component using skeletons and w = 2 * EDT.
Input: a 2D boolean mask (True = crack), without resizing or filtering.
Output: (cracks, labels); widths and lengths are in pixels. Label 0 is
background and label k corresponds to cracks with id k. Empty masks return
([], all-zero labels). Severity thresholds use TRAINING reference widths only.
Author: Junsen Liang

The discrete 2 * EDT convention is an estimate, not exact pixel thickness or
physical millimetres. All skeleton branches contribute to length. A single
skeleton pixel has length zero. The exterior is treated as background and
border components are flagged as potentially truncated.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

from . import config

CRACK_KEYS = ("id", "area_px", "length_px", "mean_width_px", "max_width_px")


def skeleton_length(skeleton):
    """Sum unique skeleton edges: axial=1, diagonal=sqrt(2).

    Exclude a diagonal if an axial two-step connection already exists,
    preventing small triangles at corners from inflating branch length.
    Total length includes all branches; an isolated pixel has length zero.
    """
    s = np.asarray(skeleton, dtype=bool)
    axial = np.count_nonzero(s[:, :-1] & s[:, 1:])
    axial += np.count_nonzero(s[:-1, :] & s[1:, :])
    down_right = s[:-1, :-1] & s[1:, 1:]
    down_right &= ~(s[:-1, 1:] | s[1:, :-1])
    down_left = s[:-1, 1:] & s[1:, :-1]
    down_left &= ~(s[:-1, :-1] | s[1:, 1:])
    diagonal = np.count_nonzero(down_right) + np.count_nonzero(down_left)
    return float(axial + np.sqrt(2.0) * diagonal)


def measure_cracks(mask: np.ndarray) -> tuple[list[dict], np.ndarray]:
    """Measure each 8-connected region without filtering or resizing.

    Coordinates are (row, column), zero-based in the original image.
    Widths are sampled at skeleton pixels and averaged with equal weight.
    Image exterior is treated as background; border components are flagged
    because their physical extent may be truncated by the image crop.
    Split fragments remain separate components, as specified in the plan.
    """
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError('mask must be a 2D boolean array; binarise it explicitly first')
    if 0 in mask.shape:
        raise ValueError('mask must have nonzero height and width')
    labels, count = ndi.label(mask, structure=np.ones((3, 3), dtype=bool))
    cracks = []
    for component_id, sl in enumerate(ndi.find_objects(labels), start=1):
        local = labels[sl] == component_id
        # Pad before EDT and thinning so boundary handling is explicit.
        padded = np.pad(local, 1, mode='constant', constant_values=False)
        skel = skeletonize(padded, method='zhang')
        dist = ndi.distance_transform_edt(padded)
        widths = 2.0 * dist[skel]
        offset = np.array([sl[0].start, sl[1].start])
        pixels = np.argwhere(local) + offset
        skel_coords = np.argwhere(skel) - 1 + offset
        cracks.append({
            'id': component_id,
            'area_px': int(local.sum()),
            'mean_width_px': float(widths.mean()) if widths.size else float('nan'),
            'max_width_px': float(widths.max()) if widths.size else float('nan'),
            'length_px': skeleton_length(skel),
            'skeleton_pixel_count': int(skel.sum()),
            'touches_border': bool(sl[0].start == 0 or sl[1].start == 0
                or sl[0].stop == mask.shape[0] or sl[1].stop == mask.shape[1]),
            'pixel_coords_rc': pixels,
            'skeleton_coords_rc': skel_coords,
            'local_widths_px': widths,
        })
    return cracks, labels



def severity_thresholds(train_max_widths: list[float]) -> tuple[float, float]:
    """Return 1/3 and 2/3 quantiles of TRAINING reference maximum widths.

    Each connected component contributes one width. The caller must supply
    training data only; this function cannot infer a list's dataset split.
    Linear interpolation is explicit for reproducible quantiles. Tied
    thresholds are valid and can leave the moderate category empty.
    """
    widths = np.asarray(train_max_widths, dtype=float)
    if widths.ndim != 1 or widths.size == 0:
        raise ValueError("train_max_widths must be a non-empty 1D sequence")
    if not np.all(np.isfinite(widths)) or np.any(widths <= 0):
        raise ValueError("training maximum widths must be finite and positive")
    low, high = np.quantile(widths, [1 / 3, 2 / 3], method="linear")
    return float(low), float(high)


def severity_grade(max_width_px: float, thresholds: tuple[float, float]) -> str:
    """Return minor (w < low), moderate (low <= w < high), or severe.

    These are experimental pixel-width categories, not structural safety grades.
    Values equal to a threshold enter the higher category.
    """
    bounds = np.asarray(thresholds, dtype=float)
    if (bounds.shape != (2,) or not np.all(np.isfinite(bounds))
            or bounds[0] <= 0 or bounds[0] > bounds[1]):
        raise ValueError("thresholds must be finite positive values with low <= high")
    width = float(max_width_px)
    if not np.isfinite(width) or width <= 0:
        raise ValueError("max_width_px must be finite and positive")
    low, high = bounds
    if width < low:
        return config.SEVERITY_LABELS[0]
    if width < high:
        return config.SEVERITY_LABELS[1]
    return config.SEVERITY_LABELS[2]
