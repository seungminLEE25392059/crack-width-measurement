"""Method A: traditional image processing.

Pipeline (Task 1 report, Method A): median filter -> histogram equalisation or
CLAHE -> Otsu threshold and/or Canny edges -> morphological opening and closing
-> connected-component filtering by area and elongation.

Inputs : RGB uint8 image (H, W, 3), parameter dict
Outputs: bool crack mask (H, W)
Owner  : Wenjie Hu
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.measure import label, regionprops

from . import config
from .utils import placeholder

DEFAULT_PARAMS = {
    "median_ksize": 5,        # odd kernel size of the median filter
    "use_clahe": True,        # CLAHE instead of global histogram equalisation
    "clahe_clip": 2.0,
    "clahe_grid": 8,
    "close_radius": 2,        # radius of the closing structuring element (pixels)
    "min_area": 100,          # minimum component area (pixels)
    "min_elongation": 3.0,    # minimum major/minor axis ratio
}


@placeholder
def segment_a(image: np.ndarray, params: dict) -> np.ndarray:
    """Segment cracks with classical image processing.

    TODO(Wenjie): implement the Task 1 pipeline with the parameters above, select
    them on the validation split and save them with utils.save_params("A", ...).
    The placeholder applies Otsu's threshold to the grey image and keeps the
    darker class, because cracks are darker than the concrete surface.
    """
    grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    threshold, _ = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return grey < threshold


@placeholder
def filter_components(mask: np.ndarray, min_area: int, min_elongation: float) -> np.ndarray:
    """Keep 8-connected components that are large and elongated enough (shared with Method D).

    TODO(Wenjie): also require region.axis_major_length / region.axis_minor_length
    >= min_elongation. The placeholder removes small components only.
    """
    labels = label(mask, connectivity=config.CONNECTIVITY)
    keep = np.zeros(labels.max() + 1, bool)
    for region in regionprops(labels):
        keep[region.label] = region.area >= min_area
    return keep[labels]
