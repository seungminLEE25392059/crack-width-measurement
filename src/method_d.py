"""Method D: U-Net mask with post-processing.

Processing (Task 1 report, Method D): Method C's binary mask -> connected-
component filtering (method_a.filter_components, thresholds chosen for U-Net
masks on validation data) -> one binary erosion with a radius-1 disk. The
no-erosion control skips the erosion. Timing covers this refinement only.

Inputs : Method C's bool mask (H, W), parameter dict
Outputs: refined bool mask (H, W)
Owner  : Lingyu Wu
"""
from __future__ import annotations

import numpy as np

from .utils import placeholder

DEFAULT_PARAMS = {
    "use_filter": True,        # ablation switch: component filtering
    "min_area": 50,
    "min_elongation": 3.0,
    "use_erosion": True,       # ablation switch: erosion
    "erosion_radius": 1,
}


@placeholder
def refine_d(mask_c: np.ndarray, params: dict) -> np.ndarray:
    """Refine a Method C mask.

    TODO(Lingyu): if params["use_filter"], apply method_a.filter_components with
    params["min_area"] and params["min_elongation"]; if params["use_erosion"],
    apply scipy.ndimage.binary_erosion(mask, structure=skimage.morphology.disk(r))
    (disk(1) is the exact 3 x 3 cross, as strel("disk", 1, 0) in the Task 1 plan).
    The placeholder returns the mask unchanged.
    """
    return mask_c.copy()
