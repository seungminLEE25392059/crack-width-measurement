"""Method A: traditional image processing (no training).

Pipeline (Task 1 report, Section 5.2), with the choices selected on the
validation split by scripts/tune_a.py:

  1. greyscale
  2. median filter                         "median_ksize"   (0 = off, else odd)
  3. contrast enhancement                  "enhance"        "none" | "histeq" | "clahe"
  4. crack response R, bright = crack      "background"
       "none"     : R = 255 - enhanced image (cracks are darker than concrete)
       "blackhat" : R = black top-hat of the enhanced image (closing - image),
                    which keeps thin dark structures and removes slow shading
  5. binarisation of R                     "threshold"      "otsu" | "adaptive" | "canny"
  6. morphological opening, then closing   "open_radius", "close_radius" (0 = off)
  7. connected-component filter            "min_area", "min_elongation", "elongation_rule"

Why step 4 has a black top-hat option: crack pixels are only about 1.4 % of
the dataset, so the grey-level histogram is almost all background and a global
Otsu threshold on the raw image tends to split the background texture rather
than separate the crack. The top-hat (Week 3 morphology) turns the crack into a
bright, roughly bimodal response. Both options are compared on validation data.

Inputs : RGB uint8 image (H, W, 3), parameter dict (missing keys use DEFAULT_PARAMS)
Outputs: bool crack mask (H, W), True = crack
Owner  : Wenjie Hu
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi

DEFAULT_PARAMS = {
    "median_ksize": 5,            # odd median kernel size; 0 switches the filter off
    "enhance": "clahe",           # "none" | "histeq" | "clahe"
    "clahe_clip": 2.0,            # CLAHE contrast limit
    "clahe_grid": 8,              # CLAHE tiles per image side
    "background": "blackhat",     # "none" | "blackhat"
    "tophat_radius": 15,          # disk radius of the black top-hat (pixels)
    "threshold": "otsu",          # "otsu" | "adaptive" | "canny"
    "adaptive_block": 51,         # adaptive threshold: neighbourhood size (odd)
    "adaptive_c": 10,             # adaptive threshold: offset above the local mean
    "canny_scale": 1.0,           # Canny high threshold = scale x Otsu level of R
    "canny_fill_radius": 3,       # closing radius that joins the two crack edges
    "open_radius": 1,             # opening disk radius (0 = off)
    "close_radius": 2,            # closing disk radius (0 = off)
    "min_area": 200,              # minimum component area (pixels)
    "min_elongation": 3.0,        # minimum elongation (0 = no shape test)
    "elongation_rule": "ellipse", # "ellipse" | "thinness" (see filter_components)
}


# --- Building blocks (also used by scripts/tune_a.py) ---------------------------------

def disk(radius: int) -> np.ndarray | None:
    """Disk-shaped structuring element; None when radius is 0."""
    if radius <= 0:
        return None
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def full_params(params: dict | None) -> dict:
    """DEFAULT_PARAMS updated with `params`; unknown keys are an error (catches typos)."""
    merged = dict(DEFAULT_PARAMS)
    if params:
        unknown = set(params) - set(DEFAULT_PARAMS)
        if unknown:
            raise KeyError(f"Unknown Method A parameter(s): {sorted(unknown)}")
        merged.update(params)
    return merged


def to_grey(image: np.ndarray) -> np.ndarray:
    """Step 1: RGB uint8 -> greyscale uint8 (a grey image is passed through)."""
    if image.ndim == 2:
        return image.astype(np.uint8)
    return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)


def enhance(grey: np.ndarray, p: dict) -> np.ndarray:
    """Steps 2-3: median filter, then histogram equalisation or CLAHE."""
    out = grey
    if p["median_ksize"] and p["median_ksize"] > 1:
        out = cv2.medianBlur(out, int(p["median_ksize"]))
    if p["enhance"] == "histeq":
        out = cv2.equalizeHist(out)
    elif p["enhance"] == "clahe":
        grid = int(p["clahe_grid"])
        out = cv2.createCLAHE(clipLimit=float(p["clahe_clip"]), tileGridSize=(grid, grid)).apply(out)
    elif p["enhance"] != "none":
        raise ValueError(f"enhance must be none, histeq or clahe, not {p['enhance']!r}")
    return out


def crack_response(enhanced: np.ndarray, p: dict) -> np.ndarray:
    """Step 4: an image in which crack pixels are bright."""
    if p["background"] == "blackhat":
        return cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, disk(int(p["tophat_radius"])))
    if p["background"] == "none":
        return 255 - enhanced
    raise ValueError(f"background must be none or blackhat, not {p['background']!r}")


def binarise(response: np.ndarray, p: dict) -> np.ndarray:
    """Step 5: crack / background decision on the response image."""
    if p["threshold"] == "otsu":
        _, bw = cv2.threshold(response, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return bw > 0
    if p["threshold"] == "adaptive":
        block = int(p["adaptive_block"]) | 1          # OpenCV needs an odd block size
        bw = cv2.adaptiveThreshold(response, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, block, -float(p["adaptive_c"]))
        return bw > 0
    if p["threshold"] == "canny":
        # Hysteresis thresholds follow the image's own Otsu level, so they adapt
        # to each photograph's contrast.
        otsu, _ = cv2.threshold(response, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        high = float(np.clip(float(p["canny_scale"]) * otsu, 1.0, 255.0))
        edges = cv2.Canny(response, 0.4 * high, high)
        # Canny gives the two crack boundaries, not a region: close them
        # together and fill the enclosed area (Task 1 report, Section 5.2).
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, disk(int(p["canny_fill_radius"])))
        return ndi.binary_fill_holes(closed > 0)
    raise ValueError(f"threshold must be otsu, adaptive or canny, not {p['threshold']!r}")


def morphology(bw: np.ndarray, p: dict) -> np.ndarray:
    """Step 6: opening removes isolated specks; closing bridges short gaps."""
    out = bw.astype(np.uint8)
    if p["open_radius"] > 0:
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, disk(int(p["open_radius"])))
    if p["close_radius"] > 0:
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, disk(int(p["close_radius"])))
    return out > 0


# --- Step 7: component filter (shared with Method D) ----------------------------------

def component_stats(mask: np.ndarray):
    """Shape statistics of every 8-connected component, without a Python loop.

    Returns (labels, area, major, minor, boundary); every array except `labels`
    is indexed by label (index 0 = background, unused).
      * major / minor: axes of the ellipse with the same normalised second
        central moments, as MATLAB regionprops (Week 3, Image Analysis). Each
        variance includes 1/12, the moment of a unit pixel, so the minor axis
        is never zero, even for a one-pixel-wide line.
      * boundary: pixels of the component with a 4-neighbour in the background.
    """
    mask = np.asarray(mask, bool)
    n, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    rows, cols = np.nonzero(labels)
    lab = labels[rows, cols]
    area = np.bincount(lab, minlength=n).astype(np.float64)
    safe = np.maximum(area, 1.0)
    r = rows.astype(np.float64)
    c = cols.astype(np.float64)
    mean_r = np.bincount(lab, r, n) / safe
    mean_c = np.bincount(lab, c, n) / safe
    var_r = np.bincount(lab, r * r, n) / safe - mean_r ** 2 + 1.0 / 12.0
    var_c = np.bincount(lab, c * c, n) / safe - mean_c ** 2 + 1.0 / 12.0
    cov_rc = np.bincount(lab, r * c, n) / safe - mean_r * mean_c
    spread = np.sqrt((var_r - var_c) ** 2 + 4.0 * cov_rc ** 2)
    major = 2.0 * np.sqrt(2.0) * np.sqrt(np.maximum(var_r + var_c + spread, 0.0))
    minor = 2.0 * np.sqrt(2.0) * np.sqrt(np.maximum(var_r + var_c - spread, 0.0))

    m8 = mask.astype(np.uint8)
    inner = cv2.erode(m8, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
                      borderType=cv2.BORDER_CONSTANT, borderValue=0)
    boundary = np.bincount(labels[(m8 > 0) & (inner == 0)], minlength=n).astype(np.float64)
    return labels, area, major, minor, boundary


def elongation(area, major, minor, boundary, rule: str = "ellipse") -> np.ndarray:
    """Elongation score per component.

    "ellipse"  : major / minor axis length (the Task 1 rule).
    "thinness" : boundary^2 / (4 x area). For a strip of length L and width W
                 the boundary is about 2L and the area LW, so the score is
                 about L / W whether or not the strip is straight. A disk
                 scores about 3.1 and a square 4.
    The ellipse rule fails on curved or branched cracks, whose equivalent
    ellipse is nearly round; the thinness rule was added for that case.
    """
    if rule == "ellipse":
        return major / np.maximum(minor, 1e-9)
    if rule == "thinness":
        return boundary ** 2 / (4.0 * np.maximum(area, 1.0))
    raise ValueError(f"elongation_rule must be ellipse or thinness, not {rule!r}")


def filter_components(mask: np.ndarray, min_area: int, min_elongation: float,
                      rule: str = "ellipse") -> np.ndarray:
    """Keep 8-connected components that are large and elongated enough (shared with Method D).

    A component is kept when area >= min_area AND elongation >= min_elongation.
    min_elongation = 0 switches the shape test off. Note the two rules use
    different scales (round blob: about 1 for "ellipse", about 3-4 for "thinness").
    """
    mask = np.asarray(mask, bool)
    if not mask.any():
        return mask.copy()
    labels, area, major, minor, boundary = component_stats(mask)
    keep = (area >= min_area) & (elongation(area, major, minor, boundary, rule) >= min_elongation)
    keep[0] = False                                  # background
    return keep[labels]


# --- Main entry point ------------------------------------------------------------------

def segment_a(image: np.ndarray, params: dict, return_steps: bool = False):
    """Segment cracks with classical image processing (steps 1-7 above).

    Returns the bool crack mask (H, W). With return_steps=True also returns a
    dict of intermediate images, used for the step-by-step report figure.
    """
    p = full_params(params)
    grey = to_grey(image)
    enhanced = enhance(grey, p)
    response = crack_response(enhanced, p)
    binary = binarise(response, p)
    cleaned = morphology(binary, p)
    mask = filter_components(cleaned, p["min_area"], p["min_elongation"], p["elongation_rule"])
    if return_steps:
        return mask, {"grey": grey, "enhanced": enhanced, "response": response,
                      "binary": binary, "morphology": cleaned, "final": mask}
    return mask
