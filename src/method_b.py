"""Method B: HOG + GLCM features with an RBF-kernel SVM, applied patch by patch.

Pipeline (Task 1 report, Method B): sample balanced crack / non-crack patches
from training images -> HOG + GLCM feature vector per patch -> standardise ->
RBF SVM (C and gamma chosen on validation data) -> slide a window over each
image and average overlapping patch scores into a confidence map -> threshold.

Inputs : RGB uint8 image (H, W, 3), fitted model, parameter dict
Outputs: bool crack mask (H, W) and float32 confidence map (H, W) in [0, 1]
Owner  : Bowen Xu
"""
from __future__ import annotations

from pathlib import Path

import cv2
import joblib
import numpy as np

from .utils import placeholder

DEFAULT_PARAMS = {
    "patch": 32,               # patch side (pixels)
    "stride": 16,              # sliding-window step (8-16 in the plan)
    "svm_C": 1.0,
    "svm_gamma": "scale",
    "threshold": 0.5,          # confidence threshold, chosen on validation data
    "patches_per_class": 2000,
}


@placeholder
def extract_features_b(patch: np.ndarray) -> np.ndarray:
    """Feature vector of one grey or RGB patch.

    TODO(Bowen): HOG (skimage.feature.hog) concatenated with GLCM properties
    (skimage.feature.graycomatrix / graycoprops: contrast, homogeneity, energy,
    correlation). The placeholder returns the mean and standard deviation only.
    """
    grey = patch if patch.ndim == 2 else cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    values = grey.astype(np.float32) / 255.0
    return np.array([values.mean(), values.std()], np.float32)


@placeholder
def train_b(train_ids: list[str], params: dict):
    """Train the patch classifier on training images only.

    TODO(Bowen): sample balanced crack / non-crack patches with data.load_pair,
    extract features, fit Pipeline([StandardScaler(), SVC(kernel="rbf", C=...,
    gamma=..., probability=True)]) and return it. The placeholder returns None,
    so segment_b predicts no cracks.
    """
    return None


def save_model_b(model, path: Path) -> None:
    """Save the fitted pipeline (models/ is not in Git; share it via Google Drive)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model_b(path: Path):
    """Load a pipeline saved by save_model_b, or None when the file is missing."""
    return joblib.load(path) if path.exists() else None


@placeholder
def segment_b(image: np.ndarray, model, params: dict) -> tuple[np.ndarray, np.ndarray]:
    """Segment cracks by scoring overlapping patches.

    TODO(Bowen): slide a params["patch"] window with params["stride"], score each
    window with model.predict_proba(features)[:, 1], average overlapping scores
    into a confidence map and threshold it. The placeholder returns an empty mask.
    """
    height, width = image.shape[:2]
    confidence = np.zeros((height, width), np.float32)
    return confidence >= params["threshold"], confidence
