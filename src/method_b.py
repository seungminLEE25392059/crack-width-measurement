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
from skimage.feature import graycomatrix, graycoprops, hog
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import config, data

DEFAULT_PARAMS = {
    "patch": 32,               # patch side (pixels)
    "stride": 16,              # sliding-window step (8-16 in the plan)
    "svm_C": 1.0,
    "svm_gamma": "scale",
    "threshold": 0.5,          # confidence threshold, chosen on validation data
    "patches_per_class": 2000,
}


def extract_features_b(patch: np.ndarray) -> np.ndarray:
    """Return concatenated HOG shape and GLCM texture features for one patch."""
    grey = patch if patch.ndim == 2 else cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    hog_features = hog(
        grey,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True,
    )
    quantised = (grey // 32).astype(np.uint8)
    matrix = graycomatrix(
        quantised,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=8,
        symmetric=True,
        normed=True,
    )
    glcm_features = np.concatenate([
        graycoprops(matrix, name).ravel()
        for name in ("contrast", "homogeneity", "energy", "correlation")
    ])
    return np.nan_to_num(np.concatenate([hog_features, glcm_features])).astype(np.float32)


def fit_classifier_b(features: np.ndarray, labels: np.ndarray, params: dict):
    """Fit the standardised RBF-SVM used by every Method B training route."""
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.uint8)
    if features.ndim != 2 or len(features) != len(labels) or set(np.unique(labels)) != {0, 1}:
        raise ValueError("features must be 2-D with matching binary labels containing both classes")
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=params["svm_C"],
            gamma=params["svm_gamma"],
            probability=True,
            random_state=config.SEED,
        )),
    ])
    model.fit(features, labels)
    return model


def predict_patch_probabilities_b(model, features: np.ndarray) -> np.ndarray:
    """Return the crack-class probability for each feature vector."""
    crack_index = int(np.flatnonzero(model.classes_ == 1)[0])
    return model.predict_proba(features)[:, crack_index]


def train_b(train_ids: list[str], params: dict):
    """Sample balanced patches from training pairs and fit the SVM pipeline."""
    patch_size = int(params["patch"])
    target = int(params["patches_per_class"])
    if not train_ids or patch_size < 16 or target < 1:
        raise ValueError("train_ids must be non-empty, patch >= 16 and patches_per_class >= 1")

    rng = np.random.default_rng(config.SEED)
    quotas = np.full(len(train_ids), target // len(train_ids), dtype=int)
    quotas[:target % len(train_ids)] += 1
    features: list[np.ndarray] = []
    labels: list[int] = []

    for image_id, quota in zip(train_ids, quotas):
        if quota == 0:
            continue
        image, mask = data.load_pair(image_id)
        height, width = mask.shape
        if height < patch_size or width < patch_size:
            raise ValueError(f"{image_id}: image is smaller than patch size {patch_size}")

        crack_pixels = np.argwhere(mask)
        if crack_pixels.size == 0:
            raise ValueError(f"{image_id}: mask has no crack pixels")

        for _ in range(quota):
            crack_r, crack_c = crack_pixels[rng.integers(len(crack_pixels))]
            row = int(np.clip(crack_r - rng.integers(patch_size), 0, height - patch_size))
            col = int(np.clip(crack_c - rng.integers(patch_size), 0, width - patch_size))
            features.append(extract_features_b(image[row:row + patch_size, col:col + patch_size]))
            labels.append(1)

        sampled = 0
        attempts = 0
        while sampled < quota and attempts < quota * 100:
            attempts += 1
            row = int(rng.integers(0, height - patch_size + 1))
            col = int(rng.integers(0, width - patch_size + 1))
            if mask[row:row + patch_size, col:col + patch_size].any():
                continue
            features.append(extract_features_b(image[row:row + patch_size, col:col + patch_size]))
            labels.append(0)
            sampled += 1
        if sampled != quota:
            raise RuntimeError(f"{image_id}: could not sample {quota} non-crack patches")

    return fit_classifier_b(np.vstack(features), labels, params)


def save_model_b(model, path: Path) -> None:
    """Save the fitted pipeline (models/ is not in Git; share it via Google Drive)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model_b(path: Path):
    """Load a pipeline saved by save_model_b, or None when the file is missing."""
    return joblib.load(path) if path.exists() else None


def segment_b(image: np.ndarray, model, params: dict) -> tuple[np.ndarray, np.ndarray]:
    """Average overlapping patch probabilities into a pixel-sized confidence map."""
    if model is None:
        raise ValueError("Method B requires a fitted model")
    patch_size = int(params["patch"])
    stride = int(params["stride"])
    if patch_size < 16 or not 0 < stride <= patch_size:
        raise ValueError("patch must be >= 16 and stride must be in [1, patch]")

    height, width = image.shape[:2]
    if height < patch_size or width < patch_size:
        pad_h, pad_w = max(0, patch_size - height), max(0, patch_size - width)
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")

    def origins(length: int) -> list[int]:
        last = max(0, length - patch_size)
        starts = list(range(0, last + 1, stride))
        if starts[-1] != last:
            starts.append(last)
        return starts

    rows, cols = origins(image.shape[0]), origins(image.shape[1])
    positions = [(row, col) for row in rows for col in cols]
    feature_matrix = np.vstack([
        extract_features_b(image[row:row + patch_size, col:col + patch_size])
        for row, col in positions
    ])
    probabilities = predict_patch_probabilities_b(model, feature_matrix)

    summed = np.zeros(image.shape[:2], np.float32)
    coverage = np.zeros(image.shape[:2], np.uint16)
    for probability, (row, col) in zip(probabilities, positions):
        summed[row:row + patch_size, col:col + patch_size] += probability
        coverage[row:row + patch_size, col:col + patch_size] += 1
    confidence = np.clip(summed / np.maximum(coverage, 1), 0.0, 1.0)[:height, :width].astype(np.float32)
    return confidence >= params["threshold"], confidence
