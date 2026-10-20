"""Evaluation: pixel segmentation scores, crack matching and measurement errors.

Implements the evaluation design of the Task 1 report (Section 6):
  * precision, recall, F1 (= Dice) and IoU per pixel, plus a tolerance F1 in
    which a boundary displaced by up to TOLERANCE_PX pixels is not counted as
    both a false positive and a false negative (6.3);
  * precision-recall curves for methods with a continuous confidence output,
    built from 8-bit score histograms (6.3);
  * each reference crack is matched to the predicted crack that overlaps it
    most; unmatched reference cracks are missed; unmatched predictions get no
    width error and are counted as fragments (they overlap a reference crack,
    e.g. one crack broken in two) or false detections (they overlap none) (6.4);
  * width and length errors are prediction minus reference, in pixels (6.4);
  * severity grades from width thresholds fixed on the TRAINING split (6.4).

Crack instances: every 8-connected component of at least MIN_CRACK_AREA_PX
pixels, in predicted and reference masks alike. Smaller specks (JPEG noise in
the reference masks, isolated noisy pixels in predictions) are left out of the
crack-level statistics only; pixel metrics always use every pixel.

Inputs : predicted and reference bool masks, label images, per-crack measurements
Outputs: metric dictionaries, per-crack error rows, PR curves, severity grades
Author : Seungmin Lee
"""
from __future__ import annotations

import json

import cv2
import numpy as np

from . import config, data, measure, utils

COUNT_KEYS = ("tp", "fp", "fn", "n_pred", "n_ref", "tol_pred_hits", "tol_ref_hits")
MIN_CRACK_AREA_PX = 20         # smaller components are noise, not crack instances
UNMEASURED = "unmeasured"      # grade of a crack whose width could not be measured (NaN)
SCORE_LEVELS = 256             # confidence maps are stored as 8-bit PNG (utils.save_prob)


# --- 1. Pixel metrics (Section 6.3) --------------------------------------------------------

def within_distance(mask: np.ndarray, tol: float) -> np.ndarray:
    """Pixels whose Euclidean distance to the nearest True pixel of `mask` is <= tol.

    Dilation with the disk of offsets |d| <= tol gives exactly the same set as
    thresholding a distance transform, and is much faster on large photographs.
    """
    radius = int(np.floor(tol))
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    disk = ((yy ** 2 + xx ** 2) <= tol ** 2).astype(np.uint8)
    return cv2.dilate(mask.astype(np.uint8), disk).astype(bool)


def confusion_counts(pred: np.ndarray, ref: np.ndarray, tol: float = config.TOLERANCE_PX) -> dict[str, int]:
    """Pixel counts behind every segmentation metric.

    Strict counts compare pixels one to one. Tolerance counts accept a predicted
    pixel within `tol` pixels of the reference (precision side) and a reference
    pixel within `tol` pixels of the prediction (recall side).
    """
    pred = np.asarray(pred, bool)
    ref = np.asarray(ref, bool)
    if pred.shape != ref.shape:
        raise ValueError(f"Prediction {pred.shape} and reference {ref.shape} differ in size")
    return {
        "tp": int(np.count_nonzero(pred & ref)),
        "fp": int(np.count_nonzero(pred & ~ref)),
        "fn": int(np.count_nonzero(~pred & ref)),
        "n_pred": int(np.count_nonzero(pred)),
        "n_ref": int(np.count_nonzero(ref)),
        "tol_pred_hits": int(np.count_nonzero(pred & within_distance(ref, tol))),
        "tol_ref_hits": int(np.count_nonzero(ref & within_distance(pred, tol))),
    }


def _ratio(numerator: float, denominator: float, if_empty: float) -> float:
    return numerator / denominator if denominator > 0 else if_empty


def metrics_from_counts(counts: dict[str, int]) -> dict[str, float]:
    """Precision, recall, F1 (= Dice), IoU and tolerance scores from pixel counts.

    Pass counts summed over a split for dataset-level (micro-averaged) scores.
    When prediction and reference are both empty, every score is 1.
    """
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    both_empty = 1.0 if counts["n_pred"] == 0 and counts["n_ref"] == 0 else 0.0
    tol_precision = _ratio(counts["tol_pred_hits"], counts["n_pred"], both_empty)
    tol_recall = _ratio(counts["tol_ref_hits"], counts["n_ref"], both_empty)
    return {
        "precision": _ratio(tp, tp + fp, both_empty),
        "recall": _ratio(tp, tp + fn, both_empty),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn, 1.0),
        "iou": _ratio(tp, tp + fp + fn, 1.0),
        "tol_precision": tol_precision,
        "tol_recall": tol_recall,
        "tol_f1": _ratio(2 * tol_precision * tol_recall, tol_precision + tol_recall, 0.0),
    }


def seg_metrics(pred: np.ndarray, ref: np.ndarray, tol: float = config.TOLERANCE_PX) -> dict[str, float]:
    """All pixel metrics for one image; the raw counts are included for summing."""
    counts = confusion_counts(pred, ref, tol)
    return {**counts, **metrics_from_counts(counts)}


# --- 2. Precision-recall curves (Section 6.3) -------------------------------------------------

def score_histograms(score: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """How many crack and background pixels sit at each of the 256 score levels.

    Summing these histograms over images gives the whole PR curve of a split
    without keeping any score map in memory.
    """
    levels = np.clip(np.rint(np.asarray(score, float) * (SCORE_LEVELS - 1)), 0, SCORE_LEVELS - 1)
    levels = levels.astype(np.int64)
    ref = np.asarray(ref, bool)
    crack = np.bincount(levels[ref], minlength=SCORE_LEVELS)
    background = np.bincount(levels[~ref], minlength=SCORE_LEVELS)
    return crack, background


def pr_curve(crack_hist: np.ndarray, background_hist: np.ndarray) -> dict[str, np.ndarray]:
    """Precision, recall and F1 at every threshold k / 255 ("crack" when score >= k / 255).

    A higher threshold accepts fewer pixels: precision usually rises and recall
    falls. With no pixel accepted, precision is taken as 1 (the usual convention).
    """
    tp = np.cumsum(crack_hist[::-1])[::-1].astype(float)        # crack pixels with level >= k
    fp = np.cumsum(background_hist[::-1])[::-1].astype(float)   # background pixels with level >= k
    n_ref = float(crack_hist.sum())
    precision = np.divide(tp, tp + fp, out=np.ones_like(tp), where=(tp + fp) > 0)
    recall = tp / n_ref if n_ref > 0 else np.zeros_like(tp)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=(precision + recall) > 0)
    return {"threshold": np.arange(len(tp)) / (len(tp) - 1), "precision": precision,
            "recall": recall, "f1": f1}


def average_precision(curve: dict[str, np.ndarray]) -> float:
    """Area under the PR curve: sum over thresholds of (recall gained) x precision."""
    recall = np.append(curve["recall"], 0.0)          # threshold above the top level: nothing accepted
    gained = recall[:-1] - recall[1:]                  # recall added when the threshold drops one level
    return float(np.sum(gained * curve["precision"]))


# --- 3. Crack instances, matching and errors (Section 6.4) --------------------------------------

def keep_cracks(cracks: list[dict], labels: np.ndarray,
                min_area: int = MIN_CRACK_AREA_PX) -> tuple[list[dict], np.ndarray]:
    """Drop components smaller than `min_area` pixels from a measurement result.

    Their ids are set to 0 (background) in the label image; the other ids keep
    their values, so they still match the crack dicts.
    """
    small = [crack["id"] for crack in cracks if crack["area_px"] < min_area]
    if not small:
        return cracks, labels
    lookup = np.arange(int(labels.max()) + 1, dtype=labels.dtype)
    lookup[small] = 0
    return [crack for crack in cracks if crack["area_px"] >= min_area], lookup[labels]


def measure_mask(mask: np.ndarray, min_area: int = MIN_CRACK_AREA_PX) -> tuple[list[dict], np.ndarray, int]:
    """measure.measure_cracks followed by keep_cracks.

    Returns (cracks, labels, number of components ignored as too small).
    Every script measures masks through this function, so predicted and
    reference masks always follow the same crack definition.
    """
    cracks, labels = measure.measure_cracks(mask)
    kept, labels = keep_cracks(cracks, labels, min_area)
    return kept, labels, len(cracks) - len(kept)


def image_max_width(cracks: list[dict]) -> float:
    """Largest max width of any crack in one image (NaN when no crack was measured)."""
    widths = [crack["max_width_px"] for crack in cracks if np.isfinite(crack["max_width_px"])]
    return float(max(widths)) if widths else float("nan")


def match_cracks(ref_labels: np.ndarray, pred_labels: np.ndarray) -> dict[int, int | None]:
    """Match each reference crack to the predicted crack that overlaps it most.

    Returns {reference id: predicted id, or None when no predicted crack overlaps
    it (a missed crack)}. Ties go to the lower predicted id. Ids removed by
    keep_cracks (label 0) never appear.
    """
    ref_ids = np.unique(ref_labels[ref_labels > 0])
    matches: dict[int, int | None] = {int(ref_id): None for ref_id in ref_ids}
    overlap = (ref_labels > 0) & (pred_labels > 0)
    if not overlap.any():
        return matches
    pairs = np.stack([ref_labels[overlap], pred_labels[overlap]]).astype(np.int64)
    unique_pairs, counts = np.unique(pairs, axis=1, return_counts=True)
    for index in np.argsort(-counts, kind="stable"):
        ref_id, pred_id = int(unique_pairs[0, index]), int(unique_pairs[1, index])
        if matches[ref_id] is None:
            matches[ref_id] = pred_id
    return matches


def unmatched_predictions(matches: dict[int, int | None], ref_labels: np.ndarray,
                          pred_labels: np.ndarray) -> tuple[list[int], list[int]]:
    """Predicted cracks that no reference crack was matched to, split in two.

    Returns (fragments, false_detections): fragments overlap some reference crack
    (for example a crack broken into pieces by erosion); false detections
    overlap no reference crack at all.
    """
    used = {pred_id for pred_id in matches.values() if pred_id is not None}
    touching = set(np.unique(pred_labels[(pred_labels > 0) & (ref_labels > 0)]).tolist())
    unmatched = [int(pred_id) for pred_id in np.unique(pred_labels[pred_labels > 0]) if pred_id not in used]
    fragments = [pred_id for pred_id in unmatched if pred_id in touching]
    false_detections = [pred_id for pred_id in unmatched if pred_id not in touching]
    return fragments, false_detections


def measure_errors(ref_cracks: list[dict], pred_cracks: list[dict],
                   matches: dict[int, int | None]) -> list[dict]:
    """One row per reference crack with its measurement errors.

    Matched cracks get width errors (prediction - reference, pixels) and a length
    error relative to the reference length; missed cracks get status "missed"
    and no errors.
    """
    pred_by_id = {crack["id"]: crack for crack in pred_cracks}
    rows = []
    for ref in ref_cracks:
        pred_id = matches.get(ref["id"])
        row = {
            "status": "missed" if pred_id is None else "matched",
            "ref_id": ref["id"],
            "pred_id": pred_id,
            "ref_area": ref["area_px"],
            "ref_mean_width": ref["mean_width_px"],
            "ref_max_width": ref["max_width_px"],
            "ref_length": ref["length_px"],
        }
        if pred_id is not None:
            pred = pred_by_id[pred_id]
            row.update({
                "pred_mean_width": pred["mean_width_px"],
                "pred_max_width": pred["max_width_px"],
                "pred_length": pred["length_px"],
                "mean_width_error": pred["mean_width_px"] - ref["mean_width_px"],
                "max_width_error": pred["max_width_px"] - ref["max_width_px"],
                "length_rel_error": (pred["length_px"] - ref["length_px"]) / max(ref["length_px"], 1.0),
            })
        rows.append(row)
    return rows


# --- 4. Severity grades (Sections 5.5 and 6.4) ---------------------------------------------------

def grade(width_px: float, thresholds: tuple[float, float]) -> str:
    """measure.severity_grade, but "unmeasured" for a NaN width instead of a wrong grade."""
    if width_px is None or not np.isfinite(width_px):
        return UNMEASURED
    return measure.severity_grade(float(width_px), thresholds)


def severity_thresholds_from_train(train_ids: list[str], recompute: bool = False) -> tuple[float, float]:
    """Pixel thresholds (low, high) between minor / moderate / severe.

    Computed ONCE from the reference max widths of the TRAINING split, then
    read from params/severity.json, so validation and test images never
    influence them. They are recomputed automatically when the crack definition
    changed since they were saved (minimum crack area, or the placeholder
    measure.py replaced by the real one), and on request (recompute=True).
    """
    path = config.PARAMS_DIR / "severity.json"
    setting = {"min_crack_area_px": MIN_CRACK_AREA_PX,
               "measure_placeholder": utils.is_placeholder(measure.measure_cracks)}
    if path.exists() and not recompute:
        saved = json.loads(path.read_text(encoding="utf-8"))
        if all(saved.get(key) == value for key, value in setting.items()):
            return saved["low"], saved["high"]
        print("Crack definition changed since params/severity.json was saved: recomputing it.")
    print(f"Computing severity thresholds from {len(train_ids)} training masks (once)...")
    widths = []
    for image_id in train_ids:
        _, ref = data.load_pair(image_id)
        cracks, _, _ = measure_mask(ref)
        widths.extend(crack["max_width_px"] for crack in cracks if np.isfinite(crack["max_width_px"]))
    if not widths:
        raise RuntimeError("No cracks found in the training masks: check the dataset and the mask threshold.")
    low, high = measure.severity_thresholds(widths)
    utils.save_params("severity", {"low": low, "high": high, "cracks": len(widths), **setting,
                                   "source": "training reference max widths, measure.severity_thresholds"})
    return low, high