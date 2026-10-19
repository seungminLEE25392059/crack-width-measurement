"""Evaluation: pixel segmentation scores, crack matching and measurement errors.

Implements the evaluation design of the Task 1 report:
  * precision, recall, F1 (= Dice) and IoU per pixel, plus a tolerance F1 in
    which a boundary displaced by up to TOLERANCE_PX pixels is not counted as
    both a false positive and a false negative;
  * each reference crack is matched to the predicted crack that overlaps it
    most; unmatched reference cracks are missed; unmatched predictions get no
    width error and are counted as fragments (they overlap a reference crack,
    e.g. one crack broken in two) or false detections (they overlap none);
  * width and length errors are prediction minus reference, in pixels.

Inputs : predicted and reference bool masks, label images, per-crack measurements
Outputs: metric dictionaries and per-crack error rows
Author : Seungmin Lee
"""
from __future__ import annotations

import cv2
import numpy as np

from . import config

COUNT_KEYS = ("tp", "fp", "fn", "n_pred", "n_ref", "tol_pred_hits", "tol_ref_hits")


def _within(mask: np.ndarray, tol: float) -> np.ndarray:
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
        "tol_pred_hits": int(np.count_nonzero(pred & _within(ref, tol))),
        "tol_ref_hits": int(np.count_nonzero(ref & _within(pred, tol))),
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


def match_cracks(ref_labels: np.ndarray, pred_labels: np.ndarray) -> dict[int, int | None]:
    """Match each reference crack to the predicted crack that overlaps it most.

    Returns {reference id: predicted id, or None when no predicted crack overlaps
    it (a missed crack)}. Ties go to the lower predicted id.
    """
    matches: dict[int, int | None] = {ref_id: None for ref_id in range(1, int(ref_labels.max()) + 1)}
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
    unmatched = [pred_id for pred_id in range(1, int(pred_labels.max()) + 1) if pred_id not in used]
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
