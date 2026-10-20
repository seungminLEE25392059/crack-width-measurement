"""Method D: whole-image component filtering then exact radius-1 erosion.

Inputs: Method C bool HxW masks and validation-selected parameters.
Outputs: new bool masks, validation search/ablation tables. Owner: Lingyu Wu.
A's filter is unfinished, so D owns its implementation without modifying A.
Widths use the shared measurement only, when its placeholder is replaced.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion
from skimage.measure import label, regionprops
from . import data, evaluate, measure, method_c, utils

DEFAULT_PARAMS = {"use_filter": True, "min_area": 50, "min_elongation": 3.,
                  "use_erosion": True, "erosion_radius": 1}
EROSION_FOOTPRINT = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
ABLATIONS = {"Baseline": (False, False), "Filtering only": (True, False),
             "Erosion only": (False, True), "Filtering + Erosion": (True, True)}


def filter_components_d(mask: np.ndarray, min_area: int, min_elongation: float) -> np.ndarray:
    """8-connected area AND moment-ellipse elongation filter.

    Degenerate line: infinite ratio; singleton: ratio 1. Include permissive
    candidates since real short cracks and branches may fail these thresholds.
    """
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.dtype != bool:
        raise ValueError("Expected a 2-D bool mask")
    if not np.isfinite(min_area) or min_area < 0 or not np.isfinite(min_elongation) or min_elongation < 0:
        raise ValueError("Area and elongation minima must be finite and nonnegative")
    if mask.size == 0:
        return mask.copy()
    labels = label(mask, connectivity=2)
    keep = np.zeros(int(labels.max())+1, bool)
    for region in regionprops(labels):
        major, minor = region.axis_major_length, region.axis_minor_length
        ratio = major/minor if minor > 1e-8 else (float("inf") if major > 1e-8 else 1.)
        keep[region.label] = region.area >= min_area and ratio >= min_elongation
    return keep[labels]


def refine_d(mask_c: np.ndarray, params: dict) -> np.ndarray:
    """Filter, then erode once on a copy; MATLAB logical erosion pads with True.

    border_value=1 matches imerode maximal-value padding. SciPy's default zero
    padding would unnecessarily erode foreground pixels at the image boundary.
    """
    p = {**DEFAULT_PARAMS, **params}
    if not isinstance(mask_c, np.ndarray) or mask_c.ndim != 2 or mask_c.dtype != bool:
        raise ValueError("Method D expects Method C's 2-D bool mask")
    mask = mask_c.copy()
    if p["erosion_radius"] != 1:
        raise ValueError("Radius is fixed at 1; skip erosion with use_erosion=False")
    if mask.size == 0:
        return mask
    if p["use_filter"]:
        mask = filter_components_d(mask, p["min_area"], p["min_elongation"])
    if p["use_erosion"]:
        mask = binary_erosion(mask, structure=EROSION_FOOTPRINT, iterations=1, border_value=1)
    return mask.astype(bool, copy=False)


def _measure_details(pred: np.ndarray, ref: np.ndarray, reference=None) -> dict:
    """Public shared measurement/matching only; retain misses, FP, fragments and merges.

    A precomputed shared reference result avoids repeating GT EDT/skeleton work
    for every candidate. No private width/length estimator is introduced.
    """
    if utils.is_placeholder(measure.measure_cracks):
        ref_labels, pred_labels = label(ref, connectivity=2), label(pred, connectivity=2)
        rows = []
        zero_length_refs = 0
    else:
        refs, ref_labels = reference if reference is not None else measure.measure_cracks(ref)
        preds, pred_labels = measure.measure_cracks(pred)
        matches = evaluate.match_cracks(ref_labels, pred_labels)
        rows = evaluate.measure_errors(refs, preds, matches)
        zero_length_refs = sum(crack['length_px'] == 0 for crack in refs)
    matches = evaluate.match_cracks(ref_labels, pred_labels)
    fragments, false_detections = evaluate.unmatched_predictions(matches, ref_labels, pred_labels)
    reused = {}
    for pred_id in matches.values():
        if pred_id is not None:
            reused[pred_id] = reused.get(pred_id, 0) + 1
    return {'errors': rows, 'missed': sum(v is None for v in matches.values()),
            'reference_cracks': len(matches), 'false_detections': len(false_detections),
            'fragments': len(fragments), 'merged_predictions': sum(n > 1 for n in reused.values()),
            'merged_extra_references': sum(max(0, n-1) for n in reused.values()),
            'zero_length_references': zero_length_refs}


def _measure_pair(pred: np.ndarray, ref: np.ndarray) -> tuple[list[dict], int, int]:
    """Backward-compatible compact view of the public measurement details."""
    details = _measure_details(pred, ref)
    return details['errors'], details['missed'], details['reference_cracks']


def evaluate_variants(ids: list[str], probability_dir: Path, threshold: float,
                      params: dict, variants: dict | None = None) -> tuple[list[dict], list[dict]]:
    """Validation-only ablations using identical lossless full-image C maps.

    Width/length fields are None while measurement remains a placeholder.
    Post-processing timings exclude the common U-Net inference cost.
    """
    method_c._check_ids(ids, "val")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0,1]")
    variants = variants or {name: {**params, "use_filter": f, "use_erosion": e}
                             for name, (f, e) in ABLATIONS.items()}
    aggregate = {name: {"counts": np.zeros(3, np.int64), "seconds": 0., "errors": [],
                        "missed": 0, "ref_cracks": 0, "false_detections": 0,
                        "fragments": 0, "merged_predictions": 0, "merged_extra_references": 0,
                        "zero_length_references": 0} for name in variants}
    per_image = []
    for image_id in ids:
        _, ref = data.load_pair(image_id)
        prob = np.load(Path(probability_dir)/f"{image_id}_prob.npy", allow_pickle=False)
        if prob.shape != ref.shape or not np.isfinite(prob).all() or prob.min() < 0 or prob.max() > 1:
            raise ValueError(f"Invalid cached probability for {image_id}")
        baseline = prob >= threshold
        reference = None if utils.is_placeholder(measure.measure_cracks) else measure.measure_cracks(ref)
        for name, setting in variants.items():
            p = {**DEFAULT_PARAMS, **setting}
            if not p["use_filter"] and not p["use_erosion"]:
                pred, seconds = baseline, 0.  # C has no refinement cost.
            else:
                start = time.perf_counter()
                pred = refine_d(baseline, setting)
                seconds = time.perf_counter()-start
            counts = method_c._strict_counts(pred, ref)
            details = _measure_details(pred, ref, reference)
            errors, missed, ref_count = details['errors'], details['missed'], details['reference_cracks']
            a = aggregate[name]
            a["counts"] += counts
            a["seconds"] += seconds
            a["errors"].extend(errors)
            a["missed"] += missed
            a["ref_cracks"] += ref_count
            for key in ('false_detections', 'fragments', 'merged_predictions',
                        'merged_extra_references', 'zero_length_references'):
                a[key] += details[key]
            per_image.append({"image_id": image_id, "variant": name, **method_c._scores(counts),
                              "tp": int(counts[0]), "fp": int(counts[1]), "fn": int(counts[2]),
                              "detection_failures": missed, "postprocessing_seconds": seconds,
                              "true_pixels_removed": int((baseline & ~pred & ref).sum()),
                              "foreground_pixels_removed": int((baseline & ~pred).sum()),
                              "prediction_erased": bool(baseline.any() and not pred.any())})
            per_image[-1].update({key: details[key] for key in ('false_detections', 'fragments',
                'merged_predictions', 'merged_extra_references', 'zero_length_references')})
    rows = []
    for name, a in aggregate.items():
        matched = [e for e in a["errors"] if e["status"] == "matched"]
        def mean(key, absolute=False):
            values = [abs(e[key]) if absolute else e[key] for e in matched]
            return float(np.mean(values)) if values else None
        rows.append({"variant": name, **method_c._scores(a["counts"]),
                     "tp": int(a['counts'][0]), "fp": int(a['counts'][1]), "fn": int(a['counts'][2]),
                     "mean_width_mae": mean("mean_width_error", True),
                     "max_width_mae": mean("max_width_error", True),
                     "signed_width_error": mean("mean_width_error"),
                     "max_width_bias": mean("max_width_error"),
                     "relative_length_error": mean("length_rel_error", True),
                     "signed_relative_length_error": mean("length_rel_error"),
                     "detection_failures": a["missed"], "reference_cracks": a["ref_cracks"],
                     "matched_width_samples": len(matched),
                     "false_detections": a['false_detections'], "fragments": a['fragments'],
                     "merged_predictions": a['merged_predictions'],
                     "merged_extra_references": a['merged_extra_references'],
                     "zero_length_references": a['zero_length_references'],
                     "postprocessing_seconds": a["seconds"]/len(ids),
                     "measurement_status": "blocked_shared_placeholder" if utils.is_placeholder(measure.measure_cracks) else "shared"})
    return rows, per_image


def select_width_candidate(rows: list[dict], max_recall_drop: float = 0.02,
                           max_extra_misses: int = 0, max_negative_bias: float = 1.) -> dict | None:
    """Original width objective/constraints, reusable after resumable per-image evaluation.

    The first row is the C baseline. Completely deleted references cannot win
    through a small matched-only MAE: misses and nonempty width samples are checked.
    """
    baseline = rows[0]
    eligible = [row for row in rows[1:] if row['recall'] >= baseline['recall']-max_recall_drop
                and row['detection_failures'] <= baseline['detection_failures']+max_extra_misses
                and (row['max_width_bias'] is None or row['max_width_bias'] >= -max_negative_bias)]
    if not utils.is_placeholder(measure.measure_cracks):
        eligible = [row for row in eligible if row['max_width_mae'] is not None]
        return min(eligible, key=lambda row:(row['max_width_mae'],row['mean_width_mae'],-row['recall'])) if eligible else None
    return max(eligible,key=lambda row:(row['f1'],row['recall'])) if eligible else None


def tune_d(ids: list[str], probability_dir: Path, threshold: float, output_dir: Path,
           areas=(0, 10, 50), elongations=(1., 2., 3.), max_recall_drop: float = 0.02,
           max_extra_misses: int = 0, max_negative_bias: float = 1.) -> tuple[dict, list[dict]]:
    """Limited validation search; width objective once shared measurement is ready.

    Constrain recall, misses and negative max-width bias. Infeasible candidates
    remain diagnostic, never final. Without widths selection is provisional.
    """
    if not areas or not elongations or min(max_recall_drop, max_extra_misses, max_negative_bias) < 0:
        raise ValueError("Provide candidates and nonnegative constraints")
    variants = {"Baseline": {**DEFAULT_PARAMS, "use_filter": False, "use_erosion": False}}
    variants.update({f"area={a},ratio={r}": {**DEFAULT_PARAMS, "min_area": a, "min_elongation": r}
                     for a in areas for r in elongations})
    rows, per_image = evaluate_variants(ids, probability_dir, threshold, DEFAULT_PARAMS, variants)
    widths_ready = not utils.is_placeholder(measure.measure_cracks)
    best = select_width_candidate(rows,max_recall_drop,max_extra_misses,max_negative_bias)
    setting = variants[best["variant"]] if best else variants[next(k for k in variants if k != "Baseline")]
    selected = {**setting, "threshold_c": threshold, "selection_source": "fixed validation split only",
                "selection_ids": list(ids), "selection_feasible": best is not None,
                "selection_complete": set(ids) == set(data.load_split()["val"]) and widths_ready and best is not None,
                "selection_status": "width_selected" if widths_ready and best else
                    "provisional_segmentation_only" if best else "no_candidate_satisfies_constraints",
                "constraints": {"max_recall_drop": max_recall_drop, "max_extra_misses": max_extra_misses,
                                "max_negative_max_width_bias_px": max_negative_bias},
                "erosion_footprint": EROSION_FOOTPRINT.astype(int).tolist(), "erosion_border_value": 1,
                "selected_validation_scores": best}
    method_c._write_json(Path(output_dir)/"d_search.json", {"params": selected, "candidates": rows,
                                                          "per_image": per_image})
    return selected, rows
