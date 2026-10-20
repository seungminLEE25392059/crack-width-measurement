"""Smoke-test every shared function on a synthetic crack image.

Usage  : python scripts/check_interfaces.py
Checks that each function accepts the agreed inputs and returns the agreed
types and shapes, and lists the functions still marked @placeholder. Run it
before every push; the exit code is 1 when a check fails.
Author : Seungmin Lee
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src import config, data, evaluate, measure, method_a, method_b, method_c, method_d, utils  # noqa: E402

THICKNESS = 5  # width of the synthetic crack, pixels


def synthetic_pair(height: int = 320, width: int = 480, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Grey textured surface with one dark polyline crack THICKNESS pixels wide."""
    rng = np.random.default_rng(seed)
    surface = rng.normal(150, 12, (height, width)).clip(0, 255)
    mask = np.zeros((height, width), np.uint8)
    points = np.array([[20, 40], [180, 110], [300, 250], [460, 290]], np.int32)
    cv2.polylines(mask, [points], isClosed=False, color=255, thickness=THICKNESS)
    surface[mask > 0] = 60
    image = np.repeat(surface[..., None], 3, axis=2).astype(np.uint8)
    return image, mask > 0


IMAGE, REF = synthetic_pair()
SHAPE = REF.shape


def is_mask(x) -> bool:
    return isinstance(x, np.ndarray) and x.dtype == bool and x.shape == SHAPE


def is_unit_map(x) -> bool:
    return (isinstance(x, np.ndarray) and x.shape == SHAPE and np.issubdtype(x.dtype, np.floating)
            and float(x.min()) >= 0.0 and float(x.max()) <= 1.0)


# --- One check per function group: returns (passed or None if skipped, note) ----------------

def check_dataset_files():
    if not config.IMAGE_DIR.is_dir():
        return None, f"skipped: no dataset in {config.DATA_DIR}"
    ids = data.list_ids()
    if not ids:
        return False, "no image/mask pairs found"
    image, mask = data.load_pair(ids[0])
    ok = image.dtype == np.uint8 and image.ndim == 3 and mask.dtype == bool and mask.shape == image.shape[:2]
    return ok, f"{len(ids)} pairs; {ids[0]} is {image.shape[1]}x{image.shape[0]}"


def check_split():
    ids = [f"img{i:03d}" for i in range(458)]
    split = data.make_split(ids)
    sizes = [len(split[name]) for name in ("train", "val", "test")]
    together = set(split["train"]) | set(split["val"]) | set(split["test"])
    return sum(sizes) == len(ids) and len(together) == len(ids), f"458 IDs -> train/val/test {sizes}"


def check_tiles():
    grey = IMAGE[..., 0].astype(np.float32) / 255.0
    tiles, origins, valid = data.make_tiles(grey)
    back = data.stitch_tiles(tiles, origins, grey.shape)
    ok = (tiles.shape[1:] == (config.TILE_SIZE, config.TILE_SIZE) and valid.shape == tiles.shape
          and np.allclose(back, grey, atol=1e-6))
    return ok, (f"{len(tiles)} tiles, stitched back exactly" if ok else "stitched map differs from the input")


def check_method_a():
    mask = method_a.segment_a(IMAGE, method_a.DEFAULT_PARAMS)
    filtered = method_a.filter_components(mask, 50, 3.0)
    return is_mask(mask) and is_mask(filtered), f"crack fraction {mask.mean():.3f}"


def check_method_b():
    features = method_b.extract_features_b(IMAGE[:32, :32])
    model = method_b.load_model_b(config.MODELS_DIR / "method_b.joblib")
    if model is None and not utils.is_placeholder(method_b.segment_b):
        return None, "skipped: no models/method_b.joblib yet (python scripts/train_b.py)"
    mask, confidence = method_b.segment_b(IMAGE, model, method_b.DEFAULT_PARAMS)
    ok = features.ndim == 1 and bool(np.all(np.isfinite(features))) and is_mask(mask) and is_unit_map(confidence)
    return ok, f"{features.size} features per patch"


def check_method_c():
    if importlib.util.find_spec("torch") is None:
        return None, "skipped: PyTorch not installed here (only needed to run Method C)"
    net = method_c.load_unet(config.MODELS_DIR / "unet.pt")
    if net is None and not utils.is_placeholder(method_c.segment_c):
        return None, "skipped: no models/unet.pt yet"
    mask, probability = method_c.segment_c(IMAGE, net, method_c.DEFAULT_PARAMS)
    return is_mask(mask) and is_unit_map(probability), "mask and probability map"


def check_method_d():
    refined = method_d.refine_d(REF, method_d.DEFAULT_PARAMS)
    return is_mask(refined), f"{int(REF.sum() - refined.sum())} pixels removed"


def check_measure():
    cracks, labels = measure.measure_cracks(REF)
    ok = (len(cracks) >= 1 and labels.shape == SHAPE
          and all(set(measure.CRACK_KEYS) <= set(crack) for crack in cracks))
    width = cracks[0]["mean_width_px"] if cracks else float("nan")
    return ok, f"{len(cracks)} crack; mean width {width:.1f} px (drawn {THICKNESS} px)"


def check_severity():
    thresholds = measure.severity_thresholds([2, 3, 4, 5, 6, 7, 8, 9, 10])
    grades = {measure.severity_grade(width, thresholds) for width in (1, 6, 20)}
    ok = thresholds[0] <= thresholds[1] and grades <= set(config.SEVERITY_LABELS)
    return ok, f"thresholds {thresholds[0]:.1f} and {thresholds[1]:.1f} px"


def check_metrics():
    same = evaluate.seg_metrics(REF, REF)
    shifted = evaluate.seg_metrics(np.roll(REF, 1, axis=1), REF)
    ok = (same["f1"] == 1 and same["iou"] == 1 and same["tol_f1"] == 1
          and shifted["f1"] < 1 and shifted["tol_f1"] == 1)
    return ok, f"same mask F1 1.00; 1-px shift F1 {shifted['f1']:.2f}, tolerance F1 {shifted['tol_f1']:.2f}"


def check_pr():
    score = np.where(REF, 0.9, 0.1)          # a perfect score map: every crack pixel above every background pixel
    curve = evaluate.pr_curve(*evaluate.score_histograms(score, REF))
    ap, best = evaluate.average_precision(curve), float(curve["f1"].max())
    return abs(ap - 1) < 1e-9 and abs(best - 1) < 1e-9, f"perfect score map: AP {ap:.2f}, best F1 {best:.2f}"


def check_grade():
    grades = [evaluate.grade(width, (3.0, 6.0)) for width in (float("nan"), 2.0, 9.0)]
    ok = grades[0] == evaluate.UNMEASURED and set(grades[1:]) <= set(config.SEVERITY_LABELS)
    return ok, f"NaN width -> '{grades[0]}', 2 px -> {grades[1]}, 9 px -> {grades[2]}"


def check_matching():
    _, labels, _ = evaluate.measure_mask(REF)
    matches = evaluate.match_cracks(labels, labels)
    return all(ref_id == pred_id for ref_id, pred_id in matches.items()), f"{len(matches)} crack matched to itself"


CHECKS = [
    ("Zitong", (data.list_ids, data.load_pair), check_dataset_files),
    ("Zitong", (data.make_split,), check_split),
    ("Zitong", (data.make_tiles, data.stitch_tiles), check_tiles),
    ("Wenjie", (method_a.segment_a, method_a.filter_components), check_method_a),
    ("Bowen", (method_b.extract_features_b, method_b.segment_b), check_method_b),
    ("Lingyu", (method_c.load_unet, method_c.segment_c), check_method_c),
    ("Lingyu", (method_d.refine_d,), check_method_d),
    ("Junsen", (measure.measure_cracks,), check_measure),
    ("Junsen", (measure.severity_thresholds, measure.severity_grade), check_severity),
    ("Seungmin", (evaluate.seg_metrics,), check_metrics),
    ("Seungmin", (evaluate.pr_curve, evaluate.average_precision), check_pr),
    ("Seungmin", (evaluate.grade,), check_grade),
    ("Seungmin", (evaluate.match_cracks,), check_matching),
]
OWNED_MODULES = (data, method_a, method_b, method_c, method_d, measure)


def short_name(fn) -> str:
    return f"{fn.__module__.split('.')[-1]}.{fn.__name__}"


def main() -> int:
    utils.set_seed()
    failures = 0
    print(f"{'Owner':<9} {'Functions':<52} {'Status':<12} Note")
    for owner, functions, test in CHECKS:
        try:
            passed, note = test()
        except NotImplementedError as error:
            passed, note = None, f"not implemented yet: {error}"
        except Exception as error:  # any other error is a broken interface
            passed, note = False, f"{type(error).__name__}: {error}"
        if passed is None:
            status = "SKIPPED"
        elif not passed:
            status = "FAIL"
            failures += 1
        elif any(utils.is_placeholder(fn) for fn in functions):
            status = "PLACEHOLDER"
        else:
            status = "OK"
        print(f"{owner:<9} {', '.join(short_name(fn) for fn in functions):<52} {status:<12} {note}")

    left = [f"{module.__name__.split('.')[-1]}.{name}" for module in OWNED_MODULES
            for name, fn in vars(module).items() if callable(fn) and utils.is_placeholder(fn)]
    print(f"\nPlaceholders left ({len(left)}): {', '.join(left) if left else 'none'}")
    print("FAILED checks: fix them before pushing." if failures else "All interfaces OK.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())