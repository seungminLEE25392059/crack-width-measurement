"""Resolution experiment with frozen segmentation parameters.

Usage (from repository root):
  python scripts/run_resolution.py --self-test
  python scripts/run_resolution.py --method C --split val --limit 2
  python scripts/run_resolution.py --method C --split test --final-resolution

Downscale RGB images using INTER_AREA, re-run the selected method at scales
1/2, 1/4, 1/8, and divide predicted widths/lengths by the requested scale.
Match nearest-neighbour upsampled prediction LABELS to full-resolution reference
labels using the team's existing matching rule (which permits many-to-one
matches). Never resize an existing prediction to simulate re-segmentation.
The reference measurements always come from the original full-size mask.

Validation is the default. Test resolution runs are a separate, preplanned
experiment AFTER the team's final baseline run, with settings frozen. They
require --final-resolution, frozen parameter files, and complete baseline test
masks plus matching params_used.json. Scale 1 reuses those final masks.
No threshold tuning or severity fitting is performed here.

Outputs: eval/<split>/resolution/<method>/resolution.csv (per reference crack),
images.csv (detection counts), summary.csv, resolution.png and settings.json.
Errors are prediction minus reference, in full-resolution pixels. MAEs use
matched reference cracks only; always read them alongside missed counts.
Author: Junsen Liang
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pandas as pd
from src import config, data, evaluate, measure, utils

SCALES = (1.0, 0.5, 0.25, 0.125)
ERROR_COLUMNS = ["status", "ref_id", "pred_id", "ref_mean_width", "ref_max_width",
                 "ref_length", "pred_mean_width", "pred_max_width", "pred_length",
                 "mean_width_error", "max_width_error", "length_rel_error"]


def scaled_image(image: np.ndarray, scale: float) -> np.ndarray:
    """Use explicit positive target dimensions, even for tiny input images."""
    h, w = image.shape[:2]
    return cv2.resize(image, (max(1, int(round(w * scale))),
                             max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)


def compare_masks(ref_cracks, ref_labels, prediction, scale):
    """Measure at native small resolution; upsample labels ONLY for matching."""
    prediction = np.asarray(prediction)
    if prediction.ndim != 2 or prediction.dtype != np.bool_:
        raise ValueError("prediction must be a 2D boolean mask")
    pred_cracks, pred_labels = measure.measure_cracks(prediction)
    converted = [{key: crack[key] for key in measure.CRACK_KEYS} for crack in pred_cracks]
    for crack in converted:
        for key in ("mean_width_px", "max_width_px", "length_px"):
            crack[key] /= scale
        crack["area_px"] /= scale ** 2
    h, w = ref_labels.shape
    # Float64 preserves integer component IDs and is accepted by OpenCV resize.
    full_labels = cv2.resize(pred_labels.astype(np.float64), (w, h),
                             interpolation=cv2.INTER_NEAREST).astype(np.int32)
    matches = evaluate.match_cracks(ref_labels, full_labels)
    rows = evaluate.measure_errors(ref_cracks, converted, matches)
    fragments, false_detections = evaluate.unmatched_predictions(matches, ref_labels, full_labels)
    matched = sum(value is not None for value in matches.values())
    counts = {"ref_cracks": len(ref_cracks), "pred_cracks": len(pred_cracks),
              "matched": matched, "missed": len(ref_cracks) - matched,
              "fragments": len(fragments), "false_detections": len(false_detections)}
    return rows, counts


def prepare_method(method: str, require_frozen: bool):
    """Load only selected method modules. D re-runs C at EACH image scale."""
    names = ["C", "D"] if method == "D" else [method]
    modules, settings = {}, {}
    for name in names:
        module = importlib.import_module(f"src.method_{name.lower()}")
        fn = getattr(module, "refine_d" if name == "D" else f"segment_{name.lower()}")
        if utils.is_placeholder(fn):
            raise RuntimeError(f"Method {name} is still @placeholder; request its completed implementation")
        path = config.PARAMS_DIR / f"{name}.json"
        if require_frozen and not path.is_file():
            raise FileNotFoundError(f"Frozen parameters missing: {path}")
        modules[name] = module
        settings[name] = utils.load_params(name, module.DEFAULT_PARAMS)
    model = None
    model_path = None
    if method == "B":
        model_path = config.MODELS_DIR / "method_b.joblib"
        if not model_path.is_file():
            raise FileNotFoundError(f"Download the model from team Drive to {model_path}")
        model = modules["B"].load_model_b(model_path)
    elif method in ("C", "D"):
        model_path = config.MODELS_DIR / "unet.pt"
        if not model_path.is_file():
            raise FileNotFoundError(f"Download the model from team Drive to {model_path}")
        model = modules["C"].load_unet(model_path)
    if method != "A" and model is None:
        raise RuntimeError(f"Could not load trained model for {method}")

    def segment(image):
        if method == "A":
            mask = modules["A"].segment_a(image, settings["A"])
        elif method == "B":
            mask, _ = modules["B"].segment_b(image, model, settings["B"])
        else:
            mask, _ = modules["C"].segment_c(image, model, settings["C"])
            if method == "D":
                mask = modules["D"].refine_d(mask, settings["D"])
        if not isinstance(mask, np.ndarray) or mask.dtype != bool or mask.shape != image.shape[:2]:
            raise ValueError(f"Method {method} must return bool mask with shape {image.shape[:2]}")
        return mask
    return segment, settings, model_path


def check_final_baseline(method, ids, settings):
    """Prevent test-resolution runs before the complete frozen baseline exists."""
    for name, params in settings.items():
        saved = config.RESULTS_DIR / name / "test" / "params_used.json"
        if not saved.is_file() or json.loads(saved.read_text(encoding="utf-8")) != params:
            raise RuntimeError(f"Missing or different final-run parameters: {saved}")
        missing = [i for i in ids if not utils.result_path(name, "test", i).is_file()]
        if missing:
            raise FileNotFoundError(f"Final Method {name} test masks missing, first IDs: {missing[:5]}")


def summary_table(cracks, images):
    """Aggregate detection counts and matched-crack errors separately."""
    rows = []
    for scale in SCALES:
        group = cracks[(cracks["scale"] == scale) & (cracks["status"] == "matched")]
        totals = images[images["scale"] == scale]
        row = {"scale": scale, "images": len(totals)}
        for key in ("ref_cracks", "pred_cracks", "matched", "missed", "fragments", "false_detections"):
            row[key] = int(totals[key].sum())
        row["match_rate"] = row["matched"] / row["ref_cracks"] if row["ref_cracks"] else np.nan
        for source, target in (("mean_width_error", "mean_width_mae_px"),
                               ("max_width_error", "max_width_mae_px"),
                               ("length_error", "length_mae_px"),
                               ("length_rel_error", "length_mare")):
            values = pd.to_numeric(group[source], errors="coerce").dropna()
            row[target] = float(values.abs().mean()) if len(values) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def plot_summary(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.arange(len(SCALES))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(x, summary["mean_width_mae_px"], "o-", label="Mean width")
    axes[0].plot(x, summary["max_width_mae_px"], "s-", label="Maximum width")
    axes[0].set_ylabel("Matched-crack MAE (original px)")
    axes[0].legend()
    axes[1].plot(x, summary["length_mae_px"], "o-")
    axes[1].set_ylabel("Matched-crack length MAE (original px)")
    axes[2].plot(x, summary["match_rate"], "o-")
    axes[2].set_ylabel("Reference-crack match rate")
    axes[2].set_ylim(-0.02, 1.02)
    for ax in axes:
        ax.set_xticks(x, ["1", "1/2", "1/4", "1/8"])
        ax.set_xlabel("Resolution scale")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def self_test():
    """Synthetic checks only; no dataset, model or split is read."""
    ref = np.zeros((64, 64), bool)
    ref[5:12, 5:40] = True
    ref[40:45, 10:50] = True
    cracks, labels = measure.measure_cracks(ref)
    rows, counts = compare_masks(cracks, labels, ref, 1.0)
    assert counts["matched"] == 2 and counts["missed"] == 0
    assert all(row["mean_width_error"] == 0 and row["length_rel_error"] == 0 for row in rows)
    rows, counts = compare_masks(cracks, labels, np.zeros((32, 32), bool), 0.5)
    assert counts["matched"] == 0 and counts["missed"] == 2
    assert all(row["status"] == "missed" for row in rows)
    small = np.zeros((32, 32), bool); small[3:7, 3:20] = True
    native, _ = measure.measure_cracks(small)
    rows, counts = compare_masks(cracks, labels, small, 0.5)
    assert counts["matched"] == 1
    assert np.isclose(rows[0]["pred_max_width"], native[0]["max_width_px"] * 2)
    assert np.isclose(rows[0]["pred_length"], native[0]["length_px"] * 2)
    empty_cracks, empty_labels = measure.measure_cracks(np.zeros_like(ref))
    rows, counts = compare_masks(empty_cracks, empty_labels, ref, 1.0)
    assert rows == [] and counts["false_detections"] == 2
    assert scaled_image(np.zeros((3, 5, 3), np.uint8), 0.125).shape == (1, 1, 3)
    print("PASS: identical masks, missed cracks, scale conversion, false detections and tiny dimensions")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", choices=config.METHODS, default="C")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--limit", type=int, help="first N split IDs; smoke check only")
    parser.add_argument("--final-resolution", action="store_true", help="confirm planned post-final-test resolution experiment")
    parser.add_argument("--self-test", action="store_true", help="synthetic checks without dataset access")
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.split == "test" and not args.final_resolution:
        parser.error("run test resolution experiments only AFTER the team's final run; add --final-resolution")
    if utils.is_placeholder(measure.measure_cracks):
        parser.error("pull Junsen's completed src/measure.py first")
    utils.set_seed()
    try:
        all_ids = data.load_split()[args.split]
        if not all_ids:
            raise ValueError("Selected split is empty")
        segment, settings, model_path = prepare_method(args.method, args.split == "test")
        if args.split == "test":
            check_final_baseline(args.method, all_ids, settings)
        ids = all_ids[:args.limit]
        out = config.EVAL_DIR / args.split / "resolution" / args.method
        if args.limit is not None:
            out = out / f"smoke_{args.limit}"
        if out.exists() and any(out.iterdir()):
            raise FileExistsError(f"Output already exists: {out}. Preserve it before starting a new run.")
        out.mkdir(parents=True, exist_ok=True)
        record = {"utc_started": datetime.now(timezone.utc).isoformat(), "method": args.method,
                  "split": args.split, "image_ids": ids, "scales": SCALES, "parameters": settings,
                  "units": "original-resolution pixels", "width_convention": "2 * EDT",
                  "parameter_status": "frozen" if args.split == "test" else "validation development",
                  "rounding": "target sizes rounded to nearest integer; dimensions >= 1; nominal scale used for metric conversion",
                  "matching": "team maximum-overlap rule; many references may share one prediction",
                  "script_sha256": sha256_file(__file__), "status": "running"}
        if model_path:
            record["model_sha256"] = sha256_file(model_path)
        record["split_sha256"] = sha256_file(config.SPLIT_FILE)
        record["measure_sha256"] = sha256_file(measure.__file__)
        (out / "settings.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        utils.record_environment(out / "environment.txt")
        image_rows, crack_rows = [], []
        for index, image_id in enumerate(ids, 1):
            image, reference = data.load_pair(image_id)
            ref_cracks, ref_labels = measure.measure_cracks(reference)
            h, w = reference.shape
            for scale in SCALES:
                small = image if scale == 1.0 else scaled_image(image, scale)
                with utils.timer() as clock:
                    reused = args.split == "test" and scale == 1.0
                    pred = utils.read_mask(utils.result_path(args.method, "test", image_id)) if reused else segment(small)
                    if pred.shape != small.shape[:2]:
                        raise ValueError(f"Wrong prediction shape for {image_id} at {scale}")
                    rows, counts = compare_masks(ref_cracks, ref_labels, pred, scale)
                meta = {"image_id": image_id, "method": args.method, "scale": scale}
                for row in rows:
                    row["length_error"] = row.get("pred_length", np.nan) - row["ref_length"]
                    crack_rows.append({**meta, **row})
                image_rows.append({**meta, **counts, "seconds_including_measurement": clock["seconds"],
                                   "baseline_mask_reused": reused, "height": h, "width": w,
                                   "small_height": small.shape[0], "small_width": small.shape[1],
                                   "actual_scale_y": small.shape[0] / h, "actual_scale_x": small.shape[1] / w})
                # Checkpoint completed image/scale rows for slow model inference.
                pd.DataFrame(crack_rows, columns=["image_id", "method", "scale"] + ERROR_COLUMNS + ["length_error"]).to_csv(out / "resolution.csv", index=False)
                pd.DataFrame(image_rows).to_csv(out / "images.csv", index=False)
                print(f"[{index}/{len(ids)}] {image_id} scale={scale:g}: matched={counts['matched']}, missed={counts['missed']}", flush=True)
        cracks, images = pd.read_csv(out / "resolution.csv"), pd.read_csv(out / "images.csv")
        summary = summary_table(cracks, images)
        summary.to_csv(out / "summary.csv", index=False)
        plot_summary(summary, out / "resolution.png")
        record["status"] = "complete"
        record["utc_completed"] = datetime.now(timezone.utc).isoformat()
        (out / "settings.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(summary.to_string(index=False))
        print(f"Saved resolution experiment to {out}")
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Resolution experiment stopped: {error}\n")


if __name__ == "__main__":
    main()
