"""Evaluate every method on one split and write the report tables and figures.

Usage:
  python scripts/run_evaluation.py --split val
  python scripts/run_evaluation.py --split val --limit 5              # quick check
  python scripts/run_evaluation.py --split val --examples id1,id2,id3 # choose overlay images
  python scripts/run_evaluation.py --split test                       # after the final run

Inputs : splits/splits.json, reference masks, results/<method>/<split>/*.png,
         params/severity.json (computed once from TRAINING masks; recompute it
         with --recompute-severity after measure.py changes)
Outputs: eval/<split>/tables/  segmentation.csv, width_errors.csv, error_by_grade.csv,
                               severity_matrix.csv, timing.csv, per_image.csv, per_crack.csv
         eval/<split>/figures/ overlays.png, severity_matrices.png, score_vs_width_error.png
         log/environment.txt
Author : Seungmin Lee
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config, data, evaluate, measure, utils, viz  # noqa: E402

plt = viz.plt
GRADES = list(config.SEVERITY_LABELS)
CRACK_COLUMNS = ["method", "image_id", "status", "ref_id", "pred_id", "ref_mean_width", "ref_max_width",
                 "ref_length", "pred_mean_width", "pred_max_width", "pred_length", "mean_width_error",
                 "max_width_error", "length_rel_error", "ref_grade", "pred_grade"]


# --- Inputs ---------------------------------------------------------------------------

def severity_thresholds(train_ids: list[str], recompute: bool) -> tuple[float, float]:
    """Severity thresholds fixed from TRAINING reference masks (params/severity.json)."""
    path = config.PARAMS_DIR / "severity.json"
    if path.exists() and not recompute:
        saved = json.loads(path.read_text(encoding="utf-8"))
        return saved["low"], saved["high"]
    print(f"Computing severity thresholds from {len(train_ids)} training masks (once)...")
    widths = []
    for image_id in train_ids:
        _, ref = data.load_pair(image_id)
        cracks, _ = measure.measure_cracks(ref)
        widths.extend(crack["max_width_px"] for crack in cracks)
    if not widths:
        sys.exit("No cracks found in the training masks: check the dataset and the mask threshold.")
    low, high = measure.severity_thresholds(widths)
    utils.save_params("severity", {"low": low, "high": high,
                                   "source": "training reference max widths, measure.severity_thresholds"})
    return low, high


def available_methods(split: str, ids: list[str], requested: list[str]) -> list[str]:
    """Methods whose masks exist for every requested image."""
    methods = []
    for method in requested:
        missing = sum(not utils.result_path(method, split, image_id).exists() for image_id in ids)
        if missing:
            print(f"Skipping Method {method}: {missing} of {len(ids)} masks missing "
                  f"(python scripts/run_method.py --method {method} --split {split})")
        else:
            methods.append(method)
    return methods


# --- Per image and per crack ------------------------------------------------------------

def evaluate_split(split: str, ids: list[str], methods: list[str],
                   thresholds: tuple[float, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pixel metrics per image and measurement errors per reference crack."""
    image_rows, crack_rows = [], []
    for n, image_id in enumerate(ids, 1):
        _, ref = data.load_pair(image_id)
        ref_cracks, ref_labels = measure.measure_cracks(ref)
        for method in methods:
            pred = utils.read_mask(utils.result_path(method, split, image_id))
            scores = evaluate.seg_metrics(pred, ref)
            pred_cracks, pred_labels = measure.measure_cracks(pred)
            matches = evaluate.match_cracks(ref_labels, pred_labels)
            fragments, false_detections = evaluate.unmatched_predictions(matches, ref_labels, pred_labels)
            for row in evaluate.measure_errors(ref_cracks, pred_cracks, matches):
                row["method"], row["image_id"] = method, image_id
                row["ref_grade"] = measure.severity_grade(row["ref_max_width"], thresholds)
                row["pred_grade"] = (measure.severity_grade(row["pred_max_width"], thresholds)
                                     if row["status"] == "matched" else "missed")
                crack_rows.append(row)
            image_rows.append({"method": method, "image_id": image_id, **scores,
                               "ref_cracks": len(ref_cracks), "pred_cracks": len(pred_cracks),
                               "fragments": len(fragments), "false_detections": len(false_detections)})
        print(f"[{n}/{len(ids)}] {image_id}")
    return pd.DataFrame(image_rows), pd.DataFrame(crack_rows, columns=CRACK_COLUMNS)


# --- Report tables ------------------------------------------------------------------------

def segmentation_table(images: pd.DataFrame) -> pd.DataFrame:
    """Dataset-level pixel scores per method (counts summed over images), plus mean image F1."""
    rows = []
    for method, group in images.groupby("method", sort=False):
        counts = {key: int(group[key].sum()) for key in evaluate.COUNT_KEYS}
        rows.append({"method": method, "images": len(group), **evaluate.metrics_from_counts(counts),
                     "mean_image_f1": group["f1"].mean()})
    return pd.DataFrame(rows)


def width_table(cracks: pd.DataFrame, images: pd.DataFrame) -> pd.DataFrame:
    """Detection counts and width / length errors of matched cracks per method."""
    rows = []
    for method in images["method"].unique():
        group = cracks[cracks["method"] == method]
        matched = group[group["status"] == "matched"]
        per_image = images[images["method"] == method]
        rows.append({
            "method": method,
            "ref_cracks": len(group),
            "matched": len(matched),
            "missed": int((group["status"] == "missed").sum()),
            "fragments": int(per_image["fragments"].sum()),
            "false_detections": int(per_image["false_detections"].sum()),
            "mean_width_mae": matched["mean_width_error"].abs().mean(),
            "mean_width_bias": matched["mean_width_error"].mean(),
            "max_width_mae": matched["max_width_error"].abs().mean(),
            "max_width_bias": matched["max_width_error"].mean(),
            "length_rel_mae": matched["length_rel_error"].abs().mean(),
        })
    return pd.DataFrame(rows)


def error_by_grade(cracks: pd.DataFrame) -> pd.DataFrame:
    """Missed cracks and max-width error per reference severity grade (width bin)."""
    rows = []
    for (method, grade), group in cracks.groupby(["method", "ref_grade"], sort=False):
        matched = group[group["status"] == "matched"]
        rows.append({"method": method, "ref_grade": grade, "ref_cracks": len(group),
                     "missed": int((group["status"] == "missed").sum()),
                     "max_width_mae": matched["max_width_error"].abs().mean(),
                     "max_width_bias": matched["max_width_error"].mean()})
    table = pd.DataFrame(rows, columns=["method", "ref_grade", "ref_cracks", "missed",
                                        "max_width_mae", "max_width_bias"])
    table["ref_grade"] = pd.Categorical(table["ref_grade"], GRADES, ordered=True)
    return table.sort_values(["method", "ref_grade"]).reset_index(drop=True)


def severity_matrix(cracks: pd.DataFrame, method: str) -> pd.DataFrame:
    """Reference grade (rows) against predicted grade or "missed" (columns)."""
    group = cracks[cracks["method"] == method]
    matrix = pd.crosstab(group["ref_grade"], group["pred_grade"])
    return matrix.reindex(index=GRADES, columns=GRADES + ["missed"], fill_value=0)


def timing_table(split: str, methods: list[str]) -> pd.DataFrame:
    """Mean processing time per image, from results/<method>/<split>/timing.csv."""
    rows = []
    for method in methods:
        path = config.RESULTS_DIR / method / split / "timing.csv"
        if path.exists():
            times = pd.read_csv(path)
            rows.append({"method": method, "images": len(times), "mean_seconds": times["seconds"].mean(),
                         "note": "refinement only; add Method C for the total" if method == "D" else ""})
    return pd.DataFrame(rows)


# --- Report figures ------------------------------------------------------------------------

def pick_examples(cracks: pd.DataFrame, ids: list[str], requested: str | None) -> list[str]:
    """Images for the overlay figure: the ones given, or the thinnest, median and widest crack."""
    if requested:
        return [image_id for image_id in requested.split(",") if image_id in ids]
    widest = (cracks.drop_duplicates(["image_id", "ref_id"]).dropna(subset=["ref_max_width"])
              .groupby("image_id")["ref_max_width"].max().sort_values())
    if widest.empty:
        return ids[:3]
    picks = [widest.index[0], widest.index[len(widest) // 2], widest.index[-1]]
    return list(dict.fromkeys(picks))


def plot_overlays(split: str, examples: list[str], methods: list[str], path: Path) -> None:
    """Rows = example images, columns = image crop and each method's TP / FP / FN overlay."""
    n_cols = len(methods) + 1
    fig, axes = plt.subplots(len(examples), n_cols, figsize=(2.1 * n_cols, 2.1 * len(examples) + 0.3),
                             squeeze=False)
    for r, image_id in enumerate(examples):
        image, ref = data.load_pair(image_id)
        rows, cols = viz.crop_around(ref)
        axes[r][0].imshow(image[rows, cols])
        axes[r][0].set_ylabel(image_id, fontsize=8)
        for c, method in enumerate(methods, start=1):
            pred = utils.read_mask(utils.result_path(method, split, image_id))
            axes[r][c].imshow(viz.overlay_outcomes(image[rows, cols], pred[rows, cols], ref[rows, cols]))
        for ax in axes[r]:
            viz.hide_axes(ax)
    for c, title in enumerate(["Image"] + [f"Method {method}" for method in methods]):
        axes[0][c].set_title(title)
    viz.outcome_legend(fig)
    viz.save_figure(fig, path)


def plot_severity(cracks: pd.DataFrame, methods: list[str], path: Path) -> None:
    """One severity matrix per method, side by side."""
    fig, axes = plt.subplots(1, len(methods), figsize=(2.6 * len(methods) + 0.6, 3.0), squeeze=False)
    for index, (ax, method) in enumerate(zip(axes[0], methods)):
        matrix = severity_matrix(cracks, method).to_numpy()
        viz.plot_confusion(ax, matrix, GRADES, GRADES + ["missed"], f"Method {method}",
                           show_row_labels=index == 0)
        ax.set_xlabel("Predicted grade")
    axes[0][0].set_ylabel("Reference grade")
    viz.save_figure(fig, path)


def plot_score_vs_error(segmentation: pd.DataFrame, widths: pd.DataFrame, path: Path) -> None:
    """Segmentation score and width error side by side, one row per method.

    Reading across a row shows whether the best segmenter also measures width best.
    """
    table = segmentation.merge(widths, on="method", how="left").set_index("method")
    methods = list(table.index)[::-1]                     # Method A at the top
    f1 = table.loc[methods, "tol_f1"].to_numpy(float)
    mae = table.loc[methods, "max_width_mae"].to_numpy(float)
    rows = np.arange(len(methods))
    fig, (left, right) = plt.subplots(1, 2, sharey=True, figsize=(6.6, 0.5 * len(methods) + 1.0))
    left.barh(rows, f1, height=0.55, color=viz.METHOD_COLORS["A"])
    left.set_xlim(0, 1.2)
    left.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    left.set_title("Tolerance F1 (higher is better)")
    right.barh(rows, np.nan_to_num(mae), height=0.55, color=viz.METHOD_COLORS["A"])
    right.set_xlim(0, (np.nanmax(mae) * 1.35) if np.isfinite(mae).any() and np.nanmax(mae) > 0 else 1.0)
    right.set_title("Max-width MAE, px (lower is better)")
    for ax, values in ((left, f1), (right, mae)):
        ax.grid(axis="y", visible=False)
        for y, value in zip(rows, values):
            label = "no matched cracks" if np.isnan(value) else f"{value:.2f}"
            ax.text(0 if np.isnan(value) else value, y, f"  {label}", va="center", fontsize=8,
                    color=viz.INK["primary"])
    left.set_yticks(rows, [f"Method {method}" for method in methods])
    viz.save_figure(fig, path)


# --- Main ------------------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--methods", default=",".join(config.METHODS), help="comma-separated, e.g. A,C,D")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N images")
    parser.add_argument("--examples", default=None, help="comma-separated image IDs for overlays.png")
    parser.add_argument("--recompute-severity", action="store_true",
                        help="recompute params/severity.json from the training masks")
    args = parser.parse_args()

    utils.set_seed()
    viz.apply_style()
    split = data.load_split()
    ids = split[args.split][: args.limit]
    requested = [method.strip() for method in args.methods.split(",") if method.strip()]
    methods = available_methods(args.split, ids, requested)
    if not methods:
        sys.exit("No method has masks for this split yet: run scripts/run_method.py first.")

    thresholds = severity_thresholds(split["train"], args.recompute_severity)
    images, cracks = evaluate_split(args.split, ids, methods, thresholds)

    tables = config.EVAL_DIR / args.split / "tables"
    figures = config.EVAL_DIR / args.split / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    segmentation = segmentation_table(images)
    widths = width_table(cracks, images)
    matrices = pd.concat([severity_matrix(cracks, m).rename_axis("ref_grade").reset_index().assign(method=m)
                          for m in methods])
    outputs = {
        "segmentation.csv": segmentation,
        "width_errors.csv": widths,
        "error_by_grade.csv": error_by_grade(cracks),
        "severity_matrix.csv": matrices,
        "timing.csv": timing_table(args.split, methods),
        "per_image.csv": images,
        "per_crack.csv": cracks,
    }
    for name, table in outputs.items():
        table.to_csv(tables / name, index=False, float_format="%.4f")

    plot_overlays(args.split, pick_examples(cracks, ids, args.examples), methods, figures / "overlays.png")
    plot_severity(cracks, methods, figures / "severity_matrices.png")
    plot_score_vs_error(segmentation, widths, figures / "score_vs_width_error.png")
    utils.record_environment()

    with pd.option_context("display.width", 140, "display.max_columns", 20, "display.precision", 3):
        print("\nSegmentation (dataset level)\n", segmentation.drop(columns=["images"]).to_string(index=False))
        print("\nWidth and length errors (matched cracks, pixels)\n", widths.to_string(index=False))
    print(f"\nTables: {tables.relative_to(ROOT)}  Figures: {figures.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
