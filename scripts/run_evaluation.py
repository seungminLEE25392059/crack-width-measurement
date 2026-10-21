"""Evaluate every method on one split and write the report tables and figures.

Usage:
  python scripts/run_evaluation.py --split val
  python scripts/run_evaluation.py --split val --limit 5              # quick check
  python scripts/run_evaluation.py --split val --examples id1,id2,id3 # choose overlay images
  python scripts/run_evaluation.py --split test                       # after the final run

Inputs : splits/splits.json, reference masks, results/<method>/<split>/*.png
         (and *_prob.png score maps for Methods B and C), params/severity.json
         (computed once from TRAINING masks; recomputed automatically when the
         crack definition changes, or with --recompute-severity)
Outputs: eval/<split>/tables/
           segmentation.csv     pixel precision, recall, F1, IoU, tolerance scores (6.3)
           pr_curves.csv        precision / recall / F1 at every threshold, B and C (6.3)
           pr_summary.csv       average precision, best-F1 threshold, operating point (6.3)
           width_errors.csv     detection counts, width MAE and bias, length error (6.4)
           error_by_grade.csv   the same errors per reference width grade (6.4)
           severity_matrix.csv  reference grade x predicted grade (6.4)
           ranking.csv          F1 rank against width-error rank (6.4, principal comparison)
           success_criteria.csv criteria 1-5 of Section 6.8
           timing.csv           seconds per image (6.7)
           per_image.csv, per_crack.csv   the raw rows behind every table
         eval/<split>/figures/  overlays.png, severity_matrices.png,
                                score_vs_width_error.png, pr_curves.png
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
from matplotlib.lines import Line2D  # noqa: E402

from src import config, data, evaluate, utils, viz  # noqa: E402

plt = viz.plt
GRADES = list(config.SEVERITY_LABELS)
CRACK_COLUMNS = ["method", "image_id", "status", "ref_id", "pred_id", "ref_area", "ref_mean_width",
                 "ref_max_width", "ref_length", "pred_mean_width", "pred_max_width", "pred_length",
                 "mean_width_error", "max_width_error", "length_rel_error", "ref_grade", "pred_grade"]
MARKERS = {"A": "o", "B": "s", "C": "^", "D": "D"}   # second identity channel next to colour

# Success criteria of Section 6.8, written as numbers so they can be checked automatically.
EDGE_DETECTION_RECALL = 0.79   # Dorafshan et al. (2018): edge detectors found 53-79 % of crack pixels
WIDTH_UNCERTAINTY_PX = 2 * config.TOLERANCE_PX  # each annotated edge may be off by about TOLERANCE_PX
MAX_UNDERESTIMATE_PX = -1.0    # a mean signed max-width error below this is systematic underestimation


# --- Inputs ---------------------------------------------------------------------------

def available_methods(split: str, ids: list[str], requested: list[str], kind: str = "mask") -> list[str]:
    """Methods whose outputs of this kind (mask or prob) exist for every requested image."""
    methods = []
    for method in requested:
        missing = sum(not utils.result_path(method, split, image_id, kind).exists() for image_id in ids)
        if not missing:
            methods.append(method)
        elif kind == "mask":
            print(f"Skipping Method {method}: {missing} of {len(ids)} masks missing "
                  f"(python scripts/run_method.py --method {method} --split {split})")
    return methods


def operating_threshold(split: str, method: str) -> float:
    """Threshold the saved masks were made with.

    Read from results/<method>/<split>/params_used.json (written by run_method.py),
    or from params/<method>.json when the masks came from a notebook instead.
    """
    for path in (config.RESULTS_DIR / method / split / "params_used.json", config.PARAMS_DIR / f"{method}.json"):
        if path.exists():
            value = json.loads(path.read_text(encoding="utf-8")).get("threshold")
            return float(value) if isinstance(value, (int, float)) else float("nan")   # e.g. A uses "otsu"
    return float("nan")


# --- Per image and per crack ------------------------------------------------------------

def evaluate_split(split: str, ids: list[str], methods: list[str], scored: list[str],
                   thresholds: tuple[float, float]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Pixel metrics per image, errors per reference crack and score histograms.

    For each image the reference mask is loaded and measured once; then, for
    each method, its mask is scored (pixel metrics), measured with the same
    code, matched crack by crack and turned into error rows. Methods in
    `scored` also add their score map to the PR-curve histograms.
    """
    image_rows, crack_rows = [], []
    histograms = {method: [np.zeros(evaluate.SCORE_LEVELS, np.int64), np.zeros(evaluate.SCORE_LEVELS, np.int64)]
                  for method in scored}
    for n, image_id in enumerate(ids, 1):
        _, ref = data.load_pair(image_id)
        ref_cracks, ref_labels, ref_ignored = evaluate.measure_mask(ref)
        for method in methods:
            pred = utils.read_mask(utils.result_path(method, split, image_id))
            scores = evaluate.seg_metrics(pred, ref)
            pred_cracks, pred_labels, pred_ignored = evaluate.measure_mask(pred)
            matches = evaluate.match_cracks(ref_labels, pred_labels)
            fragments, false_detections = evaluate.unmatched_predictions(matches, ref_labels, pred_labels)
            for row in evaluate.measure_errors(ref_cracks, pred_cracks, matches):
                row["method"], row["image_id"] = method, image_id
                row["ref_grade"] = evaluate.grade(row["ref_max_width"], thresholds)
                row["pred_grade"] = (evaluate.grade(row["pred_max_width"], thresholds)
                                     if row["status"] == "matched" else "missed")
                crack_rows.append(row)
            image_rows.append({"method": method, "image_id": image_id, **scores,
                               "ref_cracks": len(ref_cracks), "pred_cracks": len(pred_cracks),
                               "fragments": len(fragments), "false_detections": len(false_detections),
                               "ref_small_ignored": ref_ignored, "pred_small_ignored": pred_ignored})
            if method in scored:
                score = utils.read_prob(utils.result_path(method, split, image_id, "prob"))
                crack_hist, background_hist = evaluate.score_histograms(score, ref)
                histograms[method] = [histograms[method][0] + crack_hist, histograms[method][1] + background_hist]
        print(f"[{n}/{len(ids)}] {image_id}")
    return pd.DataFrame(image_rows), pd.DataFrame(crack_rows, columns=CRACK_COLUMNS), histograms


# --- Report tables ------------------------------------------------------------------------

def segmentation_table(images: pd.DataFrame) -> pd.DataFrame:
    """Dataset-level pixel scores per method (counts summed over images), plus mean image F1."""
    rows = []
    for method, group in images.groupby("method", sort=False):
        counts = {key: int(group[key].sum()) for key in evaluate.COUNT_KEYS}
        rows.append({"method": method, "images": len(group), **evaluate.metrics_from_counts(counts),
                     "mean_image_f1": group["f1"].mean()})
    return pd.DataFrame(rows)


def pr_tables(histograms: dict, segmentation: pd.DataFrame, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full PR curve per scored method, and one summary row per method.

    best_threshold is the threshold with the highest F1 on THIS split. On the
    validation split it is the candidate operating point; on the test split it
    is reported only, never used to change a parameter.
    """
    curves, summary = [], []
    for method, (crack_hist, background_hist) in histograms.items():
        curve = evaluate.pr_curve(crack_hist, background_hist)
        curves.append(pd.DataFrame({"method": method, **curve}))
        best = int(np.argmax(curve["f1"]))
        used = segmentation.set_index("method").loc[method]
        summary.append({"method": method, "average_precision": evaluate.average_precision(curve),
                        "best_f1": curve["f1"][best], "best_threshold": curve["threshold"][best],
                        "operating_threshold": operating_threshold(split, method),
                        "operating_precision": used["precision"], "operating_recall": used["recall"],
                        "operating_f1": used["f1"]})
    columns = ["method", "threshold", "precision", "recall", "f1"]
    return (pd.concat(curves, ignore_index=True) if curves else pd.DataFrame(columns=columns),
            pd.DataFrame(summary))


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
    for (method, ref_grade), group in cracks.groupby(["method", "ref_grade"], sort=False):
        matched = group[group["status"] == "matched"]
        rows.append({"method": method, "ref_grade": ref_grade, "ref_cracks": len(group),
                     "missed": int((group["status"] == "missed").sum()),
                     "max_width_mae": matched["max_width_error"].abs().mean(),
                     "max_width_bias": matched["max_width_error"].mean()})
    table = pd.DataFrame(rows, columns=["method", "ref_grade", "ref_cracks", "missed",
                                        "max_width_mae", "max_width_bias"])
    table["ref_grade"] = pd.Categorical(table["ref_grade"], GRADES + [evaluate.UNMEASURED], ordered=True)
    return table.sort_values(["method", "ref_grade"]).reset_index(drop=True)


def severity_matrix(cracks: pd.DataFrame, method: str) -> pd.DataFrame:
    """Reference grade (rows) against predicted grade or "missed" (columns).

    An "unmeasured" row or column appears only when some width was NaN.
    """
    group = cracks[cracks["method"] == method]
    rows = GRADES + ([evaluate.UNMEASURED] if (group["ref_grade"] == evaluate.UNMEASURED).any() else [])
    cols = GRADES + ["missed"] + ([evaluate.UNMEASURED] if (group["pred_grade"] == evaluate.UNMEASURED).any()
                                  else [])
    return pd.crosstab(group["ref_grade"], group["pred_grade"]).reindex(index=rows, columns=cols, fill_value=0)


def ranking_table(segmentation: pd.DataFrame, widths: pd.DataFrame) -> pd.DataFrame:
    """Rank 1 = best: highest F1, highest tolerance F1, lowest max-width MAE.

    The principal comparison of Section 6.4 asks whether these rankings agree.
    """
    table = segmentation[["method", "f1", "tol_f1"]].merge(
        widths[["method", "max_width_mae", "max_width_bias"]], on="method", how="left")
    table["rank_f1"] = table["f1"].rank(ascending=False, method="min").astype("Int64")
    table["rank_tol_f1"] = table["tol_f1"].rank(ascending=False, method="min").astype("Int64")
    table["rank_max_width_mae"] = table["max_width_mae"].rank(ascending=True, method="min").astype("Int64")
    return table


def success_criteria(segmentation: pd.DataFrame, widths: pd.DataFrame, cracks: pd.DataFrame,
                     ranking: pd.DataFrame) -> pd.DataFrame:
    """Section 6.8 criteria with the value found and whether the rule is met."""
    rows = []
    best = segmentation.loc[segmentation["f1"].idxmax()]
    rows.append({"criterion": "1 Segmentation", "method": best["method"],
                 "value": f"recall {best['recall']:.3f}, F1 {best['f1']:.3f} (best F1)",
                 "rule": f"recall > {EDGE_DETECTION_RECALL} (top of the edge-detector range)",
                 "met": "yes" if best["recall"] > EDGE_DETECTION_RECALL else "no"})
    measured = widths.dropna(subset=["max_width_mae"])
    if measured.empty:
        rows.append({"criterion": "2 Measurement", "method": "-", "value": "no matched cracks",
                     "rule": "-", "met": "no"})
    else:
        top = measured.loc[measured["max_width_mae"].idxmin()]
        ok = top["max_width_mae"] <= WIDTH_UNCERTAINTY_PX and top["max_width_bias"] >= MAX_UNDERESTIMATE_PX
        rows.append({"criterion": "2 Measurement", "method": top["method"],
                     "value": f"max-width MAE {top['max_width_mae']:.2f} px, bias {top['max_width_bias']:+.2f} px",
                     "rule": f"MAE <= {WIDTH_UNCERTAINTY_PX} px and bias >= {MAX_UNDERESTIMATE_PX} px",
                     "met": "yes" if ok else "no"})
    for method in segmentation["method"]:
        matrix = severity_matrix(cracks, method)
        severe_as_minor = int(matrix.loc["severe", "minor"])
        severe_missed = int(matrix.loc["severe", "missed"])     # reported too: a miss is the worst underestimate
        rows.append({"criterion": "3 Severity", "method": method,
                     "value": f"severe cracks: {severe_as_minor} graded minor, {severe_missed} missed",
                     "rule": "0 severe cracks graded minor", "met": "yes" if severe_as_minor == 0 else "no"})
    rows.append({"criterion": "4 Resolution", "method": "-", "value": "see scripts/run_resolution.py",
                 "rule": "a minimum measurable width can be stated", "met": "-"})
    order = ranking.sort_values("method")
    same = order["rank_f1"].astype(str).tolist() == order["rank_max_width_mae"].astype(str).tolist()
    rows.append({"criterion": "5 Comparison", "method": "all",
                 "value": ("F1 ranks " + ", ".join(f"{m}={r}" for m, r in zip(order["method"], order["rank_f1"]))
                           + "; width ranks " + ", ".join(f"{m}={r}" for m, r in
                                                          zip(order["method"], order["rank_max_width_mae"]))),
                 "rule": "written up in Results and Discussion",
                 "met": "same ranking" if same else "different ranking"})
    return pd.DataFrame(rows)


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
        matrix = severity_matrix(cracks, method)
        viz.plot_confusion(ax, matrix.to_numpy(), list(matrix.index), list(matrix.columns), f"Method {method}",
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


def plot_pr_curves(curves: pd.DataFrame, summary: pd.DataFrame, segmentation: pd.DataFrame, path: Path) -> None:
    """PR curves of the scored methods (lines) and every method's saved masks (markers).

    Grey lines join points of equal F1, so a curve or marker closer to the top
    right is better. A method without a score map (A, D) is a single point.
    """
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    recall = np.linspace(0.01, 1, 200)
    for f1 in (0.2, 0.4, 0.6, 0.8):
        r = recall[recall > f1 / 2 + 1e-6]
        ax.plot(r, f1 * r / (2 * r - f1), color=viz.INK["grid"], lw=0.8, zorder=1)
        ax.text(1.02, f1 / (2 - f1), f" F1 {f1:.1f}", fontsize=7, color=viz.INK["muted"], va="center")
    handles = []
    ap = summary.set_index("method")["average_precision"] if not summary.empty else pd.Series(dtype=float)
    for method, row in segmentation.set_index("method").iterrows():
        color = viz.METHOD_COLORS.get(method, viz.INK["secondary"])
        has_curve = method in ap.index
        if has_curve:
            part = curves[curves["method"] == method]
            ax.plot(part["recall"], part["precision"], color=color, lw=1.6, solid_capstyle="round", zorder=2)
        ax.plot(row["recall"], row["precision"], MARKERS.get(method, "o"), color=color, ms=7,
                mec="white", mew=1.2, zorder=3)
        label = f"Method {method}" + (f" (AP {ap[method]:.2f})" if has_curve else " (mask only)")
        handles.append(Line2D([], [], color=color, lw=1.6 if has_curve else 0, marker=MARKERS.get(method, "o"),
                              ms=7, mec="white", mew=1.2, label=label))
    ax.set_xlim(0, 1.02)                                 # room for markers at recall 1
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall: lines = all thresholds,\nmarkers = the saved masks", loc="left")
    ax.legend(handles=handles, loc="lower left", fontsize=8)
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
    scored = available_methods(args.split, ids, methods, kind="prob")

    thresholds = evaluate.severity_thresholds_from_train(split["train"], args.recompute_severity)
    print(f"Severity thresholds: minor < {thresholds[0]:.2f} px <= moderate < {thresholds[1]:.2f} px <= severe")
    images, cracks, histograms = evaluate_split(args.split, ids, methods, scored, thresholds)

    tables = config.EVAL_DIR / args.split / "tables"
    figures = config.EVAL_DIR / args.split / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    segmentation = segmentation_table(images)
    curves, pr_summary = pr_tables(histograms, segmentation, args.split)
    widths = width_table(cracks, images)
    ranking = ranking_table(segmentation, widths)
    matrices = pd.concat([severity_matrix(cracks, m).rename_axis("ref_grade").reset_index().assign(method=m)
                          for m in methods])
    outputs = {
        "segmentation.csv": segmentation,
        "pr_curves.csv": curves,
        "pr_summary.csv": pr_summary,
        "width_errors.csv": widths,
        "error_by_grade.csv": error_by_grade(cracks),
        "severity_matrix.csv": matrices,
        "ranking.csv": ranking,
        "success_criteria.csv": success_criteria(segmentation, widths, cracks, ranking),
        "timing.csv": timing_table(args.split, methods),
        "per_image.csv": images,
        "per_crack.csv": cracks,
    }
    for name, table in outputs.items():
        table.to_csv(tables / name, index=False, float_format="%.4f")

    plot_overlays(args.split, pick_examples(cracks, ids, args.examples), methods, figures / "overlays.png")
    plot_severity(cracks, methods, figures / "severity_matrices.png")
    plot_score_vs_error(segmentation, widths, figures / "score_vs_width_error.png")
    plot_pr_curves(curves, pr_summary, segmentation, figures / "pr_curves.png")
    utils.record_environment()

    ignored = int(images.drop_duplicates("image_id")["ref_small_ignored"].sum())
    with pd.option_context("display.width", 140, "display.max_columns", 20, "display.precision", 3):
        print("\nSegmentation (dataset level)\n", segmentation.drop(columns=["images"]).to_string(index=False))
        if not pr_summary.empty:
            print("\nPrecision-recall\n", pr_summary.to_string(index=False))
        print("\nWidth and length errors (matched cracks, pixels)\n", widths.to_string(index=False))
        print("\nSuccess criteria (Section 6.8)\n", outputs["success_criteria.csv"].to_string(index=False))
    print(f"\nReference components under {evaluate.MIN_CRACK_AREA_PX} px ignored as noise: {ignored}")
    if args.split == "test" and not pr_summary.empty:
        print("Note: best_threshold on the test split is for reporting only; thresholds come from validation.")
    print(f"Tables: {tables.relative_to(ROOT)}  Figures: {figures.relative_to(ROOT)}")


if __name__ == "__main__":
    main()