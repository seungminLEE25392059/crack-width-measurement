"""Select Method A parameters on the VALIDATION split, then run the ablation.

Usage (from the repository root):
  python scripts/tune_a.py                         # full search + ablation, all val images
  python scripts/tune_a.py --limit 10 --workers 4  # quick trial on 10 images
  python scripts/tune_a.py --ablation-only         # ablation for the saved params/A.json

Search (staged, because a full grid is too slow on 12-megapixel photographs):
  stage 1 structure     median size x enhancement x background x threshold method
  stage 2 enhancement   CLAHE clip and grid, black top-hat radius (if selected)
  stage 3 threshold     adaptive block and offset, or Canny scale and fill (if selected)
  stage 4 cleanup       opening, closing, minimum area, elongation rule and threshold
Each stage starts from the best setting of the previous one. The objective is
the mean per-image F1 (= Dice) of the crack class. The test split is never read.

Inputs : splits/splits.json, the dataset, src/method_a.py
Outputs: params/A.json                      selected parameters (utils.save_params)
         eval/val/method_a/tune_stage<k>.csv one row per configuration, best first
         eval/val/method_a/tuning_summary.json
         eval/val/method_a/ablation_A.csv   one step switched off at a time
Author : Wenjie Hu
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src import config, data, method_a, utils  # noqa: E402

OUT_DIR = config.EVAL_DIR / "val" / "method_a"
THRESHOLD_KEYS = {"otsu": (), "adaptive": ("adaptive_block", "adaptive_c"),
                  "canny": ("canny_scale", "canny_fill_radius")}


# --- Evaluating many configurations on one image ------------------------------------------

def level_keys(p: dict) -> tuple:
    """Cache keys for each pipeline level; configurations that share a key share the work."""
    pre = (p["median_ksize"], p["enhance"],
           p["clahe_clip"] if p["enhance"] == "clahe" else None,
           p["clahe_grid"] if p["enhance"] == "clahe" else None)
    resp = pre + (p["background"], p["tophat_radius"] if p["background"] == "blackhat" else None)
    bw = resp + (p["threshold"],) + tuple(p[k] for k in THRESHOLD_KEYS[p["threshold"]])
    return pre, resp, bw, bw + (p["open_radius"], p["close_radius"])


def evaluate_image(task) -> list[dict]:
    """Score every configuration on one validation image.

    Configurations are sorted by their cache keys and each level keeps only its
    latest result, so memory stays at a few full-size arrays per worker.
    Component statistics are computed once per morphology result and reused for
    every (min_area, min_elongation, rule) combination.
    """
    image_id, configs = task
    image, ref = data.load_pair(image_id)
    grey = method_a.to_grey(image)
    del image
    n_ref = int(ref.sum())
    cache_key, cache = {}, {}

    def cached(level, key, compute):
        if cache_key.get(level) != key:
            cache[level] = compute()
            cache_key[level] = key
        return cache[level]

    full = sorted(((i, method_a.full_params(c)) for i, c in enumerate(configs)),
                  key=lambda item: repr(level_keys(item[1])))
    rows = []
    for index, p in full:
        k_pre, k_resp, k_bw, k_morph = level_keys(p)
        enhanced = cached("pre", k_pre, lambda: method_a.enhance(grey, p))
        response = cached("resp", k_resp, lambda: method_a.crack_response(enhanced, p))
        binary = cached("bw", k_bw, lambda: method_a.binarise(response, p))
        cleaned = cached("morph", k_morph, lambda: method_a.morphology(binary, p))
        labels, area, major, minor, boundary = cached(
            "stats", k_morph, lambda: method_a.component_stats(cleaned))
        keep = (area >= p["min_area"]) & (
            method_a.elongation(area, major, minor, boundary, p["elongation_rule"]) >= p["min_elongation"])
        keep[0] = False
        # Strict pixel counts from per-label sums: no full-size mask is built.
        kept_area = float(area[keep].sum())
        tp = float(np.bincount(labels[ref], minlength=len(keep))[keep].sum())
        f1 = 2 * tp / (kept_area + n_ref) if kept_area + n_ref else 1.0
        rows.append({"config": index, "image_id": image_id, "tp": tp,
                     "n_pred": kept_area, "n_ref": n_ref, "f1": f1})
    return rows


def evaluate(ids: list[str], configs: list[dict], workers: int) -> list[dict]:
    """Mean per-image F1, micro F1, precision and recall for each configuration."""
    tasks = [(i, configs) for i in ids]
    rows: list[dict] = []
    started = time.time()
    with Pool(workers) as pool:
        for n, image_rows in enumerate(pool.imap_unordered(evaluate_image, tasks), 1):
            rows.extend(image_rows)
            print(f"    {n}/{len(ids)} images ({time.time() - started:.0f} s)", end="\r", flush=True)
    print()
    summary = []
    for index, partial in enumerate(configs):
        own = [r for r in rows if r["config"] == index]
        tp = sum(r["tp"] for r in own)
        n_pred = sum(r["n_pred"] for r in own)
        n_ref = sum(r["n_ref"] for r in own)
        precisions = [r["tp"] / r["n_pred"] if r["n_pred"] else float(r["n_ref"] == 0) for r in own]
        recalls = [r["tp"] / r["n_ref"] if r["n_ref"] else 1.0 for r in own]
        summary.append({**partial,
                        "mean_f1": float(np.mean([r["f1"] for r in own])),
                        "micro_f1": 2 * tp / (n_pred + n_ref) if n_pred + n_ref else 1.0,
                        "mean_precision": float(np.mean(precisions)),
                        "mean_recall": float(np.mean(recalls)),
                        "empty_masks": sum(r["n_pred"] == 0 for r in own)})
    return summary


# --- Search stages ---------------------------------------------------------------------------

def grid(**axes) -> list[dict]:
    return [dict(zip(axes, values)) for values in itertools.product(*axes.values())]


def stage_grid(stage: int, best: dict) -> list[dict]:
    """Settings tried in each stage, given the current best parameters."""
    if stage == 1:
        # Shape test off here (min_elongation 0) so that it cannot bias the
        # choice of pre-processing; it is tuned in stage 4.
        return grid(median_ksize=[0, 3, 5, 7], enhance=["none", "histeq", "clahe"],
                    background=["none", "blackhat"], threshold=["otsu", "adaptive", "canny"],
                    min_elongation=[0.0])
    if stage == 2:
        axes = {}
        if best["enhance"] == "clahe":
            axes.update(clahe_clip=[1.0, 2.0, 3.0, 4.0], clahe_grid=[8, 16])
        if best["background"] == "blackhat":
            axes.update(tophat_radius=[7, 11, 15, 21, 31])
        return grid(**axes) if axes else []
    if stage == 3:
        if best["threshold"] == "adaptive":
            return grid(adaptive_block=[25, 51, 101, 201], adaptive_c=[5, 10, 15, 20])
        if best["threshold"] == "canny":
            return grid(canny_scale=[0.6, 0.8, 1.0, 1.3], canny_fill_radius=[2, 3, 5])
        return []
    cleanup = dict(open_radius=[0, 1, 2], close_radius=[0, 1, 2, 3],
                   min_area=[0, 50, 200, 500, 1000, 2000])
    return (grid(**cleanup, elongation_rule=["ellipse"], min_elongation=[0.0, 2.0, 3.0, 4.0, 6.0])
            + grid(**cleanup, elongation_rule=["thinness"], min_elongation=[5.0, 8.0, 12.0, 20.0]))


STAGE_NAMES = {1: "structure", 2: "enhancement", 3: "threshold", 4: "cleanup"}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def tune(ids: list[str], workers: int) -> dict:
    best = dict(method_a.DEFAULT_PARAMS)
    history = []
    for stage in (1, 2, 3, 4):
        changes = stage_grid(stage, best)
        if not changes:
            print(f"Stage {stage} ({STAGE_NAMES[stage]}): nothing to tune for the selected method")
            continue
        configs = [{**best, **change} for change in changes]
        print(f"Stage {stage} ({STAGE_NAMES[stage]}): {len(configs)} settings x {len(ids)} images")
        started = time.time()
        table = sorted(evaluate(ids, configs, workers), key=lambda r: r["mean_f1"], reverse=True)
        write_csv(OUT_DIR / f"tune_stage{stage}_{STAGE_NAMES[stage]}.csv", table)
        winner = {k: table[0][k] for k in changes[0]}
        best.update(winner)
        history.append({"stage": stage, "name": STAGE_NAMES[stage], "settings": len(configs),
                        "seconds": round(time.time() - started), "chosen": winner,
                        "mean_f1": table[0]["mean_f1"], "micro_f1": table[0]["micro_f1"]})
        print(f"  best mean F1 {table[0]['mean_f1']:.4f} (micro {table[0]['micro_f1']:.4f}) with {winner}")
    path = utils.save_params("A", best)
    summary = {"objective": "mean per-image F1, validation split", "n_images": len(ids),
               "history": history, "params": best}
    (OUT_DIR / "tuning_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved {path.relative_to(ROOT)}")
    return best


# --- Ablation ---------------------------------------------------------------------------------

def ablation(ids: list[str], best: dict, workers: int) -> None:
    """Switch one step off (or swap one choice) at a time, Task 1 report Section 6.5."""
    variants = [("Full Method A (selected)", {}),
                ("No median filter", {"median_ksize": 0}),
                ("No contrast enhancement", {"enhance": "none"}),
                ("No median filter and no enhancement", {"median_ksize": 0, "enhance": "none"}),
                ("No opening or closing", {"open_radius": 0, "close_radius": 0}),
                ("No component filter", {"min_area": 0, "min_elongation": 0.0}),
                ("No clean-up at all", {"open_radius": 0, "close_radius": 0,
                                        "min_area": 0, "min_elongation": 0.0})]
    if best["background"] == "blackhat":
        variants.append(("No black top-hat (inverted image)", {"background": "none"}))
    else:
        variants.append(("With black top-hat", {"background": "blackhat"}))
    for option in ("none", "histeq", "clahe"):
        if option != best["enhance"]:
            variants.append((f"Enhancement: {option}", {"enhance": option}))
    for option in ("otsu", "adaptive", "canny"):
        if option != best["threshold"]:
            variants.append((f"Threshold: {option}", {"threshold": option}))
    if best["elongation_rule"] == "thinness":
        variants.append(("Ellipse rule instead of thinness (ratio 3)",
                         {"elongation_rule": "ellipse", "min_elongation": 3.0}))
    else:
        variants.append(("Thinness rule instead of ellipse (score 8)",
                         {"elongation_rule": "thinness", "min_elongation": 8.0}))
    print(f"Ablation: {len(variants)} variants x {len(ids)} images")
    rows = evaluate(ids, [{**best, **change} for _, change in variants], workers)
    base = rows[0]["mean_f1"]
    table = [{"variant": name, "mean_precision": round(r["mean_precision"], 4),
              "mean_recall": round(r["mean_recall"], 4), "mean_f1": round(r["mean_f1"], 4),
              "micro_f1": round(r["micro_f1"], 4), "delta_f1": round(r["mean_f1"] - base, 4),
              "empty_masks": r["empty_masks"]}
             for (name, _), r in zip(variants, rows)]
    write_csv(OUT_DIR / "ablation_A.csv", table)
    for row in table:
        print(f"  {row['variant']:<45} F1 {row['mean_f1']:.3f}  ΔF1 {row['delta_f1']:+.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, help="use only the first N validation images")
    parser.add_argument("--workers", type=int, default=2, help="parallel processes (each holds a few 12-MP arrays)")
    parser.add_argument("--ablation-only", action="store_true", help="skip the search; use params/A.json")
    args = parser.parse_args()

    utils.set_seed()
    ids = data.load_split()["val"][: args.limit]          # the test split is never loaded
    if args.ablation_only:
        best = utils.load_params("A", method_a.DEFAULT_PARAMS)
    else:
        best = tune(ids, args.workers)
    ablation(ids, best, args.workers)
    if args.limit:
        print("\nNote: --limit was used; rerun on all validation images before the final settings are frozen.")


if __name__ == "__main__":
    main()
