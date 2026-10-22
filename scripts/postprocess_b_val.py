"""Evaluate small-component removal on saved Method B validation masks.

Run: python scripts/postprocess_b_val.py --min-area 20
By default this reads saved binary masks. --threshold rebuilds them from saved
8-bit probability PNGs, whose scores have quantisation error up to 0.5/255.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from src import config, data, evaluate, utils


def remove_small_components(mask: np.ndarray, min_area: int):
    """Keep 8-connected components with at least min_area pixels."""
    if mask.ndim != 2 or mask.dtype != bool or min_area < 1:
        raise ValueError("Expected a 2-D bool mask and min_area >= 1")
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.r_[False, areas >= min_area]
    return keep[labels], int(np.count_nonzero(areas < min_area)), (
        int(areas.min()) if count > 1 else None
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-area", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()
    if args.min_area < 1:
        parser.error("--min-area must be at least 1")
    if args.threshold is not None and not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")

    ids = data.load_split()["val"]
    kind = "prob" if args.threshold is not None else "mask"
    inputs = [utils.result_path("B", "val", image_id, kind) for image_id in ids]
    references = {p.stem: p for p in config.MASK_DIR.iterdir()
                  if p.suffix.lower() in config.IMAGE_EXTS}
    for image_id, path in zip(ids, inputs):
        if not path.exists() or image_id not in references:
            raise FileNotFoundError(f"Missing prediction or reference for {image_id}")
    prefix = "val" if args.threshold is None else f"val_threshold{args.threshold:g}"
    output = config.RESULTS_DIR / "B" / f"{prefix}_clean_area{args.min_area}"
    if output.exists():
        raise FileExistsError(f"Preserving existing experiment: {output}")
    output.mkdir(parents=True)

    totals = {stage: dict.fromkeys(evaluate.COUNT_KEYS, 0)
              for stage in ("raw", "cleaned")}
    rows = []
    removed_pixels = removed_components = changed_images = 0
    smallest_component = None
    for index, (image_id, path) in enumerate(zip(ids, inputs), 1):
        raw = (utils.read_mask(path) if args.threshold is None else
               utils.read_prob(path) >= args.threshold)
        ref = utils.read_mask(references[image_id])
        cleaned, removed, smallest = remove_small_components(raw, args.min_area)
        pixels = int(np.count_nonzero(raw)) - int(np.count_nonzero(cleaned))
        removed_pixels += pixels
        removed_components += removed
        changed_images += int(pixels > 0)
        if smallest is not None:
            smallest_component = smallest if smallest_component is None else min(smallest_component, smallest)
        raw_counts = evaluate.confusion_counts(raw, ref)
        clean_counts = evaluate.confusion_counts(cleaned, ref) if pixels else raw_counts
        for stage, counts in (("raw", raw_counts), ("cleaned", clean_counts)):
            for key in evaluate.COUNT_KEYS:
                totals[stage][key] += counts[key]
            rows.append({"image_id": image_id, "stage": stage,
                         "removed_components": removed if stage == "cleaned" else 0,
                         "removed_pixels": pixels if stage == "cleaned" else 0,
                         **evaluate.metrics_from_counts(counts)})
        utils.save_mask(cleaned, output / f"{image_id}.png")
        if index % 10 == 0 or index == len(ids):
            print(f"Validation {index}/{len(ids)}; removed pixels so far: {removed_pixels}", flush=True)

    summary = {
        "split": "val", "images": len(ids), "min_area": args.min_area,
        "connectivity": 8, "tolerance_px": config.TOLERANCE_PX,
        "input_masks": str(config.RESULTS_DIR / "B" / "val"),
        "threshold": ("unchanged from saved input masks" if args.threshold is None
                      else args.threshold),
        "input_kind": kind,
        "probability_quantisation": (None if args.threshold is None else
                                     "8-bit PNG; rounding error up to 0.5/255"),
        "changed_images": changed_images, "removed_components": removed_components,
        "removed_pixels": removed_pixels, "smallest_input_component_area": smallest_component,
        "aggregation": "micro: sum pixel counts across all validation images",
        "metrics": {stage: evaluate.metrics_from_counts(counts)
                    for stage, counts in totals.items()},
    }
    with (output / "per_image.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
