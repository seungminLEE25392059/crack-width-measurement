"""Step-by-step processing figure for Method A (report Section 2.3).

Usage  : python scripts/figure_a.py --id 040        (a VALIDATION image)
Shows original -> median + enhancement -> crack response -> binarised ->
after clean-up, and the final mask against the reference (green = correct,
red = false positive, blue = missed), using the selected params/A.json.

Inputs : one validation image ID, params/A.json
Outputs: eval/val/method_a/steps_<id>.png (300 dpi)
Author : Wenjie Hu
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src import config, data, method_a, utils  # noqa: E402


def error_overlay(image: np.ndarray, pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Darkened photograph with true positives, false positives and misses coloured."""
    out = (image.astype(np.float32) * 0.5).astype(np.uint8)
    out[pred & ref] = (40, 200, 60)
    out[pred & ~ref] = (230, 40, 40)
    out[~pred & ref] = (40, 110, 255)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--id", default="040", help="validation image ID")
    args = parser.parse_args()
    if args.id not in data.load_split()["val"]:
        sys.exit(f"{args.id} is not a validation image; figures must not use the test split.")

    p = utils.load_params("A", method_a.DEFAULT_PARAMS)
    image, ref = data.load_pair(args.id)
    mask, steps = method_a.segment_a(image, p, return_steps=True)

    enhancement = {"none": "no enhancement", "histeq": "histogram equalisation", "clahe": "CLAHE"}[p["enhance"]]
    median = f"median {p['median_ksize']}×{p['median_ksize']}" if p["median_ksize"] else "no median"
    response = "black top-hat" if p["background"] == "blackhat" else "inverted intensity"
    panels = [
        (image, "(a) Original"),
        (steps["enhanced"], f"(b) {median}, {enhancement}"),
        (steps["response"], f"(c) Crack response: {response}"),
        (steps["binary"], f"(d) Binarised: {p['threshold']}"),
        (mask, f"(e) Opening r={p['open_radius']}, closing r={p['close_radius']}, component filter"),
        (error_overlay(image, mask, ref), "(f) Green correct, red false positive, blue missed"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5) if image.shape[1] > image.shape[0] else (10, 9.5))
    for ax, (picture, title) in zip(axes.ravel(), panels):
        ax.imshow(picture, cmap="gray" if picture.ndim == 2 else None, interpolation="antialiased")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    out = config.EVAL_DIR / "val" / "method_a" / f"steps_{args.id}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
