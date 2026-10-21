"""Diagnose why maximum crack widths are underestimated (matched cracks only).

Usage  : python scripts/width_diagnostics.py --split val
         python scripts/width_diagnostics.py --split val --methods C,D --limit 10   # quick check
Inputs : results/<method>/<split>/<id>.png and the reference masks
Outputs: eval/<split>/tables/width_diagnostics.csv          one row per matched reference crack
         eval/<split>/tables/width_diagnostics_summary.csv  one row per method
run_evaluation.py reports the max-width error of the matched predicted crack
(the one overlapping the reference crack most). This script recomputes that
bias in four other ways; the one that removes the bias points to its cause:
  max_all    largest max width over ALL predicted cracks overlapping the reference
             crack. Bias gone -> fragmentation: the widest part is in another piece.
  at_widest  largest predicted skeleton width inside the reference's widest
             inscribed circle (0 when the prediction is absent there). Bias still
             there -> the prediction is thin exactly where the reference is widest.
  p95        95th percentile of the skeleton widths instead of the maximum.
             Bias much smaller -> a few extreme points drive it.
  relative   errors divided by the reference width. Similar for mean and max
             width -> the prediction is proportionally thinner everywhere.
It also flags reference cracks whose widest point is a junction (three or more
skeleton branches leave it). Diagnostic only: it changes no parameter and no
reported metric.
Author : Seungmin Lee
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

from src import config, data, evaluate, utils  # noqa: E402


def widest_point(crack: dict) -> np.ndarray:
    """(row, col) of the skeleton pixel where the crack is widest."""
    return crack["skeleton_coords_rc"][int(np.argmax(crack["local_widths_px"]))]


def p95(crack: dict) -> float:
    """95th percentile of the widths along the skeleton (robust alternative to the maximum)."""
    return float(np.percentile(crack["local_widths_px"], 95))


def branches_leaving(crack: dict, point: np.ndarray, radius: float) -> int:
    """Number of skeleton branches leaving the circle of `radius` px around `point`.

    A branch counts only when it runs from that circle out to twice the radius,
    so the short spurs that rough mask edges add to a skeleton are ignored.
    1 = end of a crack, 2 = along a crack, 3 or more = junction.
    """
    coords = crack["skeleton_coords_rc"]
    distance = np.hypot(*(coords - point).T)
    outer = 2 * radius + 2
    keep = (distance >= radius) & (distance < outer)                  # skeleton inside the ring
    if not keep.any():
        return 0
    ring, ring_distance = coords[keep], distance[keep]
    origin = ring.min(axis=0)
    picture = np.zeros(ring.max(axis=0) - origin + 1, bool)
    picture[tuple((ring - origin).T)] = True
    pieces = ndi.label(picture, structure=np.ones((3, 3), bool))[0][tuple((ring - origin).T)]
    inner, far = set(pieces[ring_distance < radius + 2]), set(pieces[ring_distance >= outer - 2])
    return len(inner & far)                                            # pieces crossing the whole ring


def overlapping(ref_labels: np.ndarray, pred_labels: np.ndarray) -> dict[int, list[int]]:
    """Reference id -> ids of every predicted crack that overlaps it."""
    both = (ref_labels > 0) & (pred_labels > 0)
    if not both.any():
        return {}
    pairs = np.unique(np.stack([ref_labels[both], pred_labels[both]]).astype(np.int64), axis=1)
    found: dict[int, list[int]] = {}
    for ref_id, pred_id in pairs.T:
        found.setdefault(int(ref_id), []).append(int(pred_id))
    return found


def width_near(point: np.ndarray, radius: float, pred_cracks: list[dict]) -> float:
    """Largest predicted skeleton width closer than `radius` px to `point` (0 when there is none)."""
    best = 0.0
    for crack in pred_cracks:
        near = np.hypot(*(crack["skeleton_coords_rc"] - point).T) < radius
        if near.any():
            best = max(best, float(crack["local_widths_px"][near].max()))
    return best


def diagnose_image(ref_cracks: list[dict], ref_labels: np.ndarray,
                   pred_cracks: list[dict], pred_labels: np.ndarray) -> list[dict]:
    """One row per matched reference crack: its widths measured in every way listed above."""
    matches = evaluate.match_cracks(ref_labels, pred_labels)
    touching = overlapping(ref_labels, pred_labels)
    pred_by_id = {crack["id"]: crack for crack in pred_cracks}
    rows = []
    for ref in ref_cracks:
        pred_id = matches.get(ref["id"])
        if pred_id is None or not np.isfinite(ref["max_width_px"]):
            continue                                     # missed cracks have no width error
        pred, pieces = pred_by_id[pred_id], [pred_by_id[i] for i in touching[ref["id"]]]
        point, radius = widest_point(ref), ref["max_width_px"] / 2   # widest inscribed circle
        rows.append({
            "ref_id": ref["id"], "pred_id": pred_id, "pieces": len(pieces),
            "ref_mean": ref["mean_width_px"], "ref_max": ref["max_width_px"], "ref_p95": p95(ref),
            "pred_mean": pred["mean_width_px"], "pred_max": pred["max_width_px"], "pred_p95": p95(pred),
            "pred_max_all": max(piece["max_width_px"] for piece in pieces),
            "pred_at_widest": width_near(point, radius, pieces),
            "widest_at_junction": branches_leaving(ref, point, radius) >= 3,
            "widest_row": int(point[0]), "widest_col": int(point[1]),
        })
    return rows


def summarise(rows: pd.DataFrame) -> pd.DataFrame:
    """Mean signed errors (prediction - reference) per method under each explanation."""
    table = []
    for method, g in rows.groupby("method", sort=False):
        max_error, junction = g["pred_max"] - g["ref_max"], g["widest_at_junction"].astype(bool)
        table.append({
            "method": method, "matched_cracks": len(g),
            "max_bias": max_error.mean(),
            "max_all_bias": (g["pred_max_all"] - g["ref_max"]).mean(),
            "at_widest_bias": (g["pred_at_widest"] - g["ref_max"]).mean(),
            "absent_at_widest": (g["pred_at_widest"] == 0).mean(),
            "p95_bias": (g["pred_p95"] - g["ref_p95"]).mean(),
            "mean_bias": (g["pred_mean"] - g["ref_mean"]).mean(),
            "max_rel_bias": (max_error / g["ref_max"]).mean(),
            "mean_rel_bias": ((g["pred_mean"] - g["ref_mean"]) / g["ref_mean"]).mean(),
            "fragmented": (g["pieces"] > 1).mean(),
            "widest_at_junction": junction.mean(),
            "max_bias_junction": max_error[junction].mean(),
            "max_bias_other": max_error[~junction].mean(),
        })
    return pd.DataFrame(table)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--methods", default=",".join(config.METHODS), help="comma-separated, e.g. C,D")
    parser.add_argument("--limit", type=int, default=None, help="only the first N images (quick check)")
    args = parser.parse_args()

    ids = data.load_split()[args.split][: args.limit]
    methods = []
    for method in args.methods.upper().split(","):
        missing = sum(not utils.result_path(method, args.split, i).exists() for i in ids)
        if missing:
            print(f"Skipping Method {method}: {missing} of {len(ids)} masks missing in results/{method}/{args.split}")
        else:
            methods.append(method)
    if not methods:
        sys.exit("No method has all its masks: run scripts/run_method.py first.")

    rows = []
    for n, image_id in enumerate(ids, 1):
        _, ref = data.load_pair(image_id)
        ref_cracks, ref_labels, _ = evaluate.measure_mask(ref)      # same crack definition as run_evaluation
        for method in methods:
            pred = utils.read_mask(utils.result_path(method, args.split, image_id))
            pred_cracks, pred_labels, _ = evaluate.measure_mask(pred)
            rows += [{"method": method, "image_id": image_id, **row}
                     for row in diagnose_image(ref_cracks, ref_labels, pred_cracks, pred_labels)]
        print(f"[{n}/{len(ids)}] {image_id}")
    if not rows:
        sys.exit("No matched cracks: nothing to diagnose.")

    rows = pd.DataFrame(rows)
    summary = summarise(rows)
    tables = config.EVAL_DIR / args.split / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    rows.to_csv(tables / "width_diagnostics.csv", index=False)
    summary.to_csv(tables / "width_diagnostics_summary.csv", index=False)
    print("\nMax-width diagnostics (prediction - reference, px; matched cracks only)\n",
          summary.set_index("method").T.round(2).to_string())
    print("\nHow to read it:\n"
          "  max_all_bias near 0 while max_bias is negative -> fragmentation\n"
          "  at_widest_bias still negative                  -> prediction thin where the reference is widest\n"
          "  p95_bias much closer to 0 than max_bias        -> a few extreme points drive the bias\n"
          "  max_rel_bias close to mean_rel_bias            -> prediction proportionally thinner everywhere\n"
          "  max_bias_junction far below max_bias_other     -> junctions drive the bias")
    print(f"Saved width_diagnostics.csv and width_diagnostics_summary.csv in {tables.relative_to(ROOT)}")


if __name__ == "__main__":
    main()