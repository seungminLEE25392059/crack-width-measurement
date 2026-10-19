"""Run one segmentation method on one split and save its outputs.

Usage:
  python scripts/run_method.py --method A --split val
  python scripts/run_method.py --method C --split val --limit 5      # quick test
  python scripts/run_method.py --method D --split val                # needs Method C masks
  python scripts/run_method.py --method A --split test --final       # once, Sun 18 Oct

Inputs : splits/splits.json, the dataset, params/<method>.json,
         models/method_b.joblib (B), models/unet.pt (C)
Outputs: results/<method>/<split>/<id>.png        binary mask (0/255)
         results/<method>/<split>/<id>_prob.png   confidence (B) or probability (C)
         results/<method>/<split>/timing.csv      seconds per image
         results/<method>/<split>/params_used.json
Author : Seungmin Lee
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src import config, data, method_a, method_b, method_c, method_d, utils  # noqa: E402

MODULES = {"A": method_a, "B": method_b, "C": method_c, "D": method_d}
MAIN_FUNCTIONS = {"A": method_a.segment_a, "B": method_b.segment_b,
                  "C": method_c.segment_c, "D": method_d.refine_d}


def check_mask(mask, shape: tuple[int, int], method: str) -> None:
    """Stop with a clear message when a method breaks the agreed mask format."""
    if not isinstance(mask, np.ndarray) or mask.dtype != bool or mask.shape != shape:
        found = f"{getattr(mask, 'dtype', type(mask).__name__)} {getattr(mask, 'shape', '')}"
        raise TypeError(f"Method {method} must return a bool mask of shape {shape}; got {found}")


def load_model(method: str):
    """Trained model for B or C, or None (A and D need none)."""
    if method == "B":
        return method_b.load_model_b(config.MODELS_DIR / "method_b.joblib")
    if method == "C":
        return method_c.load_unet(config.MODELS_DIR / "unet.pt")
    return None


def run_one(method: str, split: str, image_id: str, model, params: dict):
    """Apply one method to one image; returns (mask, probability map or None, seconds)."""
    if method == "D":
        mask_c_path = utils.result_path("C", split, image_id)
        if not mask_c_path.exists():
            raise FileNotFoundError(f"Run Method C on the {split} split first ({mask_c_path} is missing)")
        mask_c = utils.read_mask(mask_c_path)
        with utils.timer() as clock:
            mask = method_d.refine_d(mask_c, params)
        check_mask(mask, mask_c.shape, method)
        return mask, None, clock["seconds"]

    image, _ = data.load_pair(image_id)
    with utils.timer() as clock:
        if method == "A":
            mask, prob = method_a.segment_a(image, params), None
        elif method == "B":
            mask, prob = method_b.segment_b(image, model, params)
        else:
            mask, prob = method_c.segment_c(image, model, params)
    check_mask(mask, image.shape[:2], method)
    return mask, prob, clock["seconds"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", choices=config.METHODS, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--limit", type=int, default=None, help="process only the first N images")
    parser.add_argument("--final", action="store_true", help="required for the single final test run")
    args = parser.parse_args()

    if args.split == "test" and not args.final:
        parser.error("the test split is run once, on Sun 18 Oct: add --final to confirm")

    utils.set_seed()
    method = args.method
    params = utils.load_params(method, MODULES[method].DEFAULT_PARAMS)
    model = load_model(method)
    if utils.is_placeholder(MAIN_FUNCTIONS[method]):
        print(f"Note: Method {method} is still a placeholder ({MODULES[method].__name__}).")
    elif method in ("B", "C") and model is None:
        sys.exit(f"Method {method} has no trained model in {config.MODELS_DIR}: train it first.")

    ids = data.load_split()[args.split][: args.limit]
    timings = []
    for n, image_id in enumerate(ids, 1):
        mask, prob, seconds = run_one(method, args.split, image_id, model, params)
        utils.save_mask(mask, utils.result_path(method, args.split, image_id))
        if prob is not None:
            utils.save_prob(prob, utils.result_path(method, args.split, image_id, "prob"))
        timings.append((image_id, seconds))
        print(f"[{n}/{len(ids)}] {image_id}: crack fraction {mask.mean():.4f}, {seconds:.2f} s")

    out_dir = config.RESULTS_DIR / method / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "timing.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image_id", "seconds"])
        writer.writerows(timings)
    (out_dir / "params_used.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    print(f"Saved {len(ids)} masks to {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
