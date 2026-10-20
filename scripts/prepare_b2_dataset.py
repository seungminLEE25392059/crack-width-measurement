"""Build B2: 32x32 patches with a 2x2 ground-truth crack centre."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data  # noqa: E402


DEFAULT_NEGATIVES = ROOT.parents[1] / "Concrete Crack Images for Classification 32x32 Centre Crop"
DEFAULT_OUTPUT = ROOT.parents[1] / "Concrete Crack B2 32x32 Center 2x2"
TARGETS = {"train": 14000, "val": 3000, "test": 3000}
PATCH_SIZE = 32
CORE_SIZE = 2
CORE_OFFSET = (PATCH_SIZE - CORE_SIZE) // 2


def valid_core_positions(mask: np.ndarray) -> np.ndarray:
    """Return top-left positions of 2x2 crack cores with room for a 32x32 crop."""
    valid = mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:] & mask[1:, 1:]
    rows, cols = np.nonzero(valid)
    keep = (
        (rows >= CORE_OFFSET)
        & (cols >= CORE_OFFSET)
        & (rows + PATCH_SIZE - CORE_OFFSET <= mask.shape[0])
        & (cols + PATCH_SIZE - CORE_OFFSET <= mask.shape[1])
    )
    return np.column_stack((rows[keep], cols[keep]))


def quotas(total: int, image_ids: list[str]) -> dict[str, int]:
    base, remainder = divmod(total, len(image_ids))
    return {image_id: base + (index < remainder) for index, image_id in enumerate(image_ids)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--negatives", type=Path, default=DEFAULT_NEGATIVES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output is not empty: {args.output}")

    split = data.load_split()
    rng = np.random.default_rng(args.seed)
    manifest_rows = []
    patch_number = 1

    for split_name, target in TARGETS.items():
        positive_dir = args.output / split_name / "Positive"
        negative_dir = args.output / split_name / "Negative"
        positive_dir.mkdir(parents=True, exist_ok=True)
        negative_dir.mkdir(parents=True, exist_ok=True)

        for image_id, count in quotas(target, split[split_name]).items():
            image, mask = data.load_pair(image_id)
            candidates = valid_core_positions(mask)
            if len(candidates) < count:
                raise RuntimeError(f"{image_id}: only {len(candidates)} valid centres for {count} patches")

            for core_row, core_col in candidates[rng.choice(len(candidates), count, replace=False)]:
                crop_row = int(core_row - CORE_OFFSET)
                crop_col = int(core_col - CORE_OFFSET)
                patch = image[crop_row:crop_row + PATCH_SIZE, crop_col:crop_col + PATCH_SIZE]
                patch_id = f"{patch_number:05d}"
                output_path = positive_dir / f"{patch_id}.png"
                if not cv2.imwrite(str(output_path), cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)):
                    raise OSError(f"Could not write {output_path}")
                manifest_rows.append({
                    "patch_id": patch_id,
                    "split": split_name,
                    "source_image_id": image_id,
                    "crop_row": crop_row,
                    "crop_col": crop_col,
                    "core_row": int(core_row),
                    "core_col": int(core_col),
                    "core_size": CORE_SIZE,
                    "seed": args.seed,
                })
                patch_number += 1

        negative_paths = sorted((args.negatives / split_name / "Negative").glob("*.png"))
        if len(negative_paths) != target:
            raise RuntimeError(f"Expected {target} B1 negatives for {split_name}, found {len(negative_paths)}")
        for source_path in negative_paths:
            shutil.copy2(source_path, negative_dir / source_path.name)
        print(f"{split_name}: {target:,} positive + {target:,} negative")

    with (args.output / "positive_manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "seed": args.seed,
        "patch_size": PATCH_SIZE,
        "positive_centre": "2x2 pixels, all crack in the ground-truth mask",
        "negative_source": str(args.negatives),
        "counts": {name: {"Positive": count, "Negative": count} for name, count in TARGETS.items()},
    }
    (args.output / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    assert len(manifest_rows) == 20000
    assert all(len(list((args.output / name / label).glob("*.png"))) == count
               for name, count in TARGETS.items() for label in ("Positive", "Negative"))
    print(f"Created 40,000 patches in {args.output}")


if __name__ == "__main__":
    main()
