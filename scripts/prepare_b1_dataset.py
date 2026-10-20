"""Create the temporary B1 dataset from the 227x227 classification patches.

The output is a deterministic, class-balanced 70/15/15 patch-level split.
Each source image is centre-cropped to 32x32 and saved losslessly as PNG.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT.parents[1] / "Concrete Crack Images for Classification"
DEFAULT_OUTPUT = ROOT.parents[1] / "Concrete Crack Images for Classification 32x32 Centre Crop"
CLASSES = ("Negative", "Positive")
SPLITS = (("train", 0.70), ("val", 0.15), ("test", 0.15))


def centre_crop(image, size: int):
    height, width = image.shape[:2]
    if height < size or width < size:
        raise ValueError(f"Image {width}x{height} is smaller than {size}x{size}")
    row = (height - size) // 2
    col = (width - size) // 2
    return image[row:row + size, col:col + size]


def split_files(files: list[Path], seed: int) -> dict[str, list[Path]]:
    files = sorted(files)
    random.Random(seed).shuffle(files)
    train_end = round(SPLITS[0][1] * len(files))
    val_end = train_end + round(SPLITS[1][1] * len(files))
    return {
        "train": files[:train_end],
        "val": files[train_end:val_end],
        "test": files[val_end:],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output is not empty: {args.output}")

    manifest = {"patch_size": args.size, "seed": args.seed, "splits": {}}
    written = 0
    for class_name in CLASSES:
        source_dir = args.source / class_name
        files = [path for path in source_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if not files:
            raise FileNotFoundError(f"No images found in {source_dir}")

        for split_name, split_paths in split_files(files, args.seed).items():
            destination = args.output / split_name / class_name
            destination.mkdir(parents=True, exist_ok=True)
            manifest["splits"].setdefault(split_name, {})[class_name] = len(split_paths)

            for source_path in split_paths:
                image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"Could not read {source_path}")
                crop = centre_crop(image, args.size)
                output_path = destination / f"{source_path.stem}.png"
                if not cv2.imwrite(str(output_path), crop):
                    raise OSError(f"Could not write {output_path}")
                written += 1
                if written % 5000 == 0:
                    print(f"Written {written:,} patches")

    manifest["total"] = written
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    sample = cv2.imread(str(next((args.output / "train" / "Positive").glob("*.png"))))
    assert written == 40000
    assert sample.shape[:2] == (args.size, args.size)
    print(f"Created {written:,} patches in {args.output}")
    print(json.dumps(manifest["splits"], indent=2))


if __name__ == "__main__":
    main()
