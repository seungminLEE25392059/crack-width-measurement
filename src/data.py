"""Dataset access, the image-level split and U-Net tiling.

Inputs : data/raw/images/<id>.<ext> and data/raw/masks/<id>.<ext>
Outputs: (image, mask) pairs, splits/splits.json, tiles for Method C
Owner  : Zitong Luo

The functions marked @placeholder are working first versions so the team can
start straight away. Zitong checks them on the real dataset, extends them and
then deletes the decorator.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import config
from .utils import placeholder, read_image, read_mask


@lru_cache(maxsize=None)
def _index(folder: str) -> dict[str, Path]:
    """Map file stem -> file path for every image file in `folder`."""
    root = Path(folder)
    if not root.is_dir():
        raise FileNotFoundError(f"Folder not found: {root} (see data/README.md)")
    return {p.stem: p for p in sorted(root.iterdir()) if p.suffix.lower() in config.IMAGE_EXTS}


def list_ids() -> list[str]:
    """Sorted IDs (file stems) that have both an image and a mask.

    TODO(Zitong): confirm on the real dataset that image and mask names match.
    """
    images = _index(str(config.IMAGE_DIR))
    masks = _index(str(config.MASK_DIR))
    return sorted(set(images) & set(masks))


def load_pair(image_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one photograph and its reference mask.

    Returns:
        image: RGB uint8 array (H, W, 3)
        mask:  bool array (H, W), True = crack
    TODO(Zitong): confirm config.MASK_THRESHOLD on the real masks.
    """
    image = read_image(_index(str(config.IMAGE_DIR))[image_id])
    mask = read_mask(_index(str(config.MASK_DIR))[image_id])
    if image.shape[:2] != mask.shape:
        raise ValueError(f"{image_id}: image {image.shape[:2]} and mask {mask.shape} differ in size")
    return image, mask


def check_dataset() -> dict:
    """Check the dataset before the split and return a summary for Section 2.2.

    TODO(Zitong): report images without masks and masks without images, size
    mismatches, mask grey levels before thresholding and image sizes. This first
    version only counts matched pairs.
    """
    return {"pairs": len(list_ids())}


def make_split(ids: list[str], ratios: tuple[float, float, float] = config.SPLIT_RATIOS,
               seed: int = config.SEED) -> dict[str, list[str]]:
    """Split image IDs into train / val / test at the photograph level.

    Each original photograph goes to exactly one subset before any tiling, so
    tiles cut from one photograph never appear in two subsets.
    """
    ordered = sorted(ids)
    rng = np.random.default_rng(seed)
    shuffled = [ordered[i] for i in rng.permutation(len(ordered))]
    n_train = round(ratios[0] * len(ordered))
    n_val = round(ratios[1] * len(ordered))
    return {
        "train": sorted(shuffled[:n_train]),
        "val": sorted(shuffled[n_train:n_train + n_val]),
        "test": sorted(shuffled[n_train + n_val:]),
    }


def save_split(split: dict[str, list[str]], path: Path = config.SPLIT_FILE) -> Path:
    """Write the split, with its seed and ratios, to splits/splits.json."""
    payload = {
        "seed": config.SEED,
        "ratios": list(config.SPLIT_RATIOS),
        "counts": {name: len(ids) for name, ids in split.items()},
        **split,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_split(path: Path = config.SPLIT_FILE) -> dict[str, list[str]]:
    """Read splits/splits.json -> {"train": [...], "val": [...], "test": [...]}."""
    if not path.exists():
        raise FileNotFoundError("splits/splits.json is missing: run python scripts/make_split.py")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {name: payload[name] for name in ("train", "val", "test")}


def _origins(length: int, tile: int, stride: int) -> list[int]:
    """Tile start positions that cover [0, length); the last tile may overhang."""
    starts = [0]
    while starts[-1] + tile < length:
        starts.append(starts[-1] + stride)
    return starts


def make_tiles(array: np.ndarray, tile: int = config.TILE_SIZE,
               overlap: int = config.TILE_OVERLAP) -> tuple[np.ndarray, list[tuple[int, int]], np.ndarray]:
    """Cut an (H, W) or (H, W, C) array into overlapping tile x tile patches.

    Border tiles are zero-padded. Returns:
        tiles:   (N, tile, tile[, C]) array
        origins: (row, col) of each tile's top-left corner in the original array
        valid:   (N, tile, tile) bool, False on padding (exclude it from loss and evaluation)
    TODO(Zitong, Lingyu): agree on the overlap and the padding mode (zeros or reflect).
    """
    height, width = array.shape[:2]
    stride = tile - overlap
    rows = _origins(height, tile, stride)
    cols = _origins(width, tile, stride)
    pad_h = rows[-1] + tile - height
    pad_w = cols[-1] + tile - width
    padded = np.pad(array, ((0, pad_h), (0, pad_w)) + ((0, 0),) * (array.ndim - 2))
    valid_full = np.pad(np.ones((height, width), bool), ((0, pad_h), (0, pad_w)))
    origins = [(r, c) for r in rows for c in cols]
    tiles = np.stack([padded[r:r + tile, c:c + tile] for r, c in origins])
    valid = np.stack([valid_full[r:r + tile, c:c + tile] for r, c in origins])
    return tiles, origins, valid


def stitch_tiles(tile_maps: np.ndarray, origins: list[tuple[int, int]],
                 shape: tuple[int, int]) -> np.ndarray:
    """Average overlapping (N, tile, tile) maps back into one (H, W) float32 map."""
    tile = tile_maps.shape[1]
    height, width = shape
    total_h = max(r for r, _ in origins) + tile
    total_w = max(c for _, c in origins) + tile
    summed = np.zeros((total_h, total_w), np.float64)
    count = np.zeros((total_h, total_w), np.float64)
    for tile_map, (r, c) in zip(tile_maps, origins):
        summed[r:r + tile, c:c + tile] += tile_map
        count[r:r + tile, c:c + tile] += 1
    return (summed / np.maximum(count, 1))[:height, :width].astype(np.float32)


def dataset_stats(ids: list[str]) -> dict:
    """Dataset summary for Section 2.2 of the report.

    TODO(Zitong): add image sizes and the reference width distribution
    (measure.measure_cracks). This first version reports the image count and the
    mean crack-pixel ratio.
    """
    ratios = [load_pair(image_id)[1].mean() for image_id in ids]
    return {"images": len(ids), "crack_pixel_ratio_mean": float(np.mean(ratios)) if ratios else 0.0}
