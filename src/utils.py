"""Shared helpers: seeding, placeholders, image and mask I/O, result paths,
parameter files, timing and environment logging.

Inputs : file paths and arrays from the other modules
Outputs: arrays, PNG files, params/<name>.json, log/environment.txt
Author : Seungmin Lee (integration)
"""
from __future__ import annotations

import importlib.metadata as metadata
import json
import platform
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np

from . import config


def set_seed(seed: int = config.SEED) -> None:
    """Seed Python, NumPy and, when installed, PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --- Placeholders --------------------------------------------------------------------

def placeholder(fn: Callable) -> Callable:
    """Mark a function as a temporary stand-in.

    A placeholder returns output of the agreed type and shape so the whole
    pipeline runs from day one, but it is not (or not yet confirmed as) the
    method described in the Task 1 report. The owner replaces or confirms the
    body and then deletes the decorator; scripts/check_interfaces.py lists every
    function that is still marked.
    """
    fn.is_placeholder = True
    return fn


def is_placeholder(fn: Callable) -> bool:
    """True while a function still carries the @placeholder decorator."""
    return bool(getattr(fn, "is_placeholder", False))


# --- Images and masks ------------------------------------------------------------------
# cv2.imread / cv2.imwrite fail on Windows paths with non-ASCII characters
# (for example a Korean or Chinese folder name), so files are read and written
# as bytes and decoded / encoded by OpenCV in memory.

def _imread(path: Path, flags: int, what: str) -> np.ndarray:
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        buffer = np.empty(0, np.uint8)
    decoded = cv2.imdecode(buffer, flags) if buffer.size else None
    if decoded is None:
        raise FileNotFoundError(f"Cannot read {what}: {path}")
    return decoded


def _imwrite(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", array)
    if not ok:
        raise OSError(f"Cannot encode {path}")
    encoded.tofile(str(path))


def read_image(path: Path) -> np.ndarray:
    """Read an image file as an RGB uint8 array of shape (H, W, 3)."""
    return cv2.cvtColor(_imread(path, cv2.IMREAD_COLOR, "image"), cv2.COLOR_BGR2RGB)


def read_mask(path: Path, threshold: int = config.MASK_THRESHOLD) -> np.ndarray:
    """Read a mask file as a bool array (H, W); crack pixels are True.

    Grey levels above `threshold` count as crack, which also removes the
    compression noise of JPEG masks.
    """
    return _imread(path, cv2.IMREAD_GRAYSCALE, "mask") > threshold


def save_mask(mask: np.ndarray, path: Path) -> None:
    """Save a bool mask as an 8-bit PNG (0 = background, 255 = crack)."""
    _imwrite(path, mask.astype(np.uint8) * 255)


def save_prob(prob: np.ndarray, path: Path) -> None:
    """Save a confidence or probability map with values in [0, 1] as an 8-bit PNG."""
    _imwrite(path, np.clip(np.rint(prob * 255.0), 0, 255).astype(np.uint8))


def read_prob(path: Path) -> np.ndarray:
    """Read a map written by save_prob back as float32 values in [0, 1]."""
    return _imread(path, cv2.IMREAD_GRAYSCALE, "probability map").astype(np.float32) / 255.0


def result_path(method: str, split: str, image_id: str, kind: str = "mask") -> Path:
    """Standard location of one output of a method.

    kind="mask" -> results/<method>/<split>/<image_id>.png
    kind="prob" -> results/<method>/<split>/<image_id>_prob.png
    """
    suffix = "" if kind == "mask" else f"_{kind}"
    return config.RESULTS_DIR / method / split / f"{image_id}{suffix}.png"


# --- Parameters ------------------------------------------------------------------------

def load_params(name: str, defaults: dict) -> dict:
    """Return `defaults` updated with params/<name>.json when that file exists."""
    params = dict(defaults)
    path = config.PARAMS_DIR / f"{name}.json"
    if path.exists():
        params.update(json.loads(path.read_text(encoding="utf-8")))
    return params


def save_params(name: str, params: dict) -> Path:
    """Save validation-selected parameters so the final test run uses exactly these."""
    config.PARAMS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.PARAMS_DIR / f"{name}.json"
    path.write_text(json.dumps(params, indent=2), encoding="utf-8")
    return path


# --- Timing and environment --------------------------------------------------------------

@contextmanager
def timer() -> Iterator[dict]:
    """Wall-clock timer: `with timer() as t: ...`, then read t["seconds"]."""
    result = {"seconds": 0.0}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["seconds"] = time.perf_counter() - start


def record_environment(path: Path | None = None) -> Path:
    """Write Python, OS and package versions to log/environment.txt for the report."""
    path = path or config.LOG_DIR / "environment.txt"
    lines = [f"python {platform.python_version()}", f"platform {platform.platform()}"]
    packages = ("numpy", "scipy", "scikit-image", "scikit-learn", "opencv-python-headless",
                "opencv-python", "matplotlib", "pandas", "joblib", "torch")
    for package in packages:
        try:
            lines.append(f"{package} {metadata.version(package)}")
        except metadata.PackageNotFoundError:
            continue
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
