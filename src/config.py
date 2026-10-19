"""Project-wide paths and shared settings.

Every module and script reads its paths from here, so a folder or a shared
setting is changed in exactly one place. The dataset location can be
overridden with the CRACK_DATA_DIR environment variable (for example on Colab,
where the dataset sits on Google Drive).

Inputs : none (optionally the CRACK_DATA_DIR environment variable)
Outputs: constants imported by the other modules
Author : Seungmin Lee (integration)
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# --- Dataset ---------------------------------------------------------------------
DATA_DIR = Path(os.environ.get("CRACK_DATA_DIR", ROOT / "data" / "raw"))
IMAGE_DIR = DATA_DIR / "images"      # original photographs
MASK_DIR = DATA_DIR / "masks"        # reference masks, same file stem as the image
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
MASK_THRESHOLD = 127                 # mask grey levels above this value are crack

# --- Project folders ---------------------------------------------------------------
SPLIT_FILE = ROOT / "splits" / "splits.json"   # fixed train/val/test IDs (in Git)
PARAMS_DIR = ROOT / "params"                   # validation-selected parameters (in Git)
RESULTS_DIR = ROOT / "results"                 # masks from run_method.py (not in Git)
MODELS_DIR = ROOT / "models"                   # trained SVM / U-Net (not in Git)
EVAL_DIR = ROOT / "eval"                       # report tables and figures
LOG_DIR = ROOT / "log"                         # decision log and environment record

# --- Shared settings agreed in the Task 1 report --------------------------------------
SEED = 42
SPLIT_RATIOS = (0.70, 0.15, 0.15)    # train / val / test, by original photograph
METHODS = ("A", "B", "C", "D")
CONNECTIVITY = 2                     # skimage convention: 2 = 8-connected crack instances
TILE_SIZE = 256                      # U-Net tiles at native resolution
TILE_OVERLAP = 32                    # TODO(Lingyu): confirm on validation data
TOLERANCE_PX = 2                     # boundary tolerance for the tolerance F1
SEVERITY_LABELS = ("minor", "moderate", "severe")
