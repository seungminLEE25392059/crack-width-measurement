"""Create the fixed image-level train/val/test split (run once, then commit it).

Usage  : python scripts/make_split.py [--force]
Inputs : data/raw/images, data/raw/masks
Outputs: splits/splits.json, used unchanged by every later step
Owner  : Zitong Luo
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config, data, utils  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing split (only if the whole team agrees)")
    args = parser.parse_args()

    if config.SPLIT_FILE.exists() and not args.force:
        sys.exit("splits/splits.json already exists. The split must stay fixed; "
                 "use --force only if the whole team agrees.")
    utils.set_seed()
    print("Dataset check:", data.check_dataset())
    split = data.make_split(data.list_ids())
    path = data.save_split(split)
    counts = ", ".join(f"{name} {len(ids)}" for name, ids in split.items())
    print(f"Saved {path.relative_to(ROOT)}: {counts}")


if __name__ == "__main__":
    main()
