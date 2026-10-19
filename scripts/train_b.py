"""Train Method B on the training split and save the fitted model.

Usage  : python scripts/train_b.py
Inputs : splits/splits.json, training images and masks, params/B.json
Outputs: models/method_b.joblib (share it through Google Drive, not Git)
Owner  : Bowen Xu

TODO(Bowen): choose C, gamma and the confidence threshold on the validation
split and save them with utils.save_params("B", params).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config, data, method_b, utils  # noqa: E402


def main() -> None:
    utils.set_seed()
    split = data.load_split()
    params = utils.load_params("B", method_b.DEFAULT_PARAMS)
    model = method_b.train_b(split["train"], params)
    if model is None:
        print("method_b.train_b is still a placeholder: no model saved.")
        return
    path = config.MODELS_DIR / "method_b.joblib"
    method_b.save_model_b(model, path)
    print(f"Saved {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
