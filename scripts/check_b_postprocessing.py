"""Small regression check: python scripts/check_b_postprocessing.py."""
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import method_b


def main():
    mask = np.zeros((100, 100), dtype=bool)
    mask[1:33, 1:81] = True       # 2,560 pixels: retain
    mask[45:77, 1:81] = True
    mask[45, 1] = False          # 2,559 pixels: remove
    cleaned = method_b.postprocess_mask_b(mask, 2560)
    assert cleaned.sum() == 2560 and cleaned[1:33, 1:81].all()
    assert not cleaned[45:77].any() and mask.sum() == 5119
    assert method_b.postprocess_mask_b(np.eye(20, dtype=bool), 20).sum() == 20
    assert not method_b.postprocess_mask_b(np.zeros((5, 5), bool)).any()
    assert np.array_equal(method_b.postprocess_mask_b(mask, 0), mask)

    params = {**method_b.DEFAULT_PARAMS, "threshold": 0.9}
    scores = np.array([[0.89, 0.90, 0.91]], dtype=np.float32)
    assert method_b.mask_from_confidence_b(scores, {**params, "min_component_area": 0}).tolist() == [[False, True, True]]
    with patch.object(method_b, "extract_features_b", return_value=np.zeros(340)), \
         patch.object(method_b, "predict_patch_probabilities_b", return_value=np.array([0.95])):
        prediction, confidence = method_b.segment_b(np.zeros((32, 32, 3), np.uint8), object(), params)
        assert prediction.shape == (32, 32) and prediction.dtype == bool
        assert not prediction.any() and np.allclose(confidence, 0.95)
        unfiltered, same_confidence = method_b.segment_b(
            np.zeros((32, 32, 3), np.uint8), object(), {**params, "min_component_area": 0}
        )
        assert unfiltered.all() and np.array_equal(confidence, same_confidence)
    print("Method B postprocessing checks passed.")


if __name__ == "__main__":
    main()
