"""Resolution experiment: how measurement error grows as image resolution drops.

Usage  : python scripts/run_resolution.py --method C
Plan (Task 1 report, resolution experiment): downscale every test image by
x1/2, x1/4 and x1/8 (images with cv2.INTER_AREA, reference masks with
cv2.INTER_NEAREST), run the method with its frozen parameters, measure the
cracks, divide widths and lengths by the scale factor to return to
full-resolution pixels, and compare with the full-resolution reference.
Run it only after the final test run, when all parameters are fixed.

Outputs: eval/test/tables/resolution.csv, eval/test/figures/resolution.png
Owner  : Junsen Liang
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config, utils  # noqa: E402

SCALES = (1.0, 0.5, 0.25, 0.125)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", choices=config.METHODS, default="C")
    parser.parse_args()
    utils.set_seed()
    # TODO(Junsen): for every test image and every scale in SCALES
    #   1. small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    #   2. run the method on `small` with utils.load_params(method, ...)
    #   3. measure.measure_cracks(prediction); divide widths and lengths by `scale`
    #   4. resize the predicted label image back to full size with cv2.INTER_NEAREST,
    #      then evaluate.match_cracks against the full-resolution reference labels
    #      and collect evaluate.measure_errors rows with a "scale" column
    #   5. save eval/test/tables/resolution.csv and plot error against scale (src.viz)
    raise NotImplementedError("TODO(Junsen): implement the resolution experiment")


if __name__ == "__main__":
    main()
