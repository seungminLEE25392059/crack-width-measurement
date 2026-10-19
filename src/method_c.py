"""Method C: U-Net crack segmentation (PyTorch).

Network (Task 1 report, Method C): four encoder stages of two 3x3 convolutions
with batch normalisation and ReLU, each followed by 2x2 max pooling; a
convolutional bottleneck; a symmetric decoder with transposed-convolution
up-sampling and skip connections; same padding; a final 1x1 convolution and
sigmoid.
Training (Colab, notebooks/train_unet_colab.ipynb): 256 x 256 tiles at native
resolution, weighted binary cross-entropy + soft Dice, Adam (learning rate
1e-4 to 1e-3), early stopping after 10 epochs without validation Dice gain.
Inference: overlapping tiles, averaged probabilities, validation-selected threshold.

Inputs : RGB uint8 image (H, W, 3), trained network, parameter dict
Outputs: bool crack mask (H, W) and float32 probability map (H, W) in [0, 1]
Owner  : Lingyu Wu

PyTorch is imported inside the functions, so the rest of the project runs
without it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config
from .utils import placeholder

DEFAULT_PARAMS = {
    "tile": config.TILE_SIZE,
    "overlap": config.TILE_OVERLAP,
    "base_channels": 32,       # channels of the first encoder stage
    "batch_size": 16,
    "learning_rate": 1e-4,
    "threshold": 0.5,          # probability threshold, chosen on validation data
}


def build_unet(params: dict):
    """Return the U-Net as a torch.nn.Module.

    TODO(Lingyu): implement the architecture in the module docstring.
    """
    raise NotImplementedError("TODO(Lingyu): build the U-Net in src/method_c.py")


@placeholder
def load_unet(path: Path, device: str | None = None):
    """Load trained weights from `path`; returns None while no model exists.

    TODO(Lingyu): net = build_unet(params); net.load_state_dict(torch.load(path,
    map_location=device)); net.eval(); return net.
    """
    return None


@placeholder
def segment_c(image: np.ndarray, net, params: dict) -> tuple[np.ndarray, np.ndarray]:
    """Segment cracks with the trained U-Net.

    TODO(Lingyu): normalise RGB as in training, cut tiles with data.make_tiles,
    predict them in batches under torch.no_grad(), average the overlaps with
    data.stitch_tiles and threshold. The placeholder returns an empty mask.
    """
    height, width = image.shape[:2]
    probability = np.zeros((height, width), np.float32)
    return probability >= params["threshold"], probability
