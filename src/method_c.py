"""Method C: native-resolution U-Net training and overlapping probability inference.

Inputs: RGB uint8 HxWx3 images, fixed photograph splits, bool reference masks.
Outputs: bool HxW masks, float32 probabilities, versioned checkpoints/logs.
Owner: Lingyu Wu. PyTorch is required only for training and inference.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import tempfile
import time
import uuid
import warnings
from pathlib import Path

import numpy as np
from . import config, data, utils

try:
    from torch.utils.data import Dataset
except ImportError:  # Keep the team's non-C pipeline importable without torch.
    Dataset = object

DEFAULT_PARAMS = {
    "tile": config.TILE_SIZE, "overlap": config.TILE_OVERLAP,
    "depth": 4, "base_channels": 32, "batch_size": 16,
    "learning_rate": 1e-4, "threshold": 0.5,
    "max_epochs": 200, "patience": 10, "bce_weight": 0.5,
    "seed": 42, "num_workers": 0, "foreground_sample_fraction": 0.7,
    "samples_per_epoch": None, "max_train_tiles": None,
    "normalization": "RGB uint8 / 255", "pos_weight_cap": 100.0,
}
CHECKPOINT_VERSION = 2
FORMAL_INITIAL_PARAMS = {
    "depth": 4, "base_channels": 32, "tile": 256, "overlap": 32,
    "batch_size": 8, "learning_rate": 1e-4, "max_epochs": 100,
    "patience": 10, "samples_per_epoch": 4096, "max_train_tiles": None,
}


def _params(params: dict) -> dict:
    p = {**DEFAULT_PARAMS, **params}
    for key in ("tile", "depth", "base_channels", "batch_size", "max_epochs", "patience"):
        if not isinstance(p[key], int) or isinstance(p[key], bool) or p[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if not isinstance(p["overlap"], int) or not 0 <= p["overlap"] < p["tile"]:
        raise ValueError("overlap must be integer pixels in [0, tile)")
    if p["tile"] % 2 ** p["depth"] or p["tile"] // 2 ** p["depth"] < 2:
        raise ValueError("tile must be divisible by 2**depth with bottleneck >= 2 pixels")
    if not 0 <= p["threshold"] <= 1 or not 0 <= p["bce_weight"] <= 1:
        raise ValueError("threshold and bce_weight must be in [0,1]")
    if not 0 < p["foreground_sample_fraction"] < 1:
        raise ValueError("foreground_sample_fraction must be strictly between 0 and 1")
    if p["learning_rate"] <= 0 or p["num_workers"] < 0 or p["pos_weight_cap"] < 1:
        raise ValueError("Invalid learning rate, worker count or class-weight cap")
    for key in ("max_train_tiles", "samples_per_epoch"):
        if p[key] is not None and (not isinstance(p[key], int) or p[key] <= 0):
            raise ValueError(f"{key} must be None or a positive integer")
    if p["normalization"] != "RGB uint8 / 255":
        raise ValueError("Only RGB uint8 / 255 normalization is supported")
    return p


def seed_training(seed: int = 42) -> None:
    """Seed augmentation and sampling; request deterministic kernels."""
    import torch
    utils.set_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def _seed_worker(worker_id: int) -> None:
    import torch
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)


def build_unet(params: dict):
    """Build symmetric U-Net returning logits, with versioned architecture params."""
    import torch
    from torch import nn
    p = _params(params)

    def block(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU())

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            channels = [p["base_channels"] * 2 ** i for i in range(p["depth"]+1)]
            self.encoders = nn.ModuleList([block(3 if i == 0 else channels[i-1], channels[i])
                                           for i in range(p["depth"])])
            self.pool = nn.MaxPool2d(2)
            self.bottleneck = block(channels[-2], channels[-1])
            self.ups = nn.ModuleList([nn.ConvTranspose2d(channels[i+1], channels[i], 2, stride=2)
                                      for i in reversed(range(p["depth"]))])
            self.decoders = nn.ModuleList([block(2*channels[i], channels[i])
                                          for i in reversed(range(p["depth"]))])
            self.head = nn.Conv2d(channels[0], 1, 1)
            self.unet_params = p

        def forward(self, x):
            skips = []
            for encoder in self.encoders:
                x = encoder(x)
                skips.append(x)
                x = self.pool(x)
            x = self.bottleneck(x)
            for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
                x = up(x)
                if x.shape[-2:] != skip.shape[-2:]:
                    raise ValueError("Input dimensions must be divisible by 2**depth")
                x = decoder(torch.cat((skip, x), dim=1))
            return self.head(x)

    return UNet()


def load_unet(path: Path, device: str | None = None):
    """Load versioned team checkpoint on CPU/GPU; missing file returns None.

    Bare state dicts are rejected since architecture metadata is required.
    Only load trusted team checkpoints; weights_only avoids arbitrary unpickling.
    """
    path = Path(path)
    if path.name in ("best.pt", "last.pt") and (path.parent/"commits").is_dir():
        receipt = _verified_receipt(path.parent)
        relative = receipt["best_checkpoint"] if path.name == "best.pt" else receipt["checkpoint"]
        if relative is None:
            return None
        path = path.parent/relative
    if not path.is_file():
        return None
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("format_version") not in (1, CHECKPOINT_VERSION):
        raise ValueError("Expected a versioned Method C checkpoint with architecture metadata")
    net = build_unet(saved["params"])
    net.load_state_dict(saved["model_state"])
    net.to(device).eval()
    net.checkpoint_metadata = {k: saved[k] for k in ("epoch", "val_dice", "run_kind")}
    return net


def _origins(shape: tuple[int, int], p: dict) -> list[tuple[int, int]]:
    stride = p["tile"]-p["overlap"]
    return [(r, c) for r in data._origins(shape[0], p["tile"], stride)
            for c in data._origins(shape[1], p["tile"], stride)]


def _crop(array: np.ndarray, r: int, c: int, tile: int) -> np.ndarray:
    part = array[r:r+tile, c:c+tile]
    return np.pad(part, ((0, tile-part.shape[0]), (0, tile-part.shape[1]))
                  + ((0, 0),)*(array.ndim-2))


def segment_c(image: np.ndarray, net, params: dict) -> tuple[np.ndarray, np.ndarray]:
    """Stream tile batches; average probabilities BEFORE whole-image thresholding.

    Zero padding is cropped from each contribution; all original pixels covered.
    """
    import torch
    if net is None:
        raise ValueError("Method C requires trained weights; no fallback mask is generated")
    effective = dict(params)
    if hasattr(net, "unet_params"):
        # The loaded network defines architecture. The public interface checker
        # passes DEFAULT_PARAMS even for a recorded CPU architecture adaptation.
        for key in ("depth", "base_channels"):
            effective[key] = net.unet_params[key]
    p = _params(effective)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or min(image.shape[:2]) == 0:
        raise ValueError("Expected nonempty RGB uint8 image (H,W,3)")
    if hasattr(net, "unet_params"):
        for key in ("normalization",):
            if p[key] != net.unet_params[key]:
                raise ValueError(f"{key} differs from checkpoint architecture/normalization")
    device = next(net.parameters()).device
    net.eval()
    total = np.zeros(image.shape[:2], np.float32)
    count = np.zeros(image.shape[:2], np.uint32)
    origins = _origins(image.shape[:2], p)
    with torch.inference_mode():
        for start in range(0, len(origins), p["batch_size"]):
            batch_origins = origins[start:start+p["batch_size"]]
            batch = np.stack([_crop(image, r, c, p["tile"]) for r, c in batch_origins])
            inputs = torch.from_numpy(batch.transpose(0, 3, 1, 2).copy()).to(device, dtype=torch.float32)/255
            maps = torch.sigmoid(net(inputs))[:, 0].cpu().numpy()
            if not np.isfinite(maps).all():
                raise FloatingPointError("Nonfinite network probability")
            for prob, (r, c) in zip(maps, batch_origins):
                h, w = min(p["tile"], image.shape[0]-r), min(p["tile"], image.shape[1]-c)
                total[r:r+h, c:c+w] += prob[:h, :w]
                count[r:r+h, c:c+w] += 1
    if not np.all(count > 0):
        raise RuntimeError("Sliding-window inference left uncovered pixels")
    probability = (total/count).astype(np.float32)
    return probability >= p["threshold"], probability


class TileDataset(Dataset):
    """DataLoader map-style dataset: coordinates on CPU, ONE photograph cached.

    No resizing. Train-only paired D4 geometry and image-only photometry.
    Tile provenance is (photograph ID, row, column, foreground pixels).
    """
    def __init__(self, ids: list[str], params: dict, train: bool):
        self.params, self.train, self.ids = _params(params), train, list(ids)
        self.items = []
        self._cache_id, self._cache_pair = None, None
        positives = pixels = 0
        for image_id in self.ids:
            image, mask = data.load_pair(image_id)
            positives += int(mask.sum())
            pixels += mask.size
            for r, c in _origins(mask.shape, self.params):
                self.items.append((image_id, r, c, int(mask[r:r+self.params["tile"], c:c+self.params["tile"]].sum())))
        if not self.items:
            raise ValueError("No tiles: supply nonempty photograph IDs")
        candidate_foreground = sum(item[3] > 0 for item in self.items)
        candidate_count = len(self.items)
        cap = self.params["max_train_tiles"] if train else None
        if cap is not None and len(self.items) > cap:
            indices = np.random.default_rng(self.params["seed"]).choice(len(self.items), cap, replace=False)
            self.items = [self.items[i] for i in sorted(indices)]
        self.stats = {
            "images": len(ids), "candidate_tiles": candidate_count,
            "candidate_crack_tiles": candidate_foreground,
            "candidate_background_tiles": candidate_count-candidate_foreground,
            "selected_tiles": len(self.items),
            "crack_tiles": sum(item[3] > 0 for item in self.items),
            "background_tiles": sum(item[3] == 0 for item in self.items),
            "foreground_pixels": positives, "valid_original_pixels": pixels,
            "foreground_pixel_ratio": positives/pixels,
            "statistics_source": "unresampled original labels for supplied IDs",
        }

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        import torch
        image_id, r, c, _ = self.items[index]
        if image_id != self._cache_id:
            self._cache_pair, self._cache_id = data.load_pair(image_id), image_id
        image, mask = self._cache_pair
        tile = self.params["tile"]
        x = _crop(image, r, c, tile).astype(np.float32)/255
        y = _crop(mask, r, c, tile).astype(np.float32)
        valid = np.zeros((tile, tile), np.float32)
        valid[:min(tile, mask.shape[0]-r), :min(tile, mask.shape[1]-c)] = 1
        if self.train:
            k, flip = np.random.randint(4), bool(np.random.randint(2))
            x, y, valid = [np.rot90(a, k) for a in (x, y, valid)]
            if flip:
                x, y, valid = [np.flip(a, axis=1) for a in (x, y, valid)]
            x = np.clip((x-0.5)*np.random.uniform(0.9, 1.1)+0.5+np.random.uniform(-0.1, 0.1), 0, 1)
            x *= valid[..., None]
        return tuple(torch.from_numpy(np.ascontiguousarray(a)) for a in (x.transpose(2, 0, 1), y[None], valid[None]))


def weighted_bce_dice(logits, target, valid, pos_weight: float, bce_weight: float = 0.5):
    """Stable logits BCE + per-tile soft Dice, excluding padding from both.

    Normalize BCE by summed class weights to keep its scale near soft Dice.
    """
    import torch
    from torch.nn import functional as F
    if pos_weight <= 0 or not 0 <= bce_weight <= 1:
        raise ValueError("Invalid loss weights")
    target, valid = target.to(logits.dtype), valid.to(logits.dtype)
    weight = logits.new_tensor(pos_weight)
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=weight, reduction="none")
    bce = (bce*valid).sum()/((1+(weight-1)*target)*valid).sum().clamp_min(1)
    probability = torch.sigmoid(logits)
    dims = tuple(range(1, logits.ndim))
    dice = (2*(probability*target*valid).sum(dims)+1e-6)/(
        (probability*valid).sum(dims)+(target*valid).sum(dims)+1e-6)
    active = valid.sum(dims) > 0
    dice_loss = ((1-dice)*active).sum()/active.sum().clamp_min(1)
    return bce_weight*bce+(1-bce_weight)*dice_loss


def _check_ids(ids: list[str], subset: str) -> None:
    if not ids or len(ids) != len(set(ids)) or not set(ids) <= set(data.load_split()[subset]):
        raise ValueError(f"IDs must be nonempty unique members of fixed {subset} split")


def _strict_counts(pred, ref):
    return np.array([np.count_nonzero(pred & ref), np.count_nonzero(pred & ~ref),
                     np.count_nonzero(~pred & ref)], dtype=np.int64)


def _scores(counts) -> dict:
    tp, fp, fn = map(int, counts)
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp/(tp+fp) if tp+fp else float(fn == 0),
            "recall": tp/(tp+fn) if tp+fn else float(fp == 0),
            "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,
            "iou": tp/(tp+fp+fn) if tp+fp+fn else 1.}


def validation_dice(net, ids: list[str], params: dict) -> float:
    """Micro crack Dice on full reconstructed validation photographs."""
    _check_ids(ids, "val")
    counts = np.zeros(3, np.int64)
    for image_id in ids:
        image, ref = data.load_pair(image_id)
        pred, _ = segment_c(image, net, params)
        counts += _strict_counts(pred, ref)
    return _scores(counts)["f1"]


def file_sha256(path: Path) -> str:
    """Hash file bytes, including checkpoint and uploaded dataset archives."""
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def _publish_file(local: Path, destination: Path) -> str:
    """Copy a complete local file, verify staged bytes, replace, verify again.

    POSIX/local replace is atomic; Drive FUSE does NOT guarantee remote atomicity
    or durability. Immutable snapshots plus verified commit receipts are the
    recovery authority. An interrupted .pending file is never a checkpoint.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = file_sha256(local)
    pending = destination.with_name(destination.name + ".pending-" + uuid.uuid4().hex)
    with Path(local).open("rb") as source, pending.open("xb") as target:
        shutil.copyfileobj(source, target, length=1024*1024)
        target.flush()
        try:
            os.fsync(target.fileno())
        except OSError as error:
            import errno
            if error.errno not in (errno.EINVAL, errno.ENOTSUP, errno.ENOSYS):
                raise
    if file_sha256(pending) != digest:
        raise OSError(f"Staged upload checksum failed: {pending}")
    os.replace(pending, destination)
    if file_sha256(destination) != digest:
        raise OSError(f"Published checksum failed: {destination}")
    return digest


def _write_json(path: Path, payload: dict) -> None:
    """Serialize locally first, then use checked temporary-file publication."""
    encoded = json.dumps(payload, indent=2, allow_nan=False).encode("utf-8")
    with tempfile.TemporaryDirectory(prefix="crack-json-") as tmp:
        local = Path(tmp)/"payload.json"
        local.write_bytes(encoded)
        _publish_file(local, Path(path))


def environment_info() -> dict:
    """Report actual runtime versions; never install/replace GPU PyTorch."""
    import torch
    import cv2
    import scipy
    import skimage
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "cuda_build": torch.version.cuda, "numpy": np.__version__,
            "scipy": scipy.__version__, "scikit_image": skimage.__version__,
            "opencv": cv2.__version__}


def check_run_mode(run_kind: str) -> dict:
    """Formal requires a successful real CUDA operation; smoke is explicit."""
    import torch
    if run_kind not in ("formal", "smoke"):
        raise ValueError("Choose run_kind='formal' or run_kind='smoke' explicitly")
    available = torch.cuda.is_available()
    if run_kind == "formal" and not available:
        raise RuntimeError("Formal training requires CUDA. Select a Colab GPU runtime; no CPU/smoke fallback is allowed.")
    info = {"device": "cuda" if available else "cpu", "gpu": None, "gpu_memory_bytes": None}
    if available:
        try:
            probe = torch.arange(4, device="cuda", dtype=torch.float32)
            result = float((probe*probe).sum().item())
            torch.cuda.synchronize()
            if result != 14.:
                raise RuntimeError("CUDA tensor result failed verification")
        except Exception as error:
            raise RuntimeError("CUDA was detected but the GPU tensor operation failed; training stopped") from error
        info.update(gpu=torch.cuda.get_device_name(0),
                    gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory)
    print(f"Mode={run_kind}; device={info['device']}; GPU={info['gpu']}; memory={info['gpu_memory_bytes']}", flush=True)
    return info


def _source_fingerprint() -> dict:
    return {name: file_sha256(Path(__file__).parent/name)
            for name in ("method_c.py", "data.py", "config.py", "utils.py")}


def _capture_rng(sampler_generator, loader_generator) -> dict:
    import torch
    state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": (state[0], state[1].tolist(), state[2], state[3], state[4]),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "sampler_generator": sampler_generator.get_state(),
            "loader_generator": loader_generator.get_state()}


def _restore_rng(state: dict, sampler_generator, loader_generator) -> None:
    import torch
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        if not torch.cuda.is_available() or len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA RNG state/device count is incompatible with this runtime")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])
    sampler_generator.set_state(state["sampler_generator"].cpu())
    loader_generator.set_state(state["loader_generator"].cpu())


def _verified_receipt(output_dir: Path) -> dict:
    """Find newest fully uploaded commit; fall back if a newer upload is damaged."""
    output_dir = Path(output_dir)
    failures = []
    for path in sorted((output_dir/"commits").glob("epoch_*.json"), reverse=True):
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
            snapshot = (output_dir/receipt["checkpoint"]).resolve()
            if snapshot.parent != (output_dir/"snapshots").resolve():
                raise ValueError("Checkpoint escaped run snapshot directory")
            if file_sha256(snapshot) != receipt["sha256"]:
                raise ValueError("Snapshot checksum mismatch")
            if receipt["best_checkpoint"]:
                best = (output_dir/receipt["best_checkpoint"]).resolve()
                if best.parent != (output_dir/"snapshots").resolve() or file_sha256(best) != receipt["best_sha256"]:
                    raise ValueError("Best snapshot checksum mismatch")
            if failures:
                warnings.warn("Ignored incomplete/corrupt commits: " + "; ".join(failures))
            return receipt
        except (OSError, ValueError, KeyError, TypeError) as error:
            failures.append(f"{path.name}: {error}")
    raise FileNotFoundError("No verified committed checkpoint in " + str(output_dir))


def read_training_checkpoint(path: Path) -> dict:
    """Read v2 resume state on CPU. last.pt resolves through checked receipts."""
    import torch
    path = Path(path)
    if path.name == "last.pt":
        receipt = _verified_receipt(path.parent)
        actual = path.parent/receipt["checkpoint"]
    else:
        actual = path
    payload = torch.load(actual, map_location="cpu", weights_only=True)
    if payload.get("format_version") != CHECKPOINT_VERSION or "rng_state" not in payload:
        raise ValueError("Resume needs a v2 complete training checkpoint; v1 weights remain loadable for inference only")
    return payload


def show_training_progress(output_dir: Path) -> dict:
    """Read committed state, plus advisory in-progress phase; no training occurs."""
    output_dir = Path(output_dir)
    receipt = _verified_receipt(output_dir)
    result = {"completed_epoch": receipt["epoch"], "checkpoint": receipt["checkpoint"],
              "sha256": receipt["sha256"], "best_epoch": receipt["best_epoch"],
              "best_val_dice": receipt["best_val_dice"]}
    try:
        result["progress"] = json.loads((output_dir/"progress.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        result["progress"] = "No readable advisory progress; committed checkpoint is authoritative"
    return result


def _save_epoch(output_dir: Path, payload: dict, previous: dict | None, improved: bool) -> dict:
    """Commit immutable epoch snapshots; preserve every old snapshot for recovery.

    Receipts are written only after upload verification. last.pt/best.pt are
    convenient aliases; recovery never trusts a potentially half-updated alias.
    Keeping snapshots costs disk space (about 0.1 GB/epoch for depth4/base32).
    """
    import torch
    with tempfile.TemporaryDirectory(prefix="crack-checkpoint-") as tmp:
        local = Path(tmp)/"complete.pt"
        torch.save(payload, local)
        # Close serialization and make the local file durable before uploading.
        with local.open("r+b") as stream:
            os.fsync(stream.fileno())
        digest = file_sha256(local)
        relative = f"snapshots/epoch_{payload['epoch']:06d}_{digest[:16]}.pt"
        immutable = output_dir/relative
        if immutable.exists():
            if file_sha256(immutable) != digest:
                raise OSError("Snapshot name collision")
        else:
            _publish_file(local, immutable)
        best_path = relative if improved else previous["best_checkpoint"] if previous else None
        best_hash = digest if improved else previous["best_sha256"] if previous else None
        receipt = {"format_version": CHECKPOINT_VERSION, "epoch": payload["epoch"],
                   "checkpoint": relative, "sha256": digest,
                   "best_checkpoint": best_path, "best_sha256": best_hash,
                   "best_epoch": payload["best_epoch"], "best_val_dice": payload["best_val_dice"]}
        _write_json(output_dir/f"commits/epoch_{payload['epoch']:06d}_{digest[:16]}.json", receipt)
        _publish_file(local, output_dir/"last.pt")
        if improved:
            _publish_file(local, output_dir/"best.pt")
        _write_json(output_dir/"training.json", payload["log"])
        return receipt


def _save_training_artifacts(output_dir: Path, log: dict) -> None:
    """Persist real curves and history after every completed epoch, not at export."""
    import csv
    from matplotlib.figure import Figure
    if not log["history"]:
        return
    with tempfile.TemporaryDirectory(prefix="crack-curves-") as tmp:
        tmp = Path(tmp)
        with (tmp/"history.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(log["history"][0]))
            writer.writeheader()
            writer.writerows(log["history"])
        _publish_file(tmp/"history.csv", output_dir/"history.csv")
        for key, title in (("training_loss", "Training loss"), ("validation_dice", "Full-image validation Dice")):
            fig = Figure(figsize=(6, 4))
            ax = fig.subplots()
            ax.plot([row["epoch"] for row in log["history"]], [row[key] for row in log["history"]], marker="o")
            ax.set(xlabel="Epoch", ylabel=title, title=log["run_kind"]+": "+title)
            if key == "validation_dice":
                ax.set_ylim(0, 1)
            ax.grid(alpha=.25)
            fig.tight_layout()
            fig.savefig(tmp/f"{key}.png", dpi=200)
            _publish_file(tmp/f"{key}.png", output_dir/f"figures/{key}.png")


def _validate_resume(saved: dict, params: dict, train_ids: list[str], val_ids: list[str],
                     run_kind: str, device: str) -> None:
    if saved["params"] != params:
        changed = [key for key in set(saved["params"]) | set(params) if saved["params"].get(key) != params.get(key)]
        raise ValueError("Resume configuration mismatch: " + ", ".join(sorted(changed)))
    log = saved["log"]
    if log["train_ids"] != train_ids or log["val_ids"] != val_ids:
        raise ValueError("Resume photograph IDs/order differ")
    if log["split_sha256"] != file_sha256(config.SPLIT_FILE):
        raise ValueError("Resume fixed split hash differs")
    if log["source_fingerprints"] != _source_fingerprint():
        raise ValueError("Resume training/data source code differs; restore the original uploaded version")
    if saved["run_kind"] != run_kind or log["device"] != device:
        raise ValueError("Resume mode/device type differs")
    current = environment_info()
    # Strict package matching makes the repeatability boundary explicit.
    for key in ("torch", "numpy", "opencv", "scipy", "scikit_image"):
        if log["environment"][key] != current[key]:
            raise ValueError(f"Resume {key} version differs: {log['environment'][key]} vs {current[key]}")
    if log["environment"]["python"].split(".")[:2] != current["python"].split(".")[:2]:
        raise ValueError("Resume Python major/minor version differs")


def read_warm_start(path: Path, params: dict, train_ids: list[str], val_ids: list[str],
                    expected_sha256: str) -> tuple[dict, dict]:
    """Read an explicitly checksummed v2 weight source; NEVER resume its progress.

    The caller must supply a frozen trusted file, not a live Drive alias. Adam,
    epoch/history/patience and random state are intentionally reset in a new run.
    Existing split/ID checks prevent accidental test-trained initialization.
    """
    import torch
    path = Path(path)
    digest = file_sha256(path)
    if not expected_sha256 or digest != expected_sha256.lower():
        raise ValueError("Warm-start source SHA256 mismatch or missing explicit hash")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if file_sha256(path) != digest:
        raise OSError("Warm-start source changed while being read")
    if state.get("format_version") != CHECKPOINT_VERSION or state.get("run_kind") not in ("formal", "smoke"):
        raise ValueError("Warm start requires a trusted v2 training checkpoint")
    p = _params(params)
    for key in ("depth", "base_channels", "normalization", "tile", "overlap"):
        if state["params"][key] != p[key]:
            raise ValueError("Warm-start architecture/preprocessing mismatch: " + key)
    log = state["log"]
    if log["split_sha256"] != file_sha256(config.SPLIT_FILE):
        raise ValueError("Warm-start split hash differs")
    if log["train_ids"] != train_ids or log["val_ids"] != val_ids:
        raise ValueError("Warm-start source photograph IDs/order differ")
    if any(not bool(torch.isfinite(value).all()) for value in state["model_state"].values()
           if value.is_floating_point()):
        raise ValueError("Warm-start model contains nonfinite weights")
    provenance = {"kind": "warm_start_weights_only", "source_path": str(path.resolve()),
                  "source_sha256": digest, "source_epoch": state["epoch"],
                  "source_run_kind": state["run_kind"],
                  "source_environment": log["environment"], "source_hardware": log["hardware"],
                  "source_fingerprints": log["source_fingerprints"],
                  "optimizer_reinitialized": True, "history_epoch_patience_rng_reset": True,
                  "strict_resume": False}
    return state["model_state"], provenance


def train_unet(train_ids: list[str], val_ids: list[str], params: dict,
               output_dir: Path, run_kind: str = "formal", resume_from: Path | None = None,
               *, stop_after_epoch: int | None = None, warm_start_from: Path | None = None,
               warm_start_sha256: str | None = None) -> dict:
    """Train or resume at a COMPLETED epoch boundary, retaining Adam and all RNGs.

    Formal mode requires CUDA and complete 320/69 splits. stop_after_epoch is an
    explicit graceful pause used for tests/manual pauses; max_epochs stays fixed.
    Same-code/same-package CPU runs with num_workers=0 are tested bit-for-bit.
    Mid-epoch replay, changed GPU/hardware and nondeterministic CUDA kernels do
    not carry that guarantee. One active writer per run is required.
    """
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler
    p, output_dir = _params(params), Path(output_dir)
    runtime = check_run_mode(run_kind)
    device = runtime["device"]
    _check_ids(train_ids, "train")
    _check_ids(val_ids, "val")
    fixed = data.load_split()
    if set(train_ids) & set(val_ids):
        raise ValueError("Training and validation photographs must not overlap")
    if run_kind == "formal":
        if len(train_ids) != 320 or len(val_ids) != 69 or set(train_ids) != set(fixed["train"]) or set(val_ids) != set(fixed["val"]):
            raise ValueError("Formal training requires all 320 train / 69 validation photographs")
        for key in ("depth", "base_channels", "tile", "overlap", "max_train_tiles"):
            if p[key] != FORMAL_INITIAL_PARAMS[key]:
                raise ValueError(f"Formal architecture/data setting {key} must be {FORMAL_INITIAL_PARAMS[key]}")
    if device == "cpu" and p["max_epochs"] > 10:
        raise ValueError("CPU smoke runs require explicit max_epochs <= 10")
    if stop_after_epoch is not None and (not isinstance(stop_after_epoch, int) or stop_after_epoch < 1):
        raise ValueError("stop_after_epoch must be a positive completed epoch")
    if warm_start_from is not None and resume_from is not None:
        raise ValueError("Warm start and strict resume are mutually exclusive")
    saved = receipt = None
    if resume_from is None:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError("Training output exists; use resume_from=last.pt or a new Run ID")
    else:
        if Path(resume_from).resolve() != (output_dir/"last.pt").resolve():
            raise ValueError("Resume must use this run's training/last.pt, never best.pt or another run")
        saved = read_training_checkpoint(resume_from)
        _validate_resume(saved, p, train_ids, val_ids, run_kind, device)
        receipt = _verified_receipt(output_dir)
    warm_weights = warm_provenance = None
    if warm_start_from is not None:
        warm_weights, warm_provenance = read_warm_start(
            warm_start_from, p, train_ids, val_ids, warm_start_sha256)
        if run_kind == "formal" and warm_provenance["source_run_kind"] != "formal":
            raise ValueError("Formal warm start must not use smoke weights")
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_training(p["seed"])
    start = time.perf_counter()
    dataset = TileDataset(train_ids, p, train=True)
    stats = dataset.stats
    if stats["foreground_pixels"] == 0:
        raise ValueError("Training labels have no foreground")
    raw_weight = (stats["valid_original_pixels"]-stats["foreground_pixels"])/stats["foreground_pixels"]
    pos_weight = float(np.clip(raw_weight, 1, p["pos_weight_cap"]))
    foreground = np.array([item[3] > 0 for item in dataset.items])
    n_fg, n_bg = int(foreground.sum()), int((~foreground).sum())
    fraction = p["foreground_sample_fraction"]
    weights = np.where(foreground, fraction/max(n_fg, 1), (1-fraction)/max(n_bg, 1))
    sampler_generator = torch.Generator().manual_seed(p["seed"])
    loader_generator = torch.Generator().manual_seed(p["seed"]+1)
    sampler = WeightedRandomSampler(weights, p["samples_per_epoch"] or len(dataset),
                                    replacement=True, generator=sampler_generator)
    loader = DataLoader(dataset, batch_size=p["batch_size"], sampler=sampler,
                        num_workers=p["num_workers"], worker_init_fn=_seed_worker,
                        generator=loader_generator, pin_memory=device == "cuda")
    net = build_unet(p).to(device)
    if warm_weights is not None:
        net.load_state_dict(warm_weights, strict=True)
        del warm_weights
    optimizer = torch.optim.Adam(net.parameters(), lr=p["learning_rate"])
    if saved is not None:
        net.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        log = saved["log"]
        if stats != log["tile_statistics"]:
            raise ValueError("Resume training label/tile statistics differ")
        best, best_epoch = saved["best_val_dice"], saved["best_epoch"]
        without_gain, completed = saved["epochs_without_gain"], saved["epoch"]
        previous_seconds = log["training_seconds"]
        # Repair aliases only from verified committed snapshots. Never replace
        # the historical best with a merely recent or worse model.
        if receipt["best_checkpoint"]:
            best_file = output_dir/"best.pt"
            if not best_file.exists() or file_sha256(best_file) != receipt["best_sha256"]:
                _publish_file(output_dir/receipt["best_checkpoint"], best_file)
        _publish_file(output_dir/receipt["checkpoint"], output_dir/"last.pt")
        _restore_rng(saved["rng_state"], sampler_generator, loader_generator)
        print(f"Resume after epoch {completed}: best={best:.6f} at {best_epoch}, no-gain={without_gain}", flush=True)
    else:
        try:
            # Never accidentally discover an unrelated parent Git repository.
            revision = (subprocess.run(["git", "rev-parse", "HEAD"], cwd=config.ROOT,
                                      capture_output=True, text=True, check=False).stdout.strip()
                        if (config.ROOT/".git").exists() else "")
        except OSError:  # A standalone Windows deployment does not require Git.
            revision = ""
        log = {"run_kind": run_kind, "params": p, "train_ids": train_ids, "val_ids": val_ids,
               "split_sha256": file_sha256(config.SPLIT_FILE), "git_revision": revision,
               "source_sha256": file_sha256(Path(__file__)), "source_fingerprints": _source_fingerprint(),
               "environment": environment_info(),
               "python": platform.python_version(), "torch": str(torch.__version__),
               "hardware": runtime["gpu"] or platform.processor(), "device": device,
               "gpu_memory_bytes": runtime["gpu_memory_bytes"],
               "platform_system": platform.system(),
               "initialization": warm_provenance or {"kind": "random_seeded", "seed": p["seed"]},
               "model_parameters": sum(v.numel() for v in net.parameters()),
               "tile_statistics": stats, "raw_pos_weight": raw_weight, "pos_weight": pos_weight,
               "sampling": {"replacement": True, "requested_foreground_fraction": fraction,
                            "expected_foreground_fraction": fraction if n_fg and n_bg else float(n_fg > 0),
                            "draws_per_epoch": len(sampler)},
               "monitor": "micro foreground Dice on ALL reconstructed validation images at threshold 0.5; every epoch",
               "cpu_patience_limit": device == "cpu" and p["max_epochs"] <= p["patience"],
               "history": [], "best_epoch": None, "best_val_dice": -1., "status": "initialized",
               "training_seconds": 0., "resume_boundary": "completed epoch only",
               "checkpoint_storage": "immutable snapshots + checksummed commit receipts; Drive remote durability is not guaranteed"}
        best, best_epoch, without_gain, completed, previous_seconds = -1., None, 0, 0, 0.
        _write_json(output_dir/"tile_provenance.json", {"columns": ["image_id", "row", "col", "foreground_pixels"],
                                                      "items": dataset.items})

    def payload(epoch, dice):
        return {"format_version": CHECKPOINT_VERSION, "params": p, "model_state": net.state_dict(),
                "optimizer_state": optimizer.state_dict(), "epoch": epoch, "val_dice": dice,
                "run_kind": run_kind, "best_val_dice": best, "best_epoch": best_epoch,
                "epochs_without_gain": without_gain, "history": log["history"], "log": log,
                "rng_state": _capture_rng(sampler_generator, loader_generator)}

    if saved is None:
        receipt = _save_epoch(output_dir, payload(0, -1.), None, improved=False)
    log["checkpoint_sha256"] = receipt["best_sha256"]
    _write_json(output_dir/"training.json", log)
    if completed >= p["max_epochs"] or without_gain >= p["patience"]:
        print("This run already reached its epoch limit or early stopping; no epochs rerun", flush=True)
        return log
    if stop_after_epoch is not None and stop_after_epoch <= completed:
        raise ValueError("Graceful pause epoch must be later than the completed epoch")
    log["status"] = "running"
    try:
        for epoch in range(completed+1, p["max_epochs"]+1):
            net.train()
            loss_sum = examples = 0
            load_seconds = compute_seconds = 0.
            last_time = time.perf_counter()
            for batch_index, (x, y, valid) in enumerate(loader, 1):
                load_seconds += time.perf_counter()-last_time
                step_start = time.perf_counter()
                x, y, valid = [v.to(device) for v in (x, y, valid)]
                optimizer.zero_grad(set_to_none=True)
                loss = weighted_bce_dice(net(x), y, valid, pos_weight, p["bce_weight"])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                loss.backward()
                optimizer.step()
                value = float(loss.detach())  # CUDA synchronization for honest progress/timing
                compute_seconds += time.perf_counter()-step_start
                loss_sum += value*len(x)
                examples += len(x)
                if batch_index == 1 or batch_index % 50 == 0 or batch_index == len(loader):
                    progress = {"phase": "training", "epoch": epoch, "completed_epoch": epoch-1,
                                "batch": batch_index, "batches": len(loader), "loss_so_far": loss_sum/examples}
                    _write_json(output_dir/"progress.json", progress)
                    print(f"Epoch {epoch}/{p['max_epochs']} batch {batch_index}/{len(loader)} loss={loss_sum/examples:.5f}", flush=True)
                last_time = time.perf_counter()
            counts = np.zeros(3, np.int64)
            val_start = time.perf_counter()
            for index, image_id in enumerate(val_ids, 1):
                image, ref = data.load_pair(image_id)
                pred, _ = segment_c(image, net, {**p, "threshold": 0.5})
                counts += _strict_counts(pred, ref)
                if index == 1 or index % 5 == 0 or index == len(val_ids):
                    _write_json(output_dir/"progress.json", {"phase": "validation", "epoch": epoch,
                        "completed_epoch": epoch-1, "images": index, "total_images": len(val_ids)})
                    print(f"Epoch {epoch} full-image validation {index}/{len(val_ids)}", flush=True)
            scores = _scores(counts)
            dice = scores["f1"]
            elapsed = previous_seconds+time.perf_counter()-start
            log["history"].append({"epoch": epoch, "training_loss": loss_sum/examples,
                "validation_dice": dice, "validation_precision": scores["precision"],
                "validation_recall": scores["recall"], "elapsed_seconds": elapsed,
                "data_loading_seconds": load_seconds, "training_compute_seconds": compute_seconds,
                "validation_seconds": time.perf_counter()-val_start})
            improved = dice > best
            if improved:
                best, best_epoch, without_gain = dice, epoch, 0
            else:
                without_gain += 1
            finished = without_gain >= p["patience"] or epoch == p["max_epochs"]
            paused = stop_after_epoch is not None and epoch >= stop_after_epoch and not finished
            log.update(training_seconds=elapsed, best_epoch=best_epoch, best_val_dice=best,
                       status="completed" if finished else "paused_at_epoch" if paused else "running",
                       stopping_reason="patience" if without_gain >= p["patience"] else
                           "epoch_limit" if finished else "explicit_epoch_pause" if paused else None)
            epoch_state = payload(epoch, dice)
            receipt = _save_epoch(output_dir, epoch_state, receipt, improved)
            try:
                _save_training_artifacts(output_dir, log)
            finally:
                _restore_rng(epoch_state["rng_state"], sampler_generator, loader_generator)
            log["checkpoint_sha256"] = receipt["best_sha256"]
            _write_json(output_dir/"training.json", log)
            _write_json(output_dir/"progress.json", {"phase": log["status"], "completed_epoch": epoch,
                "best_epoch": best_epoch, "best_val_dice": best})
            print(f"Committed epoch {epoch}: loss={loss_sum/examples:.6f}, full-image val Dice={dice:.6f}", flush=True)
            if finished or paused:
                break
    except (Exception, KeyboardInterrupt) as error:
        # Never promote a partial epoch. A failed advisory write must not mask
        # the original exception or invalidate the last committed checkpoint.
        try:
            _write_json(output_dir/"interruption.json", {"message": str(error),
                "last_committed_epoch": receipt["epoch"], "resume": "training/last.pt; incomplete epoch will be replayed"})
        except OSError:
            pass
        raise
    return log


def benchmark_training(train_ids: list[str], val_ids: list[str], params: dict,
                       run_kind: str = "formal", batches: int = 3) -> dict:
    """Independent throwaway-model benchmark, never a training/validation result.

    One full validation photograph estimates 69-image cost by actual tile count.
    Full training still validates EVERY photograph EVERY epoch. FP32 algorithm
    and batch size stay unchanged. No automatic AMP, validation reduction or OOM
    fallback. The estimate excludes checkpoint upload and setup overhead.
    """
    import torch
    from torch.utils.data import DataLoader
    runtime = check_run_mode(run_kind)
    p = _params(params)
    _check_ids(train_ids, "train")
    _check_ids(val_ids, "val")
    if batches < 1:
        raise ValueError("batches must be positive")
    sampler_generator = torch.Generator().manual_seed(p["seed"])
    loader_generator = torch.Generator().manual_seed(p["seed"]+1)
    original_rng = _capture_rng(sampler_generator, loader_generator)
    device = runtime["device"]
    def sync():
        if device == "cuda":
            torch.cuda.synchronize()
    def tile_count(ids):
        from PIL import Image
        total = 0
        for image_id in ids:
            with Image.open(data._index(str(config.IMAGE_DIR))[image_id]) as im:
                w, h = im.size[::-1] if im.getexif().get(274, 1) in (5, 6, 7, 8) else im.size
            total += len(_origins((h, w), p))
        return total
    net = optimizer = loader = None
    try:
        start = time.perf_counter()
        dataset = TileDataset(train_ids[:min(2, len(train_ids))], p, train=True)
        setup_seconds = time.perf_counter()-start
        # Shuffle a small benchmark subset so repeated JPEG decoding is visible.
        loader = DataLoader(dataset, batch_size=p["batch_size"], shuffle=True,
                            num_workers=p["num_workers"], generator=loader_generator,
                            worker_init_fn=_seed_worker, pin_memory=device == "cuda")
        net = build_unet(p).to(device).train()
        optimizer = torch.optim.Adam(net.parameters(), lr=p["learning_rate"])
        iterator = iter(loader)
        loads, steps = [], []
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for index in range(batches+1):  # exclude one warm-up batch
            tick = time.perf_counter()
            try:
                x, y, valid = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                x, y, valid = next(iterator)
            loading = time.perf_counter()-tick
            sync()
            tick = time.perf_counter()
            x, y, valid = [value.to(device) for value in (x, y, valid)]
            optimizer.zero_grad(set_to_none=True)
            loss = weighted_bce_dice(net(x), y, valid, 1., p["bce_weight"])
            loss.backward()
            optimizer.step()
            sync()
            if index:
                loads.append(loading)
                steps.append(time.perf_counter()-tick)
        tick = time.perf_counter()
        image, _ = data.load_pair(val_ids[0])
        segment_c(image, net, p)
        sync()
        validation_seconds = time.perf_counter()-tick
        n_train_tiles, n_val_tiles = tile_count(train_ids), tile_count(val_ids)
        sample_tiles = len(_origins(image.shape[:2], p))
        n_batches = int(np.ceil((p["samples_per_epoch"] or n_train_tiles)/p["batch_size"]))
        estimate_train = n_batches*(float(np.mean(loads))+float(np.mean(steps)))
        estimate_val = validation_seconds*n_val_tiles/sample_tiles
        return {"estimate_only": True, "run_kind": run_kind, "device": device,
                "measured_batches": batches, "warmup_batches": 1,
                "data_loading_seconds_per_batch": float(np.mean(loads)),
                "training_compute_seconds_per_batch": float(np.mean(steps)),
                "benchmark_dataset_setup_seconds": setup_seconds,
                "validation_sample_id": val_ids[0], "validation_sample_full_image_seconds": validation_seconds,
                "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
                "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved() if device == "cuda" else None,
                "train_images": len(train_ids), "validation_images_in_real_training": len(val_ids),
                "candidate_train_tiles": n_train_tiles, "validation_tiles": n_val_tiles,
                "estimated_training_seconds_per_epoch": estimate_train,
                "estimated_validation_seconds_per_epoch": estimate_val,
                "estimated_epoch_seconds_excluding_setup_and_upload": estimate_train+estimate_val,
                "bottleneck_hint": "validation" if estimate_val > estimate_train else
                    "JPEG decoding/loading" if np.mean(loads) > np.mean(steps) else "training compute",
                "limitations": "Preliminary extrapolation from 2 training images and 1 FULL validation image, not actual epoch time; cache/size/content/Drive variation excluded"}
    finally:
        del net, optimizer, loader
        _restore_rng(original_rng, sampler_generator, loader_generator)
        if device == "cuda":
            torch.cuda.empty_cache()


def select_threshold(net, ids: list[str], params: dict, output_dir: Path,
                     thresholds=(0.3, 0.4, 0.5, 0.6, 0.7)) -> tuple[dict, list[dict]]:
    """Validation-only micro Dice selection, recall tie-break, lossless map cache.

    Returns proposed params. Publish C.json only after complete formal validation.
    """
    _check_ids(ids, "val")
    thresholds = tuple(float(t) for t in thresholds)
    if not thresholds or any(not 0 <= t <= 1 for t in thresholds):
        raise ValueError("Provide nonempty thresholds in [0,1]")
    p, output_dir = _params(params), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {t: np.zeros(3, np.int64) for t in thresholds}
    timings = {}
    for image_id in ids:
        image, ref = data.load_pair(image_id)
        start = time.perf_counter()
        _, prob = segment_c(image, net, p)
        timings[image_id] = time.perf_counter()-start
        np.save(output_dir/f"{image_id}_prob.npy", prob)
        for t in thresholds:
            counts[t] += _strict_counts(prob >= t, ref)
    rows = [{"threshold": t, **_scores(counts[t])} for t in thresholds]
    best = max(rows, key=lambda row: (row["f1"], row["recall"], -abs(row["threshold"]-0.5)))
    chosen = {**p, "threshold": best["threshold"], "selection_source": "fixed validation split only",
              "selection_ids": list(ids), "selection_metric": "micro full-image foreground Dice; recall tie-break",
              "validation_scores": best, "selection_complete": set(ids) == set(data.load_split()["val"])}
    _write_json(output_dir/"threshold_search.json", {"params": chosen, "candidates": rows,
                                                  "inference_seconds": timings})
    return chosen, rows
