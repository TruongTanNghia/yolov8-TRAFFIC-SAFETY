"""Compatibility loader for the supplied ECA YOLO checkpoint and inference helpers."""
from __future__ import annotations

import os
import threading
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("HELMET_MODEL", ROOT / "models" / "v8s_ECA_B.pt"))
LABELS = {0: "Motorcyclist", 1: "Plate", 2: "Helmet", 3: "Non_helmet"}
INFERENCE_LOCK = threading.Lock()
MODEL_LOAD_LOCK = threading.Lock()
_MODEL = None


class ECAAttention(nn.Module):
    """Efficient Channel Attention layer serialized by the project's checkpoint."""

    def __init__(self, channels: int | None = None, kernel_size: int = 3, k_size: int | None = None, **_: object):
        super().__init__()
        size = int(k_size or kernel_size)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=size, padding=(size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        weights = self.conv(weights).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(weights)


def load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with MODEL_LOAD_LOCK:
        if _MODEL is not None:
            return _MODEL
        _MODEL = _load_model()
    return _MODEL


def _load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {MODEL_PATH}")
    # The supplied trusted checkpoint pickles this class by its Ultralytics module path.
    import ultralytics.nn.modules.conv as conv_module
    conv_module.ECAAttention = ECAAttention
    from ultralytics import YOLO

    model = YOLO(str(MODEL_PATH))
    model.to("cuda:0" if torch.cuda.is_available() else "cpu")
    model.overrides["verbose"] = False
    return model


def reset_tracker(model) -> None:
    predictor = getattr(model, "predictor", None)
    for tracker in getattr(predictor, "trackers", []):
        tracker.reset()


def infer_track(model, frame: np.ndarray, confidence: float = 0.35):
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=confidence,
        imgsz=960,
        device=0 if torch.cuda.is_available() else "cpu",
        verbose=False,
    )
    return results[0]


def device_info() -> dict:
    return {
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "labels": LABELS,
    }
