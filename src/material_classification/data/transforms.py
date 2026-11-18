"""Resize trực tiếp từ RGB gốc và chuyển không gian màu (OpenCV, uint8)."""

from __future__ import annotations

import cv2
import numpy as np


def ensure_rgb_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Cần ảnh RGB (H, W, 3), nhận shape {arr.shape}")
    if arr.dtype != np.uint8:
        raise ValueError(f"Cần ảnh uint8, nhận {arr.dtype}")
    return np.ascontiguousarray(arr)


def resize_rgb(image: np.ndarray, size: int) -> np.ndarray:
    """Resize vuông bằng INTER_AREA. Luôn gọi từ ảnh RGB gốc, không resize tuần tự."""
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def to_hsv(rgb: np.ndarray) -> np.ndarray:
    """OpenCV uint8: H ∈ [0, 179], S, V ∈ [0, 255]."""
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)


def to_gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
