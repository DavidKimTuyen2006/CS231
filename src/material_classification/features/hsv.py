"""Histogram HSV: chuẩn hóa L1 từng kênh rồi nối theo thứ tự H, S, V."""

import numpy as np

from material_classification.data.transforms import (
    ensure_rgb_uint8,
    resize_rgb,
    to_hsv,
)


def extract_hsv(rgb: np.ndarray, hsv_cfg: dict) -> np.ndarray:
    """Nhận RGB uint8, trả vector float32 có số chiều bằng tổng số bins."""
    # Kiểm tra đầu vào và resize từ ảnh RGB gốc.
    rgb = ensure_rgb_uint8(rgb)
    resized = resize_rgb(rgb, hsv_cfg["image_size"])

    # Chuyển RGB sang HSV.
    hsv = to_hsv(resized)

    # Miền giá trị của H, S, V trong OpenCV uint8.
    upper_bounds = (180, 256, 256)
    parts = []

    for channel, (bins, upper) in enumerate(zip(hsv_cfg["bins"], upper_bounds, strict=True)):
        # Đếm số pixel trong từng bin.
        hist, _ = np.histogram(
            hsv[..., channel],
            bins=bins,
            range=(0, upper),
        )

        # Chuẩn hóa L1 riêng từng kênh.
        hist = hist.astype(np.float64)
        hist /= hist.sum()
        parts.append(hist)

    # Nối histogram H, S, V.
    return np.concatenate(parts).astype(np.float32)
