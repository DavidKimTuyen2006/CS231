"""Điểm gọi chung để tạo các vector cơ sở cho feature cache."""

import numpy as np

from material_classification.data.transforms import ensure_rgb_uint8
from material_classification.features.hog import extract_hog
from material_classification.features.hsv import extract_hsv


def extract_base_vectors(rgb: np.ndarray, features_cfg: dict) -> dict[str, np.ndarray]:
    """Nhận RGB uint8, trả HSV và HOG ở từng scale theo cấu hình.

    Với default.yaml: hsv (32,), hog64 (1764,), hog128 (8100,).
    Mỗi nhánh resize từ cùng ảnh RGB gốc; chưa ghép hay chuẩn hóa L2 toàn vector.
    """
    rgb = ensure_rgb_uint8(rgb)
    vectors = {"hsv": extract_hsv(rgb, features_cfg["hsv"])}

    hog_cfg = features_cfg["hog"]
    for size in hog_cfg["scales"]:
        vectors[f"hog{size}"] = extract_hog(rgb, size, hog_cfg)

    return vectors
