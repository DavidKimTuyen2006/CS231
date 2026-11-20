"""Trích xuất HOG từ ảnh xám ở một kích thước cho trước."""

import numpy as np
from skimage.feature import hog

from material_classification.data.transforms import ensure_rgb_uint8, resize_rgb, to_gray


def extract_hog(rgb: np.ndarray, size: int, hog_cfg: dict) -> np.ndarray:
    """Resize trực tiếp từ RGB gốc, trả HOG float32 một chiều.

    Chỉ chuẩn hóa từng block theo cấu hình; chưa chuẩn hóa L2 toàn vector.
    """
    rgb = ensure_rgb_uint8(rgb)
    gray = to_gray(resize_rgb(rgb, size))

    vector = hog(
        gray,
        orientations=hog_cfg["orientations"],
        pixels_per_cell=tuple(hog_cfg["pixels_per_cell"]),
        cells_per_block=tuple(hog_cfg["cells_per_block"]),
        block_norm=hog_cfg["block_norm"],
        visualize=False,
        transform_sqrt=False,
        feature_vector=True,
        channel_axis=None,
    )
    return np.asarray(vector, dtype=np.float32)
