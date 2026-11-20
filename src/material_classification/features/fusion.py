"""Các hàm kết hợp đặc trưng đa tỉ lệ và fusion giữa HOG và HSV."""

from __future__ import annotations

import numpy as np


def combine_multiscale(hog64: np.ndarray, hog128: np.ndarray) -> np.ndarray:
    """Kết hợp đặc trưng HOG đa tỉ lệ (HOG-64 + HOG-128).

    Kích thước chuẩn:
        - hog64: 1.764 chiều
        - hog128: 8.100 chiều
        - Tổng sau khi ghép: 9.864 chiều

    Tham số:
        hog64: Mảng 1 chiều (1764,) hoặc 2 chiều (N, 1764).
        hog128: Mảng 1 chiều (8100,) hoặc 2 chiều (N, 8100).

    Trả về:
        Mảng float32 kết hợp: (9864,) hoặc (N, 9864).
    """
    arr64 = np.asarray(hog64, dtype=np.float32)
    arr128 = np.asarray(hog128, dtype=np.float32)

    if arr64.ndim != arr128.ndim:
        raise ValueError(
            f"hog64 và hog128 cần cùng số chiều không gian (ndim): {arr64.ndim} ≠ {arr128.ndim}"
        )

    last_dim_64 = arr64.shape[-1]
    last_dim_128 = arr128.shape[-1]
    if last_dim_64 != 1764 or last_dim_128 != 8100:
        raise ValueError(
            f"Kích thước vector không đúng chuẩn: hog64 cần 1764 (nhận {last_dim_64}), "
            f"hog128 cần 8100 (nhận {last_dim_128})"
        )

    return np.concatenate([arr64, arr128], axis=-1)


def fuse_features(hog_vec: np.ndarray, hsv_vec: np.ndarray, alpha: float) -> np.ndarray:
    """Kết hợp đặc trưng HOG và HSV theo trọng số căn bậc hai của alpha.

    Công thức:
        x_fusion = [sqrt(alpha) * HOG, sqrt(1 - alpha) * HSV]
        Trong đó HOG luôn đứng trước, HSV luôn đứng sau.

    Kích thước chuẩn:
        - HOG-128 (8.100) + HSV (32) -> 8.132 chiều (C4)
        - HOG đa tỉ lệ (9.864) + HSV (32) -> 9.896 chiều (C5)

    Tham số:
        hog_vec: Vector HOG đã chuẩn hóa L2, 1D hoặc 2D.
        hsv_vec: Vector HSV đã chuẩn hóa L2, 1D hoặc 2D.
        alpha: Hệ số trọng số cho HOG, bắt buộc thuộc đoạn [0.0, 1.0].

    Trả về:
        Mảng float32 sau khi ghép.
    """
    if not (0.0 <= float(alpha) <= 1.0):
        raise ValueError(f"alpha phải nằm trong khoảng [0, 1], nhận giá trị: {alpha}")

    hog_arr = np.asarray(hog_vec, dtype=np.float32)
    hsv_arr = np.asarray(hsv_vec, dtype=np.float32)

    if hog_arr.ndim != hsv_arr.ndim:
        raise ValueError(
            f"hog_vec và hsv_vec cần cùng số chiều (ndim): {hog_arr.ndim} ≠ {hsv_arr.ndim}"
        )

    hsv_dim = hsv_arr.shape[-1]
    if hsv_dim != 32:
        raise ValueError(f"hsv_vec cần có 32 chiều, nhận: {hsv_dim}")

    hog_dim = hog_arr.shape[-1]
    if hog_dim not in (1764, 8100, 9864):
        raise ValueError(f"hog_vec cần có 1764, 8100 hoặc 9864 chiều, nhận: {hog_dim}")

    w_hog = np.float32(np.sqrt(alpha))
    w_hsv = np.float32(np.sqrt(1.0 - alpha))

    weighted_hog = w_hog * hog_arr
    weighted_hsv = w_hsv * hsv_arr

    return np.concatenate([weighted_hog, weighted_hsv], axis=-1)
