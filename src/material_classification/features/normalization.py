"""Chuẩn hóa L2 cho các vector đặc trưng."""

from __future__ import annotations

import numpy as np


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Chuẩn hóa L2 theo từng mẫu (theo hàng), vector toàn 0 giữ nguyên.

    Tham số:
        x: Mảng numpy 1 chiều (D,) hoặc 2 chiều (N, D).
        eps: Ngưỡng nhỏ để xác định vector 0, tránh chia cho 0.

    Trả về:
        Mảng float32 cùng shape với x, trong đó mỗi vector có chuẩn L2 bằng 1
        (nếu vector ban đầu khác 0) hoặc bằng 0 (nếu vector ban đầu là vector 0).
    """
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        norm = float(np.linalg.norm(arr))
        if norm <= eps:
            return np.zeros_like(arr)
        return arr / norm
    elif arr.ndim == 2:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        mask = norms > eps
        out = np.zeros_like(arr)
        np.divide(arr, norms, out=out, where=mask)
        return out
    else:
        raise ValueError(f"Chỉ hỗ trợ mảng 1 chiều hoặc 2 chiều, nhận shape: {arr.shape}")
