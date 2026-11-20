"""Gói features: trích xuất, chuẩn hóa và kết hợp đặc trưng."""

from __future__ import annotations

from material_classification.features.fusion import combine_multiscale, fuse_features
from material_classification.features.normalization import l2_normalize

__all__ = ["l2_normalize", "combine_multiscale", "fuse_features"]
