"""Mô hình phân loại K-Nearest Neighbors (KNN)."""

from __future__ import annotations

from typing import Any

from sklearn.neighbors import KNeighborsClassifier


def create_knn(
    n_neighbors: int = 5,
    weights: str = "uniform",
    metric: str = "euclidean",
    n_jobs: int | None = -1,
    **kwargs: Any,
) -> KNeighborsClassifier:
    """Tạo đối tượng KNeighborsClassifier theo quy chuẩn của đồ án.

    Tham số:
        n_neighbors: Số láng giềng k.
        weights: Hàm trọng số ('uniform' hoặc 'distance').
        metric: Khoảng cách đo lường ('euclidean', 'cosine', 'minkowski').
        n_jobs: Số luồng CPU tính toán song song (-1 là dùng toàn bộ lõi).

    Trả về:
        Instance của sklearn.neighbors.KNeighborsClassifier.
    """
    return KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        metric=metric,
        n_jobs=n_jobs,
        **kwargs,
    )


def create_knn_from_config(
    classifier_cfg: dict[str, Any], k: int, n_jobs: int | None = -1
) -> KNeighborsClassifier:
    """Tạo KNN từ khối classifier của file cấu hình YAML."""
    return create_knn(
        n_neighbors=k,
        weights=classifier_cfg.get("weights", "uniform"),
        metric=classifier_cfg.get("metric", "euclidean"),
        n_jobs=n_jobs,
    )
