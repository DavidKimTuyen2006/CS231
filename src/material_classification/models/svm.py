"""Mô hình phân loại Linear SVM."""

from __future__ import annotations

from typing import Any

from sklearn.svm import LinearSVC


def create_linear_svm(
    C: float = 1.0,
    penalty: str = "l2",
    loss: str = "squared_hinge",
    max_iter: int = 5000,
    tol: float = 1e-4,
    random_state: int | None = 42,
    **kwargs: Any,
) -> LinearSVC:
    """Tạo đối tượng LinearSVC theo đúng tham số quy chuẩn của đồ án.

    Tham số:
        C: Tham số điều chuẩn (regularization parameter).
        penalty: Chuẩn phạt ('l2').
        loss: Hàm mất mát ('squared_hinge').
        max_iter: Số vòng lặp tối đa để hội tụ (mặc định 5000).
        tol: Ngưỡng dừng (tolerance, mặc định 1e-4).
        random_state: Seed ngẫu nhiên để đảm bảo tính tái lập.

    Trả về:
        Instance của sklearn.svm.LinearSVC.
    """
    return LinearSVC(
        C=C,
        penalty=penalty,
        loss=loss,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
        **kwargs,
    )


def create_svm_from_config(classifier_cfg: dict[str, Any], C: float, seed: int = 42) -> LinearSVC:
    """Tạo LinearSVC từ khối classifier của file cấu hình YAML."""
    return create_linear_svm(
        C=C,
        penalty=classifier_cfg.get("penalty", "l2"),
        loss=classifier_cfg.get("loss", "squared_hinge"),
        max_iter=classifier_cfg.get("max_iter", 5000),
        tol=classifier_cfg.get("tol", 1e-4),
        random_state=seed,
    )
