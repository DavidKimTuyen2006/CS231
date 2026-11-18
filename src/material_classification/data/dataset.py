"""Đọc dataset MINC-2500 đã lưu cục bộ bằng save_to_disk()."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from datasets import DatasetDict, load_from_disk
from PIL import Image

from material_classification.utils import local_path


def dataset_root() -> Path:
    return local_path("DATASET_ROOT", "./data/minc-2500")


def load_minc_dataset(root: Path | None = None) -> DatasetDict:
    """Load DatasetDict cục bộ. Không gọi Hugging Face Hub."""
    root = Path(root) if root is not None else dataset_root()
    if not (root / "dataset_dict.json").is_file():
        raise FileNotFoundError(
            f"Chưa có dataset cục bộ tại {root}. Chạy: uv run python scripts/download_dataset.py"
        )
    return load_from_disk(str(root), keep_in_memory=False)


def check_dataset_schema(ds: DatasetDict, cfg: dict[str, Any]) -> None:
    """Kiểm tra split, cột ảnh/nhãn và thứ tự 23 lớp so với configs/default.yaml."""
    spec = cfg["dataset"]
    if set(ds.keys()) != set(spec["split_names"]):
        raise ValueError(f"Split không khớp: {sorted(ds.keys())} ≠ {spec['split_names']}")
    for split in spec["split_names"]:
        features = ds[split].features
        for column in (spec["image_column"], spec["label_column"]):
            if column not in features:
                raise ValueError(f"Split '{split}' thiếu cột '{column}'")
        names = getattr(features[spec["label_column"]], "names", None)
        if names is None or list(names) != list(spec["class_names"]):
            raise ValueError(
                f"Thứ tự nhãn của split '{split}' khác configs/default.yaml; "
                "không được dùng dataset này cho thí nghiệm."
            )


def decode_rgb(image: Image.Image) -> np.ndarray:
    """Ảnh PIL → mảng RGB uint8 (H, W, 3)."""
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def iter_samples(
    split_ds: Any, image_column: str, label_column: str
) -> Iterator[tuple[int, np.ndarray, int]]:
    """Duyệt theo thứ tự cố định: trả về (index, ảnh RGB, nhãn số)."""
    for index, sample in enumerate(split_ds):
        yield index, decode_rgb(sample[image_column]), int(sample[label_column])
