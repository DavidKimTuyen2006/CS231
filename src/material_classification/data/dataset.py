"""Đọc MINC-2500 từ bản save_to_disk() hoặc cache Arrow Hugging Face cục bộ."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
from PIL import Image

from material_classification.utils import load_default_config, local_path, read_json


def dataset_root() -> Path:
    return local_path("DATASET_ROOT", "./data/minc-2500")


def load_minc_dataset(root: Path | None = None) -> DatasetDict:
    """Load DatasetDict cục bộ. Không gọi Hugging Face Hub."""
    root = Path(root) if root is not None else dataset_root()
    if (root / "dataset_dict.json").is_file():
        return load_from_disk(str(root), keep_in_memory=False)

    # Cache load_dataset() có dataset_info.json và các shard Arrow, không có
    # dataset_dict.json. Chỉ tự tìm cache thuộc revision đã ghim trong config.
    revision = load_default_config()["dataset"]["revision"]
    candidates = (
        [root / "dataset_info.json"]
        if (root / "dataset_info.json").is_file()
        else sorted(root.glob(f"*/*/{revision}/dataset_info.json"))
    )
    if not candidates:
        raise FileNotFoundError(
            f"Không tìm thấy bản save_to_disk() hoặc cache Arrow revision {revision} tại {root}. "
            "Kiểm tra DATASET_ROOT trong .env hoặc chạy: uv run python scripts/download_dataset.py"
        )
    if len(candidates) != 1:
        raise ValueError(
            "Có nhiều cache phù hợp; đặt DATASET_ROOT tới thư mục chứa dataset_info.json."
        )

    info_path = candidates[0]
    info = read_json(info_path)
    result = DatasetDict()
    for split, spec in info["splits"].items():
        prefix = f"{info['dataset_name']}-{split}"
        shard_lengths = spec.get("shard_lengths")
        filenames = (
            [
                f"{prefix}-{i:05d}-of-{len(shard_lengths):05d}.arrow"
                for i in range(len(shard_lengths))
            ]
            if shard_lengths
            else [f"{prefix}.arrow"]
        )
        shards = []
        for filename in filenames:
            path = info_path.parent / filename
            if not path.is_file():
                raise FileNotFoundError(f"Cache Arrow chưa đầy đủ, thiếu: {path}")
            shards.append(Dataset.from_file(str(path), split=split, in_memory=False))
        result[split] = concatenate_datasets(shards) if len(shards) > 1 else shards[0]
        if len(result[split]) != spec["num_examples"]:
            raise ValueError(
                f"Cache Arrow split '{split}' không đủ số mẫu trong dataset_info.json."
            )
    return result


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
