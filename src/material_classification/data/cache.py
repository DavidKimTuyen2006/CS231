"""Cache đặc trưng theo split, ghi bằng memmap để không giữ toàn bộ ma trận trong RAM.

Hàm trích xuất (extract_fn) do nhóm đặc trưng cung cấp; nhóm dữ liệu chỉ ghi kết quả vào cache.

Cấu trúc:
    cache/minc2500_split1/<signature>/
        metadata.json          (ghi cuối cùng; thiếu file này = cache chưa hoàn chỉnh)
        train/ validation/ test/
            <key>.npy  labels.npy  sample_ids.json
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from datasets import DatasetDict

from material_classification.data.dataset import iter_samples
from material_classification.utils import (
    git_info,
    local_path,
    package_versions,
    read_json,
    sha256_bytes,
    sha256_json,
    utc_now,
    write_json,
)

CACHE_FORMAT_VERSION = 1
RESIZE_METHOD = "cv2.INTER_AREA"

# extract_fn(rgb_uint8, features_cfg) -> {tên_vector: mảng 1 chiều}
ExtractFn = Callable[[np.ndarray, dict[str, Any]], Mapping[str, np.ndarray]]


# Các file mã nguồn quyết định giá trị đặc trưng.
# Đổi nội dung file nào thì cache phải trích xuất lại.
_FEATURE_CODE_FILES = (
    "data/transforms.py",
    "features/extractor.py",
    "features/hog.py",
    "features/hsv.py",
)


def feature_code_sha() -> str:
    """Băm nội dung mã nguồn đặc trưng; cùng config nhưng khác thuật toán vẫn ra chữ ký khác."""
    package_dir = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for rel in _FEATURE_CODE_FILES:
        digest.update(rel.encode("utf-8"))
        digest.update((package_dir / rel).read_bytes())
    return digest.hexdigest()


def cache_signature(cfg: dict[str, Any]) -> str:
    """Băm mọi thứ ảnh hưởng đến giá trị đặc trưng đã trích xuất."""
    payload = {
        "format": CACHE_FORMAT_VERSION,
        "dataset": {"id": cfg["dataset"]["id"], "revision": cfg["dataset"]["revision"]},
        "class_names": cfg["dataset"]["class_names"],
        "features": cfg["features"],
        "resize": RESIZE_METHOD,
        "code": feature_code_sha(),
    }
    return sha256_json(payload)[:16]


def cache_dir(cfg: dict[str, Any], cache_root: Path | None = None) -> Path:
    root = Path(cache_root) if cache_root is not None else local_path("CACHE_ROOT", "./cache")
    return root / "minc2500_split1" / cache_signature(cfg)


def split_summary(split: str, labels: np.ndarray, sample_ids: list[str]) -> dict[str, Any]:
    return {
        "count": int(len(labels)),
        "labels_sha256": sha256_bytes(np.asarray(labels, dtype=np.int64).tobytes()),
        "sample_ids_sha256": sha256_bytes(
            json.dumps(sample_ids, ensure_ascii=False).encode("utf-8")
        ),
    }


def build_cache(ds: DatasetDict, cfg: dict[str, Any], out_dir: Path, extract_fn: ExtractFn) -> Path:
    """Duyệt từng mẫu theo thứ tự cố định, ghi kết quả extract_fn vào .npy bằng memmap."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec, features_cfg = cfg["dataset"], cfg["features"]
    summaries: dict[str, Any] = {}
    base_keys: list[str] = []

    for split in spec["split_names"]:
        split_ds = ds[split]
        n = len(split_ds)
        if n == 0:
            raise ValueError(f"Split '{split}' rỗng")
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        labels = np.empty(n, dtype=np.int64)
        sample_ids: list[str] = []
        memmaps: dict[str, np.memmap] | None = None
        for index, rgb, label in iter_samples(split_ds, spec["image_column"], spec["label_column"]):
            vectors = extract_fn(rgb, features_cfg)
            if memmaps is None:
                base_keys = list(vectors)
                memmaps = {
                    key: np.lib.format.open_memmap(
                        split_dir / f"{key}.npy",
                        mode="w+",
                        dtype=np.float32,
                        shape=(n, np.asarray(vector).shape[0]),
                    )
                    for key, vector in vectors.items()
                }
            for key, vector in vectors.items():
                memmaps[key][index] = vector
            labels[index] = label
            sample_ids.append(f"{split}:{index}")
            if (index + 1) % 5000 == 0:
                print(f"  {split}: {index + 1}/{n}")

        assert memmaps is not None
        for array in memmaps.values():
            array.flush()
        del memmaps
        np.save(split_dir / "labels.npy", labels)
        (split_dir / "sample_ids.json").write_text(json.dumps(sample_ids), encoding="utf-8")
        summaries[split] = split_summary(split, labels, sample_ids)
        print(f"Xong split {split}: {n} mẫu")

    finalize_cache(out_dir, cfg, summaries, base_keys)
    return out_dir


def finalize_cache(
    out_dir: Path, cfg: dict[str, Any], summaries: Mapping[str, Any], base_keys: list[str]
) -> None:
    """Ghi metadata.json; file này được ghi sau cùng nên là dấu hiệu cache đã đầy đủ."""
    write_json(
        out_dir / "metadata.json",
        {
            "complete": True,
            "signature": cache_signature(cfg),
            "format": CACHE_FORMAT_VERSION,
            "dataset": {"id": cfg["dataset"]["id"], "revision": cfg["dataset"]["revision"]},
            "class_names": cfg["dataset"]["class_names"],
            "features": cfg["features"],
            "base_keys": base_keys,
            "dtype": "float32",
            "splits": dict(summaries),
            "created_utc": utc_now(),
            "git": git_info(),
            "packages": package_versions(),
        },
    )


def load_cache(cfg: dict[str, Any], out_dir: Path) -> dict[str, dict[str, Any]]:
    """Đọc cache dạng memmap; từ chối cache thiếu file, ghi dở, sai chữ ký hoặc lệch số mẫu."""
    out_dir = Path(out_dir)
    meta_path = out_dir / "metadata.json"
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"Chưa có cache hoàn chỉnh tại {out_dir}. Chạy scripts/extract_features.py"
        )
    meta = read_json(meta_path)
    if not meta.get("complete") or meta.get("signature") != cache_signature(cfg):
        raise ValueError(f"Cache tại {out_dir} không khớp cấu hình hiện tại. Chạy lại với --force.")

    data: dict[str, dict[str, Any]] = {}
    for split in cfg["dataset"]["split_names"]:
        split_dir = out_dir / split
        info = meta["splits"][split]
        labels = np.load(split_dir / "labels.npy")
        sample_ids = json.loads((split_dir / "sample_ids.json").read_text(encoding="utf-8"))
        if split_summary(split, labels, sample_ids) != info:
            raise ValueError(
                f"Split '{split}' của cache bị lệch so với metadata. Chạy lại với --force."
            )
        entry: dict[str, Any] = {"labels": labels, "sample_ids": sample_ids}
        for key in meta["base_keys"]:
            array = np.load(split_dir / f"{key}.npy", mmap_mode="r")
            if array.shape[0] != info["count"]:
                raise ValueError(f"{split}/{key}.npy có {array.shape[0]} dòng, cần {info['count']}")
            entry[key] = array
        data[split] = entry
    return data


def clear_cache_dir(out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
