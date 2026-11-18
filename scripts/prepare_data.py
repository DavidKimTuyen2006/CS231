"""Kiểm tra dataset cục bộ và ghi manifest.json (không tự tải lại, không tự chia split).

Chạy: uv run python scripts/prepare_data.py
"""

from __future__ import annotations

import sys
from collections import Counter

import numpy as np

from material_classification.data.cache import split_summary
from material_classification.data.dataset import (
    check_dataset_schema,
    dataset_root,
    load_minc_dataset,
)
from material_classification.utils import load_default_config, read_json, write_json


def main() -> None:
    cfg = load_default_config()
    spec = cfg["dataset"]
    root = dataset_root()
    ds = load_minc_dataset(root)
    check_dataset_schema(ds, cfg)

    source_path = root / "source.json"
    if source_path.is_file() and read_json(source_path)["revision"] != spec["revision"]:
        sys.exit(
            "Revision của bản local khác configs/default.yaml. Chạy download_dataset.py --force."
        )

    manifest: dict = {
        "dataset_id": spec["id"],
        "revision": spec["revision"],
        "class_names": spec["class_names"],
        "splits": {},
    }
    for split in spec["split_names"]:
        labels = np.asarray(ds[split][spec["label_column"]], dtype=np.int64)
        expected = spec["expected_counts"][split]
        if len(labels) != expected:
            sys.exit(f"Split '{split}' có {len(labels)} mẫu, cần {expected}.")
        counts = Counter(labels.tolist())
        if len(counts) != len(spec["class_names"]):
            sys.exit(f"Split '{split}' thiếu lớp.")
        ids = [f"{split}:{i}" for i in range(len(labels))]
        summary = split_summary(split, labels, ids)
        summary["label_counts"] = {spec["class_names"][k]: counts[k] for k in sorted(counts)}
        manifest["splits"][split] = summary
        print(f"{split}: {len(labels)} mẫu, {len(counts)} lớp")

    write_json(root / "manifest.json", manifest)
    print(f"Đã ghi {root / 'manifest.json'}")


if __name__ == "__main__":
    main()
