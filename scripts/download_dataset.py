"""Tải MINC-2500 Split 1 từ Hugging Face đúng một lần rồi lưu bản Arrow cục bộ.

Chạy: uv run python scripts/download_dataset.py [--force]
"""

from __future__ import annotations

import argparse
import shutil
import sys

from datasets import load_dataset

from material_classification.data.dataset import check_dataset_schema, dataset_root
from material_classification.utils import load_default_config, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Xóa và tải lại bản local đã có")
    args = parser.parse_args()

    cfg = load_default_config()
    spec = cfg["dataset"]
    if spec["revision"].startswith("REPLACE"):
        sys.exit("configs/default.yaml chưa có dataset.revision hợp lệ.")

    target = dataset_root()
    if target.exists() and any(target.iterdir()):
        if not args.force:
            sys.exit(f"Đã có dataset tại {target}. Dùng --force để tải lại.")
        shutil.rmtree(target)

    print(f"Tải {spec['id']} @ {spec['revision'][:10]} ...")
    ds = load_dataset(spec["id"], revision=spec["revision"])
    check_dataset_schema(ds, cfg)

    target.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(target))
    write_json(
        target / "source.json",
        {
            "dataset_id": spec["id"],
            "revision": spec["revision"],
            "splits": {name: len(ds[name]) for name in spec["split_names"]},
        },
    )
    print(f"Đã lưu dataset tại {target}")
    print(ds)


if __name__ == "__main__":
    main()
