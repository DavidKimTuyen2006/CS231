"""Trích xuất đặc trưng cho ba split rồi ghi feature cache.

Hàm `extract_base_vectors` do nhóm đặc trưng cung cấp; thiếu thì script dừng và báo rõ.
Cache đã hợp lệ thì tái sử dụng; chỉ trích xuất lại khi dữ liệu hoặc cấu hình đặc trưng đổi.
Chạy: uv run python scripts/extract_features.py [--force]
"""

from __future__ import annotations

import argparse
import sys

from material_classification.data.cache import (
    build_cache,
    cache_dir,
    clear_cache_dir,
    load_cache,
)
from material_classification.data.dataset import check_dataset_schema, load_minc_dataset
from material_classification.utils import load_default_config


def _load_extractor():
    try:
        from material_classification.features.extractor import extract_base_vectors
    except ImportError:
        sys.exit("Chưa có material_classification.features.extractor (nhóm đặc trưng cung cấp).")
    return extract_base_vectors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Xóa cache cũ và trích xuất lại")
    args = parser.parse_args()

    cfg = load_default_config()
    out = cache_dir(cfg)
    if out.exists() and not args.force:
        try:
            load_cache(cfg, out)
            print(f"Cache hợp lệ, tái sử dụng: {out}")
            return
        except (FileNotFoundError, ValueError) as exc:
            sys.exit(f"{exc}\nDùng --force để trích xuất lại.")
    if args.force:
        clear_cache_dir(out)

    extract_fn = _load_extractor()
    ds = load_minc_dataset()
    check_dataset_schema(ds, cfg)
    print(f"Trích xuất đặc trưng vào {out}")
    build_cache(ds, cfg, out, extract_fn)
    print("Hoàn tất.")


if __name__ == "__main__":
    main()
