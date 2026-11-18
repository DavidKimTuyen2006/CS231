import copy
import json

import numpy as np
import pytest
from datasets import ClassLabel, Dataset, DatasetDict, Features
from datasets import Image as HFImage
from PIL import Image

from material_classification.data.cache import (
    build_cache,
    cache_signature,
    finalize_cache,
    load_cache,
    split_summary,
)
from material_classification.data.dataset import check_dataset_schema, load_minc_dataset
from material_classification.utils import load_default_config


def _cfg():
    return load_default_config()


def _toy_dataset(class_names, rows=3):
    features = Features({"image": HFImage(), "label": ClassLabel(names=class_names)})

    def split(n_rows):
        images = [Image.new("RGB", (8, 8), color=(i * 10, 0, 0)) for i in range(n_rows)]
        labels = [i % len(class_names) for i in range(n_rows)]
        return Dataset.from_dict({"image": images, "label": labels}, features=features)

    return DatasetDict({name: split(rows) for name in ("train", "validation", "test")})


def _fake_extract(rgb, features_cfg):
    """Hàm trích xuất giả: đủ để kiểm tra luồng ghi/đọc cache, không phải đặc trưng thật."""
    value = float(rgb[..., 0].mean()) / 255.0
    return {"hsv": np.full(4, value, dtype=np.float32), "hog64": np.full(6, 1.0, dtype=np.float32)}


def test_missing_dataset_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_dataset"):
        load_minc_dataset(tmp_path / "absent")


def test_schema_accepts_matching_dataset():
    cfg = _cfg()
    check_dataset_schema(_toy_dataset(cfg["dataset"]["class_names"]), cfg)


def test_schema_rejects_reordered_labels():
    cfg = _cfg()
    shuffled = list(reversed(cfg["dataset"]["class_names"]))
    with pytest.raises(ValueError, match="Thứ tự nhãn"):
        check_dataset_schema(_toy_dataset(shuffled), cfg)


def test_cache_signature_changes_with_revision_and_features():
    base = _cfg()
    other_rev = copy.deepcopy(base)
    other_rev["dataset"]["revision"] = "0" * 40
    other_bins = copy.deepcopy(base)
    other_bins["features"]["hsv"]["bins"] = [8, 8, 8]
    assert cache_signature(base) != cache_signature(other_rev)
    assert cache_signature(base) != cache_signature(other_bins)
    assert cache_signature(base) == cache_signature(copy.deepcopy(base))


def _write_fake_cache(out_dir, cfg, n=2, *, tamper=False):
    names = cfg["dataset"]["class_names"]
    summaries, base_keys = {}, ["hsv", "hog64", "hog128"]
    dims = {"hsv": 32, "hog64": 1764, "hog128": 8100}
    rng = np.random.default_rng(0)
    for split in cfg["dataset"]["split_names"]:
        d = out_dir / split
        d.mkdir(parents=True, exist_ok=True)
        labels = np.arange(n, dtype=np.int64) % len(names)
        ids = [f"{split}:{i}" for i in range(n)]
        for key in base_keys:
            np.save(d / f"{key}.npy", rng.random((n, dims[key]), dtype=np.float32))
        np.save(d / "labels.npy", labels)
        (d / "sample_ids.json").write_text(json.dumps(ids), encoding="utf-8")
        summaries[split] = split_summary(split, labels, ids)
    if tamper:
        np.save(out_dir / "train" / "labels.npy", np.zeros(n, dtype=np.int64))
    finalize_cache(out_dir, cfg, summaries, base_keys)


def test_load_cache_roundtrip(tmp_path):
    cfg = _cfg()
    _write_fake_cache(tmp_path, cfg)
    data = load_cache(cfg, tmp_path)
    assert set(data) == {"train", "validation", "test"}
    assert data["train"]["hog128"].shape == (2, 8100)


def test_load_cache_rejects_tampered_labels(tmp_path):
    cfg = _cfg()
    _write_fake_cache(tmp_path, cfg, tamper=True)
    with pytest.raises(ValueError, match="lệch"):
        load_cache(cfg, tmp_path)


def test_load_cache_rejects_incomplete_cache(tmp_path):
    cfg = _cfg()
    with pytest.raises(FileNotFoundError):
        load_cache(cfg, tmp_path)


def test_load_cache_rejects_other_config(tmp_path):
    cfg = _cfg()
    _write_fake_cache(tmp_path, cfg)
    other = copy.deepcopy(cfg)
    other["features"]["hsv"]["bins"] = [8, 8, 8]
    with pytest.raises(ValueError, match="không khớp"):
        load_cache(other, tmp_path)


def test_build_cache_with_injected_extractor(tmp_path):
    cfg = _cfg()
    names = cfg["dataset"]["class_names"]
    ds = _toy_dataset(names, rows=2)

    out = build_cache(ds, cfg, tmp_path / "cache", _fake_extract)

    data = load_cache(cfg, out)
    assert data["train"]["hsv"].shape == (2, 4)
    assert data["train"]["hog64"].shape == (2, 6)
    assert data["test"]["labels"].tolist() == [0, 1]
    assert data["train"]["sample_ids"] == ["train:0", "train:1"]
