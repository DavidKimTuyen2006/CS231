"""Kiểm tra đặc trưng bằng ảnh tổng hợp, không cần tải MINC-2500."""

import numpy as np
import pytest
from datasets import ClassLabel, Dataset, DatasetDict, Features
from datasets import Image as HFImage
from PIL import Image

from material_classification.data.cache import build_cache, load_cache
from material_classification.features.extractor import extract_base_vectors
from material_classification.features.hog import extract_hog
from material_classification.features.hsv import extract_hsv
from material_classification.utils import load_default_config


@pytest.fixture
def cfg():
    return load_default_config()


@pytest.fixture
def rgb():
    # Ảnh không vuông, có nhiều màu và cạnh để kiểm tra cả hai loại đặc trưng.
    return np.random.default_rng(42).integers(0, 256, size=(93, 157, 3), dtype=np.uint8)


def test_default_vectors_have_expected_shapes_and_dtype(cfg, rgb):
    original = rgb.copy()
    vectors = extract_base_vectors(rgb, cfg["features"])

    assert set(vectors) == {"hsv", "hog64", "hog128"}
    for key, dimension in {"hsv": 32, "hog64": 1764, "hog128": 8100}.items():
        vector = vectors[key]
        assert vector.shape == (dimension,)
        assert vector.dtype == np.float32
        assert np.isfinite(vector).all()
        assert (vector >= 0).all()
    np.testing.assert_array_equal(rgb, original)


@pytest.mark.parametrize(
    ("color", "h_bin", "s_bin", "v_bin"),
    [
        ((255, 0, 0), 0, 7, 7),
        ((0, 255, 0), 5, 7, 7),
        ((0, 0, 255), 10, 7, 7),
        ((0, 0, 0), 0, 0, 0),
        ((255, 255, 255), 0, 0, 7),
    ],
)
def test_hsv_known_colors_use_rgb_order_and_channel_ranges(cfg, color, h_bin, s_bin, v_bin):
    image = np.full((37, 61, 3), color, dtype=np.uint8)
    vector = extract_hsv(image, cfg["features"]["hsv"])

    expected = np.zeros(32, dtype=np.float32)
    expected[h_bin] = 1
    expected[16 + s_bin] = 1
    expected[24 + v_bin] = 1
    np.testing.assert_array_equal(vector, expected)


def test_hsv_counts_color_proportions_before_concatenation(cfg):
    image = np.empty((128, 128, 3), dtype=np.uint8)
    image[:, :64] = (255, 0, 0)
    image[:, 64:] = (0, 255, 0)

    vector = extract_hsv(image, cfg["features"]["hsv"])

    expected = np.zeros(32, dtype=np.float32)
    expected[0] = expected[5] = 0.5
    expected[23] = expected[31] = 1
    np.testing.assert_array_equal(vector, expected)


def test_hsv_normalizes_each_channel_without_global_l2(cfg, rgb):
    vector = extract_hsv(rgb, cfg["features"]["hsv"])

    np.testing.assert_allclose(
        [vector[:16].sum(), vector[16:24].sum(), vector[24:].sum()],
        [1, 1, 1],
        atol=1e-6,
    )
    assert not np.isclose(np.linalg.norm(vector), 1)


@pytest.mark.parametrize("size", [64, 128])
@pytest.mark.parametrize("color", [(0, 0, 0), (127, 127, 127), (255, 0, 0), (255, 255, 255)])
def test_hog_constant_images_are_finite_zero_vectors(cfg, size, color):
    image = np.full((83, 109, 3), color, dtype=np.uint8)
    vector = extract_hog(image, size, cfg["features"]["hog"])

    assert vector.dtype == np.float32
    assert np.isfinite(vector).all()
    assert np.count_nonzero(vector) == 0


@pytest.mark.parametrize(("axis", "direction_bin"), [(1, 0), (0, 4)])
def test_hog_detects_gradient_direction_without_global_l2(cfg, axis, direction_bin):
    # Cường độ tăng theo cột hoặc hàng: gradient có hướng 0 hoặc 90 độ.
    gray = np.tile(np.arange(128, dtype=np.uint8), (128, 1))
    if axis == 0:
        gray = gray.T
    image = np.repeat(gray[..., None], 3, axis=2)

    vector = extract_hog(image, 128, cfg["features"]["hog"])
    cell_histograms = vector.reshape(-1, 9)

    assert np.any(cell_histograms[:, direction_bin] > 0)
    assert np.count_nonzero(np.delete(cell_histograms, direction_bin, axis=1)) == 0
    assert np.linalg.norm(vector) > 1


def test_extractor_uses_configured_bins_scales_and_hog_geometry(cfg, rgb):
    features_cfg = cfg["features"]
    features_cfg["hsv"] = {"image_size": 80, "bins": [8, 4, 4]}
    features_cfg["hog"].update(
        scales=[48],
        orientations=6,
        pixels_per_cell=[8, 8],
        cells_per_block=[3, 3],
        block_norm="L1",
    )

    vectors = extract_base_vectors(rgb, features_cfg)

    assert set(vectors) == {"hsv", "hog48"}
    assert vectors["hsv"].shape == (16,)
    assert vectors["hog48"].shape == (4 * 4 * 3 * 3 * 6,)
    # Với L1, tổng mỗi block không rỗng gần bằng 1.
    blocks = vectors["hog48"].reshape(-1, 3 * 3 * 6)
    np.testing.assert_allclose(blocks.sum(axis=1), 1, atol=1e-5)


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((32, 32), dtype=np.uint8),
        np.zeros((32, 32, 4), dtype=np.uint8),
        np.zeros((32, 32, 3), dtype=np.float32),
    ],
)
def test_extractors_reject_wrong_image_shape_or_dtype(cfg, image):
    with pytest.raises(ValueError):
        extract_hsv(image, cfg["features"]["hsv"])
    with pytest.raises(ValueError):
        extract_hog(image, 64, cfg["features"]["hog"])
    with pytest.raises(ValueError):
        extract_base_vectors(image, cfg["features"])


def test_extractor_accepts_noncontiguous_rgb(cfg, rgb):
    view = rgb[:, ::-1]
    assert not view.flags.c_contiguous

    actual = extract_base_vectors(view, cfg["features"])
    expected = extract_base_vectors(view.copy(), cfg["features"])

    for key in actual:
        np.testing.assert_array_equal(actual[key], expected[key])


def test_real_extractor_roundtrips_through_cache(tmp_path, cfg):
    images = [
        Image.new("RGB", (41, 59), color=(255, 0, 0)),
        Image.new("RGB", (67, 43), color=(0, 255, 0)),
    ]
    schema = Features(
        {"image": HFImage(), "label": ClassLabel(names=cfg["dataset"]["class_names"])}
    )
    ds = DatasetDict(
        {
            split: Dataset.from_dict({"image": images, "label": [0, 1]}, features=schema)
            for split in cfg["dataset"]["split_names"]
        }
    )

    out = build_cache(ds, cfg, tmp_path / "cache", extract_base_vectors)
    cached = load_cache(cfg, out)

    expected_hsv = np.zeros((2, 32), dtype=np.float32)
    expected_hsv[0, 0] = expected_hsv[1, 5] = 1
    expected_hsv[:, 23] = expected_hsv[:, 31] = 1
    for split in cfg["dataset"]["split_names"]:
        entry = cached[split]
        np.testing.assert_array_equal(entry["hsv"], expected_hsv)
        for key, dimension in {"hsv": 32, "hog64": 1764, "hog128": 8100}.items():
            assert entry[key].shape == (2, dimension)
            assert entry[key].dtype == np.float32
        assert np.count_nonzero(entry["hog64"]) == 0
        assert np.count_nonzero(entry["hog128"]) == 0
        assert entry["labels"].tolist() == [0, 1]
        assert entry["sample_ids"] == [f"{split}:0", f"{split}:1"]
