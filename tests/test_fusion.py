import numpy as np
import pytest

from material_classification.features.fusion import combine_multiscale, fuse_features
from material_classification.features.normalization import l2_normalize


def test_l2_normalize_1d():
    vec = np.array([3.0, 4.0], dtype=np.float32)
    normed = l2_normalize(vec)
    assert np.isclose(np.linalg.norm(normed), 1.0)
    assert np.allclose(normed, np.array([0.6, 0.8], dtype=np.float32))

    # Vector 0 giữ nguyên
    zero_vec = np.zeros(10, dtype=np.float32)
    normed_zero = l2_normalize(zero_vec)
    assert np.allclose(normed_zero, zero_vec)


def test_l2_normalize_2d():
    mat = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, -1.0]], dtype=np.float32)
    normed = l2_normalize(mat)
    assert np.isclose(np.linalg.norm(normed[0]), 1.0)
    assert np.allclose(normed[1], 0.0)
    assert np.isclose(np.linalg.norm(normed[2]), 1.0)


def test_combine_multiscale_1d():
    hog64 = np.ones(1764, dtype=np.float32)
    hog128 = np.full(8100, 2.0, dtype=np.float32)
    combined = combine_multiscale(hog64, hog128)

    assert combined.shape == (9864,)
    assert combined.dtype == np.float32
    assert np.allclose(combined[:1764], 1.0)
    assert np.allclose(combined[1764:], 2.0)


def test_combine_multiscale_2d():
    hog64 = np.ones((5, 1764), dtype=np.float32)
    hog128 = np.full((5, 8100), 2.0, dtype=np.float32)
    combined = combine_multiscale(hog64, hog128)

    assert combined.shape == (5, 9864)
    assert np.allclose(combined[:, :1764], 1.0)
    assert np.allclose(combined[:, 1764:], 2.0)


def test_combine_multiscale_invalid():
    with pytest.raises(ValueError, match="Kích thước vector không đúng chuẩn"):
        combine_multiscale(np.ones(100), np.ones(8100))


def test_fuse_features_dimensions():
    # C4: HOG-128 (8100) + HSV (32) = 8132
    hog128 = np.ones(8100, dtype=np.float32)
    hsv = np.ones(32, dtype=np.float32)
    fused_c4 = fuse_features(hog128, hsv, alpha=0.5)
    assert fused_c4.shape == (8132,)

    # C5: Multiscale HOG (9864) + HSV (32) = 9896
    mhog = np.ones(9864, dtype=np.float32)
    fused_c5 = fuse_features(mhog, hsv, alpha=0.5)
    assert fused_c5.shape == (9896,)

    # 2D batch
    mhog_2d = np.ones((4, 9864), dtype=np.float32)
    hsv_2d = np.ones((4, 32), dtype=np.float32)
    fused_2d = fuse_features(mhog_2d, hsv_2d, alpha=0.5)
    assert fused_2d.shape == (4, 9896)


def test_fuse_features_alpha_values():
    hog128 = l2_normalize(np.ones(8100, dtype=np.float32))
    hsv = l2_normalize(np.ones(32, dtype=np.float32))

    # alpha = 0.0: HOG triệt tiêu, HSV giữ nguyên
    fused_0 = fuse_features(hog128, hsv, alpha=0.0)
    assert np.allclose(fused_0[:8100], 0.0)
    assert np.allclose(fused_0[8100:], hsv)
    assert np.isclose(np.linalg.norm(fused_0), 1.0)

    # alpha = 1.0: HOG giữ nguyên, HSV triệt tiêu
    fused_1 = fuse_features(hog128, hsv, alpha=1.0)
    assert np.allclose(fused_1[:8100], hog128)
    assert np.allclose(fused_1[8100:], 0.0)
    assert np.isclose(np.linalg.norm(fused_1), 1.0)

    # alpha = 0.5: Trọng số căn bậc 2 và bảo toàn chuẩn L2 = 1.0
    fused_half = fuse_features(hog128, hsv, alpha=0.5)
    w = np.sqrt(0.5, dtype=np.float32)
    assert np.allclose(fused_half[:8100], w * hog128)
    assert np.allclose(fused_half[8100:], w * hsv)
    assert np.isclose(np.linalg.norm(fused_half), 1.0, atol=1e-5)


def test_fuse_features_alpha_invalid():
    hog128 = np.ones(8100, dtype=np.float32)
    hsv = np.ones(32, dtype=np.float32)

    with pytest.raises(ValueError, match="alpha phải nằm trong khoảng"):
        fuse_features(hog128, hsv, alpha=-0.1)

    with pytest.raises(ValueError, match="alpha phải nằm trong khoảng"):
        fuse_features(hog128, hsv, alpha=1.1)


def test_fuse_features_order():
    # HOG đứng trước (giá trị 10), HSV đứng sau (giá trị 20)
    hog_vec = np.full(8100, 10.0, dtype=np.float32)
    hsv_vec = np.full(32, 20.0, dtype=np.float32)
    fused = fuse_features(hog_vec, hsv_vec, alpha=1.0)

    assert np.allclose(fused[:8100], 10.0)
    assert np.allclose(fused[8100:], 0.0)
