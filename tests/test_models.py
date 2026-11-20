import numpy as np

from material_classification.models import (
    create_knn,
    create_knn_from_config,
    create_linear_svm,
    create_svm_from_config,
)


def test_create_linear_svm():
    clf = create_linear_svm(C=0.1, penalty="l2", loss="squared_hinge", max_iter=1000)
    assert clf.C == 0.1
    assert clf.penalty == "l2"
    assert clf.loss == "squared_hinge"
    assert clf.max_iter == 1000

    # Fit và predict với dữ liệu đồ chơi
    X = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    y = np.array([0, 1, 0, 1])
    clf.fit(X, y)
    preds = clf.predict(X)
    assert len(preds) == 4


def test_create_knn():
    clf = create_knn(n_neighbors=3, weights="uniform", metric="euclidean")
    assert clf.n_neighbors == 3
    assert clf.weights == "uniform"
    assert clf.metric == "euclidean"

    X = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    y = np.array([0, 1, 0, 1])
    clf.fit(X, y)
    preds = clf.predict(X)
    assert len(preds) == 4


def test_models_from_config():
    svm_cfg = {
        "type": "linear_svm",
        "penalty": "l2",
        "loss": "squared_hinge",
        "max_iter": 3000,
        "tol": 1e-4,
    }
    svm = create_svm_from_config(svm_cfg, C=0.5, seed=123)
    assert svm.C == 0.5
    assert svm.max_iter == 3000
    assert svm.random_state == 123

    knn_cfg = {
        "type": "knn",
        "weights": "distance",
        "metric": "cosine",
    }
    knn = create_knn_from_config(knn_cfg, k=7)
    assert knn.n_neighbors == 7
    assert knn.weights == "distance"
    assert knn.metric == "cosine"
