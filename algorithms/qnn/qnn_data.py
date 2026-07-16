from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from .constants import _BUCKET_EDGES, CIRCUIT_NORM_THRESHOLD

logger = logging.getLogger(__name__)

QNN_DATASETS = {
    "iris": {
        "loader": load_iris,
        "n_features": 4,
        "n_qubits": 4,
        "n_classes": 3,
    },
    "breast_cancer": {
        "loader": load_breast_cancer,
        "n_features": 30,
        "n_qubits": 5,
        "n_classes": 2,
    },
    "wine": {
        "loader": load_wine,
        "n_features": 13,
        "n_qubits": 4,
        "n_classes": 3,
    },
}


def _amplitude_encoding(features: np.ndarray) -> np.ndarray:
    state = features.astype(complex)
    norm = np.linalg.norm(state)
    if norm > CIRCUIT_NORM_THRESHOLD:
        state /= norm
    else:
        state[0] = 1.0
    return state


def _clifford_angle_encoding(
    features: np.ndarray, n_qubits: int, bucket_edges: np.ndarray | None = None
) -> np.ndarray:
    if bucket_edges is not None:
        buckets = np.zeros(n_qubits, dtype=np.int64)
        for i in range(n_qubits):
            buckets[i] = np.digitize(features[i], bins=bucket_edges[i])
        return buckets
    else:
        buckets = np.digitize(features[:n_qubits], bins=_BUCKET_EDGES)
        return buckets.astype(np.int64)


def _pad_to(X: np.ndarray, target: int) -> np.ndarray:
    if X.shape[1] >= target:
        return X[:, :target]
    pad_width = target - X.shape[1]
    return np.pad(X, ((0, 0), (0, pad_width)), mode="constant")


def load_qnn_data(
    dataset_name: str,
    test_split: float = 0.2,
    val_split: float = 0.2,
    random_state: int = 42,
    encoding_mode: str = "amplitude",
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict
]:
    if dataset_name not in QNN_DATASETS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Options: {list(QNN_DATASETS.keys())}"
        )
    if encoding_mode not in ("amplitude", "clifford_angle"):
        raise ValueError(
            f"Unknown encoding_mode '{encoding_mode}'. Options: amplitude, clifford_angle"
        )

    info = QNN_DATASETS[dataset_name]
    data = info["loader"]()
    X_raw = data.data.astype(np.float64)
    y = data.target.astype(np.int64)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=test_split, random_state=random_state
    )
    train_idx, test_idx = next(sss.split(X_scaled, y))
    X_train_raw, X_test_raw = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    split = StratifiedShuffleSplit(
        n_splits=1, test_size=val_split, random_state=random_state
    )
    train_idx2, val_idx = next(split.split(X_train_raw, y_train))
    X_train, X_val = X_train_raw[train_idx2], X_train_raw[val_idx]
    y_train2, y_val = y_train[train_idx2], y_train[val_idx]

    n_qubits = info["n_qubits"]

    if encoding_mode == "amplitude":
        n_components = min(2**n_qubits, X_train.shape[1])
    else:
        n_components = n_qubits

    pca = PCA(n_components=n_components, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test_raw)

    if encoding_mode == "clifford_angle":
        bucket_edges = np.percentile(X_train_pca, [25, 50, 75], axis=0).T

        X_train_enc = np.array(
            [_clifford_angle_encoding(x, n_qubits, bucket_edges) for x in X_train_pca]
        )
        X_val_enc = np.array(
            [_clifford_angle_encoding(x, n_qubits, bucket_edges) for x in X_val_pca]
        )
        X_test_enc = np.array(
            [_clifford_angle_encoding(x, n_qubits, bucket_edges) for x in X_test_pca]
        )
    else:
        target = 2**n_qubits
        X_train_pca = _pad_to(X_train_pca, target)
        X_val_pca = _pad_to(X_val_pca, target)
        X_test_pca = _pad_to(X_test_pca, target)

        X_train_enc = np.array([_amplitude_encoding(x) for x in X_train_pca])
        X_val_enc = np.array([_amplitude_encoding(x) for x in X_val_pca])
        X_test_enc = np.array([_amplitude_encoding(x) for x in X_test_pca])

    dataset_info = {
        "name": dataset_name,
        "n_features": info["n_features"],
        "n_qubits": n_qubits,
        "n_classes": info["n_classes"],
        "n_train": X_train.shape[0],
        "n_val": X_val.shape[0],
        "n_test": X_test_enc.shape[0],
        "encoding_mode": encoding_mode,
    }

    logger.info(
        "Loaded %s: %d train, %d val, %d test, PCA %d -> %d qubits, %d classes (encoding=%s)",
        dataset_name,
        X_train.shape[0],
        X_val.shape[0],
        X_test_enc.shape[0],
        info["n_features"],
        n_qubits,
        info["n_classes"],
        encoding_mode,
    )

    return X_train_enc, y_train2, X_val_enc, y_val, X_test_enc, y_test, dataset_info
