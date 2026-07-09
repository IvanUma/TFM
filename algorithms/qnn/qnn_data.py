from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

QNN_DATASETS = {
    "iris": {"loader": load_iris, "n_features": 4, "n_qubits": 2, "n_classes": 3},
    "breast_cancer": {"loader": load_breast_cancer, "n_features": 30, "n_qubits": 5, "n_classes": 2},
}


def _amplitude_encoding(features: np.ndarray, n_qubits: int) -> np.ndarray:
    n_features = features.shape[0]
    state = np.zeros(2**n_qubits, dtype=complex)
    state[:n_features] = features.astype(complex)
    norm = np.linalg.norm(state)
    if norm > 1e-12:
        state /= norm
    else:
        state[0] = 1.0
    return state


def load_qnn_data(
    dataset_name: str,
    test_split: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    if dataset_name not in QNN_DATASETS:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Options: {list(QNN_DATASETS.keys())}")

    info = QNN_DATASETS[dataset_name]
    data = info["loader"]()
    X_raw = data.data.astype(np.float64)
    y = data.target.astype(np.int64)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_split, random_state=random_state)
    train_idx, test_idx = next(sss.split(X_scaled, y))
    X_train_raw, X_test_raw = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    split = StratifiedShuffleSplit(n_splits=1, test_size=test_split, random_state=random_state)
    train_idx2, val_idx = next(split.split(X_train_raw, y_train))
    X_train, X_val = X_train_raw[train_idx2], X_train_raw[val_idx]
    y_train2, y_val = y_train[train_idx2], y_train[val_idx]

    n_qubits = info["n_qubits"]
    X_train_enc = np.array([_amplitude_encoding(x, n_qubits) for x in X_train])
    X_val_enc = np.array([_amplitude_encoding(x, n_qubits) for x in X_val])
    X_test_enc = np.array([_amplitude_encoding(x, n_qubits) for x in X_test_raw])

    dataset_info = {
        "name": dataset_name,
        "n_features": info["n_features"],
        "n_qubits": n_qubits,
        "n_classes": info["n_classes"],
        "n_train": X_train.shape[0],
        "n_val": X_val.shape[0],
        "n_test": X_test_enc.shape[0],
    }

    logger.info(
        "Loaded %s: %d train, %d val, %d test, %d features -> %d qubits, %d classes",
        dataset_name, X_train.shape[0], X_val.shape[0], X_test_enc.shape[0],
        info["n_features"], n_qubits, info["n_classes"],
    )

    return X_train_enc, y_train2, X_val_enc, y_val, X_test_enc, y_test, dataset_info