from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

logger = logging.getLogger(__name__)


def run_baselines(
    dataset_name: str = "iris",
    test_split: float = 0.2,
    random_state: int = 42,
    output_dir: str | None = None,
) -> Dict:
    supported = {"iris": load_iris, "breast_cancer": load_breast_cancer}
    if dataset_name not in supported:
        raise ValueError(f"Unknown dataset '{dataset_name}'")

    data = supported[dataset_name]()
    X_raw = data.data.astype(np.float64)
    y = data.target.astype(np.int64)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_split, random_state=random_state)
    train_idx, test_idx = next(sss.split(X_scaled, y))
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    results = {"dataset": dataset_name, "n_features": X_raw.shape[1], "n_classes": len(np.unique(y)), "baselines": {}}

    for name, clf in [("RandomForest", RandomForestClassifier(n_estimators=100, random_state=random_state)), ("SVM", SVC(kernel="rbf", random_state=random_state))]:
        start = time.perf_counter()
        clf.fit(X_train, y_train)
        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)
        elapsed = time.perf_counter() - start
        results["baselines"][name] = {"train_accuracy": float(train_acc), "test_accuracy": float(test_acc), "training_seconds": float(elapsed)}
        logger.info("%s: train=%.4f test=%.4f (%.2fs)", name, train_acc, test_acc, elapsed)

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        with open(out_path / f"baselines_{dataset_name}.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        logger.info("Baselines saved to %s", out_path / f"baselines_{dataset_name}.json")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for ds in ["iris", "breast_cancer"]:
        run_baselines(ds, output_dir=".")