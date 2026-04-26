from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_early_fusion_features(
    features: Mapping[str, np.ndarray],
    modality_order: Sequence[str],
) -> np.ndarray:
    arrays = []
    sample_count = None

    for modality in modality_order:
        if modality not in features:
            raise KeyError(f"Missing modality features: {modality}")
        array = features[modality]
        if array.ndim != 2:
            raise ValueError(f"Feature array for {modality} must be 2D, got {array.shape}")
        if sample_count is None:
            sample_count = array.shape[0]
        elif array.shape[0] != sample_count:
            raise ValueError(
                f"Sample mismatch for {modality}: {array.shape[0]} vs expected {sample_count}"
            )
        arrays.append(array)

    return np.concatenate(arrays, axis=1).astype(np.float32)


def train_data_level_fusion(
    features: Mapping[str, np.ndarray],
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    modality_order: Sequence[str] = ("skeleton", "inertial", "depth", "rgb"),
) -> Tuple[np.ndarray, np.ndarray, Pipeline]:
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D, got {labels.shape}")

    x_fused = build_early_fusion_features(features=features, modality_order=modality_order)
    if x_fused.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Label/sample mismatch: X has {x_fused.shape[0]} rows, labels has {labels.shape[0]}"
        )

    x_train = x_fused[train_idx]
    x_test = x_fused[test_idx]
    y_train = labels[train_idx]
    y_test = labels[test_idx]

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=10.0, gamma="scale", probability=False)),
        ]
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    return y_test, y_pred, model
