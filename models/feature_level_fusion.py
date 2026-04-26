from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def compute_class_variance(features: np.ndarray, labels: np.ndarray, n_classes: int) -> np.ndarray:
    if features.ndim != 2:
        raise ValueError(f"features must be 2D, got {features.shape}")
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D, got {labels.shape}")
    if features.shape[0] != labels.shape[0]:
        raise ValueError("features/labels sample count mismatch")

    n_features = features.shape[1]
    class_means = np.zeros((n_classes, n_features), dtype=np.float64)

    for class_idx in range(n_classes):
        mask = labels == class_idx
        if np.any(mask):
            class_means[class_idx] = features[mask].mean(axis=0)

    return class_means.var(axis=0)


def select_features_by_variance_train_only(
    features: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    threshold_percentile: float,
) -> Tuple[np.ndarray, np.ndarray]:
    train_features = features[train_idx]
    train_labels = labels[train_idx]
    n_classes = int(labels.max()) + 1

    variance = compute_class_variance(train_features, train_labels, n_classes=n_classes)
    threshold = np.percentile(variance, threshold_percentile)
    selected_indices = np.where(variance >= threshold)[0]
    if selected_indices.size == 0:
        selected_indices = np.array([int(np.argmax(variance))], dtype=np.int64)

    return features[:, selected_indices], selected_indices


def train_feature_level_fusion(
    features: Mapping[str, np.ndarray],
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    modality_order: Sequence[str] = ("skeleton", "inertial", "depth", "rgb"),
    threshold_percentile: float = 75.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], SVC, StandardScaler]:
    selected_per_modality: Dict[str, np.ndarray] = {}
    selected_counts: Dict[str, int] = {}

    for modality in modality_order:
        if modality not in features:
            raise KeyError(f"Missing modality features: {modality}")
        selected, selected_idx = select_features_by_variance_train_only(
            features=features[modality],
            labels=labels,
            train_idx=train_idx,
            threshold_percentile=threshold_percentile,
        )
        selected_per_modality[modality] = selected
        selected_counts[modality] = int(selected_idx.shape[0])

    fused = np.concatenate([selected_per_modality[m] for m in modality_order], axis=1).astype(np.float32)

    x_train = fused[train_idx]
    x_test = fused[test_idx]
    y_train = labels[train_idx]
    y_test = labels[test_idx]

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    clf = SVC(kernel="rbf", C=10.0, gamma="scale", probability=False)
    clf.fit(x_train_scaled, y_train)
    y_pred = clf.predict(x_test_scaled)

    return y_test, y_pred, selected_counts, clf, scaler
