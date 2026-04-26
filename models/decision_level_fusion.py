from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass
class UnimodalModel:
    scaler: StandardScaler
    classifier: SVC


def train_unimodal_classifiers(
    features: Mapping[str, np.ndarray],
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    modality_order: Sequence[str] = ("skeleton", "inertial", "depth", "rgb"),
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, UnimodalModel]]:
    score_train: Dict[str, np.ndarray] = {}
    score_test: Dict[str, np.ndarray] = {}
    unimodal_pred: Dict[str, np.ndarray] = {}
    models: Dict[str, UnimodalModel] = {}

    for modality in modality_order:
        if modality not in features:
            raise KeyError(f"Missing modality features: {modality}")
        x = features[modality]
        if x.ndim != 2:
            raise ValueError(f"Feature array for {modality} must be 2D, got {x.shape}")

        x_train = x[train_idx]
        x_test = x[test_idx]
        y_train = labels[train_idx]

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        clf = SVC(kernel="rbf", C=10.0, gamma="scale", probability=True)
        clf.fit(x_train_scaled, y_train)

        score_train[modality] = clf.predict_proba(x_train_scaled)
        score_test[modality] = clf.predict_proba(x_test_scaled)
        unimodal_pred[modality] = clf.predict(x_test_scaled)
        models[modality] = UnimodalModel(scaler=scaler, classifier=clf)

    return score_train, score_test, unimodal_pred, models


def decision_level_majority_vote(
    score_test: Mapping[str, np.ndarray],
    labels_test: np.ndarray,
    modality_order: Sequence[str] = ("skeleton", "inertial", "depth", "rgb"),
) -> Tuple[np.ndarray, np.ndarray]:
    stacked = np.stack([score_test[m] for m in modality_order], axis=0)
    combined = stacked.sum(axis=0)
    y_pred = combined.argmax(axis=1)
    return labels_test, y_pred


class LateFusionNet(nn.Module):
    def __init__(self, n_modalities: int, n_classes: int) -> None:
        super().__init__()
        k = n_modalities
        n = n_classes
        self.net = nn.Sequential(
            nn.Linear(k * n, 4 * n),
            nn.ReLU(),
            nn.Linear(4 * n, 2 * n),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(2 * n, n),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def decision_level_fusion_learned(
    score_train: Mapping[str, np.ndarray],
    score_test: Mapping[str, np.ndarray],
    labels_train: np.ndarray,
    labels_test: np.ndarray,
    modality_order: Sequence[str] = ("skeleton", "inertial", "depth", "rgb"),
    epochs: int = 120,
    lr: float = 1e-3,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, LateFusionNet]:
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    x_train = np.concatenate([score_train[m] for m in modality_order], axis=1).astype(np.float32)
    x_test = np.concatenate([score_test[m] for m in modality_order], axis=1).astype(np.float32)

    n_classes = int(labels_train.max()) + 1
    model = LateFusionNet(n_modalities=len(modality_order), n_classes=n_classes)

    x_train_t = torch.from_numpy(x_train)
    y_train_t = torch.from_numpy(labels_train.astype(np.int64))
    x_test_t = torch.from_numpy(x_test)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(x_train_t)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits_test = model(x_test_t)
        y_pred = logits_test.argmax(dim=1).cpu().numpy()

    return labels_test, y_pred, model
