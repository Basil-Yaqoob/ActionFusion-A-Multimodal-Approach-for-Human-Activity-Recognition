from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold


@dataclass(frozen=True)
class SplitResult:
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_subjects: Tuple[int, ...]
    test_subjects: Tuple[int, ...]


def get_subject_split(
    subjects: np.ndarray,
    test_subjects: Sequence[int] = (1, 3, 5, 7),
) -> SplitResult:
    """Create deterministic subject-based train/test split.

    This follows the common UTD-MHAD cross-subject setup where odd subjects
    are test and even subjects are train by default.
    """
    if subjects.ndim != 1:
        raise ValueError(f"subjects must be 1D, got shape {subjects.shape}")

    test_subject_set = tuple(sorted(set(int(s) for s in test_subjects)))
    if not test_subject_set:
        raise ValueError("test_subjects cannot be empty")

    test_mask = np.isin(subjects, test_subject_set)
    test_idx = np.where(test_mask)[0]
    train_idx = np.where(~test_mask)[0]

    if train_idx.size == 0 or test_idx.size == 0:
        raise ValueError(
            f"Invalid subject split. train={train_idx.size}, test={test_idx.size}. "
            "Check subject IDs and provided test_subjects."
        )

    observed_train = tuple(sorted(set(int(x) for x in subjects[train_idx].tolist())))
    observed_test = tuple(sorted(set(int(x) for x in subjects[test_idx].tolist())))

    overlap = set(observed_train).intersection(observed_test)
    if overlap:
        raise RuntimeError(f"Leakage detected: subjects in both train and test: {sorted(overlap)}")

    return SplitResult(
        train_idx=train_idx,
        test_idx=test_idx,
        train_subjects=observed_train,
        test_subjects=observed_test,
    )


def get_stratified_kfold_splits(
    labels: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return deterministic stratified K-fold splits on labels."""
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D, got shape {labels.shape}")
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    dummy_x = np.zeros((labels.shape[0], 1), dtype=np.float32)
    return list(splitter.split(dummy_x, labels))


def summarize_split(
    labels: np.ndarray,
    subjects: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Dict[str, object]:
    """Collect split diagnostics used to validate Step 4 and detect leakage."""
    train_subjects = sorted(set(int(x) for x in subjects[train_idx].tolist()))
    test_subjects = sorted(set(int(x) for x in subjects[test_idx].tolist()))
    subject_overlap = sorted(set(train_subjects).intersection(test_subjects))

    train_classes = sorted(set(int(x) for x in labels[train_idx].tolist()))
    test_classes = sorted(set(int(x) for x in labels[test_idx].tolist()))
    class_overlap = sorted(set(train_classes).intersection(test_classes))

    return {
        "train_samples": int(train_idx.shape[0]),
        "test_samples": int(test_idx.shape[0]),
        "train_subjects": train_subjects,
        "test_subjects": test_subjects,
        "subject_overlap": subject_overlap,
        "train_classes": train_classes,
        "test_classes": test_classes,
        "class_overlap_count": len(class_overlap),
    }
