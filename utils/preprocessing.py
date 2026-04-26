from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import cv2
import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm

from utils.data_loader import get_modality_files, _load_mat_array


PREFIX_META_PATTERN = re.compile(r"^a(\d{2})_s(\d{2})_t(\d{2})$")


@dataclass(frozen=True)
class PreprocessConfig:
    skeleton_target_length: int = 40
    inertial_target_length: int = 40
    depth_img_size: Tuple[int, int] = (64, 64)
    rgb_img_size: Tuple[int, int] = (64, 64)
    rgb_n_frames: int = 16


def _resample_indices(length: int, target_length: int) -> np.ndarray:
    if length <= 0:
        raise ValueError("Cannot resample empty sequence")
    if target_length <= 0:
        raise ValueError("target_length must be positive")
    return np.linspace(0, length - 1, target_length).astype(int)


def _parse_prefix(prefix: str) -> Tuple[int, int, int]:
    match = PREFIX_META_PATTERN.match(prefix)
    if match is None:
        raise ValueError(f"Unexpected prefix format: {prefix}")
    action = int(match.group(1))
    subject = int(match.group(2))
    trial = int(match.group(3))
    return action, subject, trial


def preprocess_skeleton(mat_path: Path, target_length: int = 40) -> np.ndarray:
    skeleton = _load_mat_array(mat_path, ["d_skel", "skeleton", "skel"])

    if skeleton.ndim != 3:
        raise ValueError(f"Skeleton must be 3D, got shape {skeleton.shape}")

    if skeleton.shape[0] != 20 and 20 in skeleton.shape:
        axis_20 = int(np.where(np.array(skeleton.shape) == 20)[0][0])
        skeleton = np.moveaxis(skeleton, axis_20, 0)

    if skeleton.shape[1] != 3 and 3 in skeleton.shape:
        axis_3 = int(np.where(np.array(skeleton.shape) == 3)[0][0])
        skeleton = np.moveaxis(skeleton, axis_3, 1)

    if skeleton.shape[0] != 20 or skeleton.shape[1] != 3:
        raise ValueError(f"Unexpected skeleton shape after alignment: {skeleton.shape}")

    hip = skeleton[0:1, :, :]
    skeleton = skeleton - hip

    temporal_length = skeleton.shape[2]
    indices = _resample_indices(temporal_length, target_length)
    skeleton = skeleton[:, :, indices]

    features = skeleton.astype(np.float32).reshape(-1)
    expected_dim = 20 * 3 * target_length
    if features.shape[0] != expected_dim:
        raise ValueError(f"Skeleton feature dim {features.shape[0]} != expected {expected_dim}")
    return features


def preprocess_inertial(mat_path: Path, target_length: int = 40) -> np.ndarray:
    inertial = _load_mat_array(mat_path, ["d_iner", "inertial", "iner"])

    if inertial.ndim != 2:
        raise ValueError(f"Inertial must be 2D, got shape {inertial.shape}")

    if inertial.shape[1] != 6 and inertial.shape[0] == 6:
        inertial = inertial.T

    if inertial.shape[1] != 6:
        raise ValueError(f"Unexpected inertial shape after alignment: {inertial.shape}")

    mean = inertial.mean(axis=0, keepdims=True)
    std = inertial.std(axis=0, keepdims=True) + 1e-8
    inertial = (inertial - mean) / std

    temporal_length = inertial.shape[0]
    indices = _resample_indices(temporal_length, target_length)
    inertial = inertial[indices, :]

    features = inertial.astype(np.float32).reshape(-1)
    expected_dim = 6 * target_length
    if features.shape[0] != expected_dim:
        raise ValueError(f"Inertial feature dim {features.shape[0]} != expected {expected_dim}")
    return features


def preprocess_depth(mat_path: Path, img_size: Tuple[int, int] = (64, 64)) -> np.ndarray:
    depth = _load_mat_array(mat_path, ["d_depth", "depth"])

    if depth.ndim != 3:
        raise ValueError(f"Depth must be 3D, got shape {depth.shape}")

    if depth.shape[2] < 2:
        raise ValueError(f"Depth temporal length too short for DMM: {depth.shape}")

    diff = np.abs(np.diff(depth, axis=2))
    dmm_front = diff.sum(axis=2)

    scale_h = img_size[0] / dmm_front.shape[0]
    scale_w = img_size[1] / dmm_front.shape[1]
    dmm = zoom(dmm_front, (scale_h, scale_w), order=1)

    dmm_min = float(dmm.min())
    dmm_max = float(dmm.max())
    dmm = (dmm - dmm_min) / (dmm_max - dmm_min + 1e-8)

    features = dmm.astype(np.float32).reshape(-1)
    expected_dim = img_size[0] * img_size[1]
    if features.shape[0] != expected_dim:
        raise ValueError(f"Depth feature dim {features.shape[0]} != expected {expected_dim}")
    return features


def _sample_video_frames(video_path: Path, img_size: Tuple[int, int], n_frames: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Could not open RGB video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        frames: List[np.ndarray] = []
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frames.append(cv2.resize(frame, img_size))
        cap.release()
        if not frames:
            raise ValueError(f"No readable frames in RGB video: {video_path}")
        frame_array = np.stack(frames, axis=0)
        indices = _resample_indices(frame_array.shape[0], n_frames)
        return frame_array[indices]

    indices = _resample_indices(total_frames, n_frames)
    sampled_frames: List[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        sampled_frames.append(cv2.resize(frame, img_size))
    cap.release()

    if not sampled_frames:
        raise ValueError(f"Failed to sample frames from RGB video: {video_path}")

    while len(sampled_frames) < n_frames:
        sampled_frames.append(sampled_frames[-1].copy())

    return np.stack(sampled_frames[:n_frames], axis=0)


def preprocess_rgb(video_path: Path, img_size: Tuple[int, int] = (64, 64), n_frames: int = 16) -> np.ndarray:
    sampled = _sample_video_frames(video_path, img_size=img_size, n_frames=n_frames)
    sampled = sampled.astype(np.float32) / 255.0
    mean_frame = sampled.mean(axis=0)

    features = mean_frame.reshape(-1)
    expected_dim = img_size[0] * img_size[1] * 3
    if features.shape[0] != expected_dim:
        raise ValueError(f"RGB feature dim {features.shape[0]} != expected {expected_dim}")
    return features


def build_dataset(
    raw_root: Path | str,
    modalities: Sequence[str] = ("skeleton", "inertial", "depth", "rgb"),
    config: PreprocessConfig = PreprocessConfig(),
    max_samples: int | None = None,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, List[str]]:
    raw_root = Path(raw_root)
    requested_modalities = tuple(modalities)

    for modality in requested_modalities:
        if modality not in {"skeleton", "inertial", "depth", "rgb"}:
            raise ValueError(f"Unsupported modality: {modality}")

    modality_files = get_modality_files(raw_root)
    prefix_sets = [set(modality_files[m].keys()) for m in requested_modalities]
    if not prefix_sets:
        raise ValueError("No modalities requested")

    common_prefixes = sorted(set.intersection(*prefix_sets))
    if not common_prefixes:
        raise ValueError("No common samples found for requested modalities")

    if max_samples is not None:
        common_prefixes = common_prefixes[: max(0, max_samples)]

    features: Dict[str, List[np.ndarray]] = {m: [] for m in requested_modalities}
    labels: List[int] = []
    subjects: List[int] = []
    kept_prefixes: List[str] = []

    for prefix in tqdm(common_prefixes, desc="Step 3 preprocessing", unit="sample"):
        try:
            per_sample: Dict[str, np.ndarray] = {}

            if "skeleton" in requested_modalities:
                per_sample["skeleton"] = preprocess_skeleton(
                    modality_files["skeleton"][prefix],
                    target_length=config.skeleton_target_length,
                )
            if "inertial" in requested_modalities:
                per_sample["inertial"] = preprocess_inertial(
                    modality_files["inertial"][prefix],
                    target_length=config.inertial_target_length,
                )
            if "depth" in requested_modalities:
                per_sample["depth"] = preprocess_depth(
                    modality_files["depth"][prefix],
                    img_size=config.depth_img_size,
                )
            if "rgb" in requested_modalities:
                per_sample["rgb"] = preprocess_rgb(
                    modality_files["rgb"][prefix],
                    img_size=config.rgb_img_size,
                    n_frames=config.rgb_n_frames,
                )

            action, subject, _trial = _parse_prefix(prefix)
            for modality in requested_modalities:
                features[modality].append(per_sample[modality])
            labels.append(action - 1)
            subjects.append(subject)
            kept_prefixes.append(prefix)
        except Exception as exc:
            print(f"[WARN] Skipping {prefix}: {exc}")

    if not kept_prefixes:
        raise RuntimeError("All samples failed during preprocessing")

    feature_arrays = {
        modality: np.stack(items, axis=0).astype(np.float32)
        for modality, items in features.items()
    }

    n_samples = len(kept_prefixes)
    for modality, feat_array in feature_arrays.items():
        if feat_array.shape[0] != n_samples:
            raise RuntimeError(
                f"Alignment error for modality {modality}: {feat_array.shape[0]} != {n_samples}"
            )

    label_array = np.asarray(labels, dtype=np.int64)
    subject_array = np.asarray(subjects, dtype=np.int64)

    if label_array.shape[0] != n_samples or subject_array.shape[0] != n_samples:
        raise RuntimeError("Label/subject alignment mismatch")

    return feature_arrays, label_array, subject_array, kept_prefixes


def save_feature_cache(
    output_dir: Path | str,
    features: Mapping[str, np.ndarray],
    labels: np.ndarray,
    subjects: np.ndarray,
    prefixes: Sequence[str],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for modality, array in features.items():
        np.save(output_dir / f"{modality}_features.npy", array)

    np.save(output_dir / "labels.npy", labels)
    np.save(output_dir / "subjects.npy", subjects)
    np.save(output_dir / "sample_prefixes.npy", np.asarray(prefixes, dtype=str))
