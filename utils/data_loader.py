from __future__ import annotations

from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Set, Tuple

import cv2
import h5py
import numpy as np
import scipy.io as sio


MODALITY_SPECS: Dict[str, Tuple[str, str]] = {
    "rgb": ("RGB", "_color.avi"),
    "depth": ("Depth", "_depth.mat"),
    "skeleton": ("Skeleton", "_skeleton.mat"),
    "inertial": ("Inertial", "_inertial.mat"),
}

PREFIX_PATTERN = re.compile(r"^a(\d{1,2})_s(\d{1,2})_t(\d{1,2})", re.IGNORECASE)


def _extract_prefix(filename: str) -> Optional[str]:
    match = PREFIX_PATTERN.match(filename)
    if not match:
        return None
    action, subject, trial = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
    return f"a{action:02d}_s{subject:02d}_t{trial:02d}"


def _list_files(path: Path, suffix: str) -> List[Path]:
    if not path.exists():
        return []
    suffix_lower = suffix.lower()
    def _is_macos_artifact(file_path: Path) -> bool:
        if file_path.name.startswith("._"):
            return True
        return any(part == "__MACOSX" for part in file_path.parts)

    return sorted(
        [
            p
            for p in path.rglob("*")
            if p.is_file()
            and p.name.lower().endswith(suffix_lower)
            and not _is_macos_artifact(p)
        ]
    )


def get_modality_files(raw_root: Path) -> Dict[str, Dict[str, Path]]:
    file_maps: Dict[str, Dict[str, Path]] = {}
    for modality, (folder, suffix) in MODALITY_SPECS.items():
        modality_dir = raw_root / folder
        per_modality: Dict[str, Path] = {}
        for file_path in _list_files(modality_dir, suffix):
            prefix = _extract_prefix(file_path.name)
            if prefix is not None and prefix not in per_modality:
                per_modality[prefix] = file_path
        file_maps[modality] = per_modality
    return file_maps


def get_modality_prefixes(raw_root: Path) -> Dict[str, Set[str]]:
    file_maps = get_modality_files(raw_root)
    return {modality: set(prefix_map.keys()) for modality, prefix_map in file_maps.items()}


def validate_naming_patterns(raw_root: Path) -> Dict[str, int]:
    invalid_counts: Dict[str, int] = {}
    for modality, (folder, suffix) in MODALITY_SPECS.items():
        modality_dir = raw_root / folder
        invalid = 0
        for file_path in _list_files(modality_dir, suffix):
            if _extract_prefix(file_path.name) is None:
                invalid += 1
        invalid_counts[modality] = invalid
    return invalid_counts


def summarize_dataset(raw_root: Path) -> Dict[str, object]:
    modality_prefixes = get_modality_prefixes(raw_root)

    counts = {modality: len(prefixes) for modality, prefixes in modality_prefixes.items()}

    all_sets = [prefixes for prefixes in modality_prefixes.values() if prefixes]
    common_prefixes = set.intersection(*all_sets) if all_sets else set()
    union_prefixes = set.union(*all_sets) if all_sets else set()

    missing_by_modality = {
        modality: len(union_prefixes - prefixes)
        for modality, prefixes in modality_prefixes.items()
    }

    return {
        "counts": counts,
        "common_samples": len(common_prefixes),
        "union_samples": len(union_prefixes),
        "missing_by_modality": missing_by_modality,
        "naming_invalid_counts": validate_naming_patterns(raw_root),
        "common_prefixes": sorted(common_prefixes),
    }


def _load_mat_array(mat_path: Path, preferred_keys: Iterable[str]) -> np.ndarray:
    try:
        content = sio.loadmat(str(mat_path))
        for key in preferred_keys:
            if key in content:
                return np.asarray(content[key])
    except NotImplementedError:
        pass

    with h5py.File(mat_path, "r") as file:
        for key in preferred_keys:
            if key in file:
                return np.array(file[key]).T

    raise KeyError(f"None of keys {list(preferred_keys)} found in {mat_path.name}")


def quick_sanity_check(raw_root: Path, prefix: Optional[str] = None) -> Dict[str, object]:
    summary = summarize_dataset(raw_root)
    modality_files = get_modality_files(raw_root)
    common_prefixes: List[str] = summary["common_prefixes"]  # type: ignore[assignment]
    if not common_prefixes:
        return {
            "ok": False,
            "reason": "No common sample prefixes found across modalities.",
        }

    chosen_prefix = prefix if prefix is not None else common_prefixes[0]

    rgb_path = modality_files["rgb"].get(chosen_prefix)
    depth_path = modality_files["depth"].get(chosen_prefix)
    skel_path = modality_files["skeleton"].get(chosen_prefix)
    iner_path = modality_files["inertial"].get(chosen_prefix)

    if not all(path is not None and path.exists() for path in (rgb_path, depth_path, skel_path, iner_path)):
        return {
            "ok": False,
            "reason": f"Missing one or more modality files for prefix {chosen_prefix}",
        }

    skeleton = _load_mat_array(skel_path, ["d_skel", "skeleton", "skel"])
    inertial = _load_mat_array(iner_path, ["d_iner", "inertial", "iner"])
    depth = _load_mat_array(depth_path, ["d_depth", "depth"])

    cap = cv2.VideoCapture(str(rgb_path))
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return {
            "ok": False,
            "reason": f"Failed to read first RGB frame for {chosen_prefix}",
        }

    return {
        "ok": True,
        "prefix": chosen_prefix,
        "skeleton_shape": tuple(int(x) for x in skeleton.shape),
        "inertial_shape": tuple(int(x) for x in inertial.shape),
        "depth_shape": tuple(int(x) for x in depth.shape),
        "rgb_frame_shape": tuple(int(x) for x in frame.shape),
    }
