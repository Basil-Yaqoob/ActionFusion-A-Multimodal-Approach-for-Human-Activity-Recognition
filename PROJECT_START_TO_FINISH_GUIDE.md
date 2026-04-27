# ActionFusion Project Guide (Start to Finish)

## What This Project Does

This project performs multimodal human action recognition on the UTD-MHAD dataset and compares four fusion strategy outputs:

1. Data-Level fusion (early fusion)
2. Feature-Level fusion (intermediate fusion)
3. Decision-Level Vote fusion (late fusion)
4. Decision-Level Learned fusion (late fusion neural combiner)

It uses four modalities per sample:

1. Skeleton
2. Inertial
3. Depth
4. RGB

Final outputs are saved as tables and plots under the results folder.

## End-to-End Pipeline (Conceptual Steps)

1. Environment setup
2. Raw dataset placement and modality indexing
3. Per-modality preprocessing
4. Alignment and cache writing
5. Evaluation split selection
6. Data-Level training and evaluation
7. Feature-Level training and evaluation
8. Decision-Level training and evaluation
9. Fusion comparison ranking and plots

## How Each Step Works

### 1) Environment setup

- Create and activate a Python virtual environment.
- Install dependencies from requirements.txt.

### 2) Raw dataset placement and indexing

- Raw files must be under data/raw with modality folders:
  - data/raw/RGB
  - data/raw/Depth
  - data/raw/Skeleton
  - data/raw/Inertial
- The loader scans recursively and extracts sample prefixes like a01_s01_t01.
- Only prefixes present across all requested modalities are used.
- Files with macOS artifact names (for example ._*) are ignored.

### 3) Per-modality preprocessing

- Skeleton:
  - Loads joint tensor
  - Aligns axes if needed
  - Hip-centers joints
  - Resamples to fixed temporal length (default 40)
  - Flattens to feature vector
- Inertial:
  - Loads IMU channels
  - Ensures 6-channel shape
  - Applies per-channel z-score normalization
  - Resamples to fixed temporal length (default 40)
  - Flattens to feature vector
- Depth:
  - Loads depth sequence
  - Computes absolute frame differences and motion map (DMM-like)
  - Resizes to 64x64
  - Normalizes to [0, 1]
  - Flattens to feature vector
- RGB:
  - Reads video frames
  - Uniformly samples fixed frames (default 16)
  - Resizes to 64x64
  - Normalizes pixel values
  - Uses mean frame summary
  - Flattens to feature vector

### 4) Alignment and cache writing

- For each kept sample prefix, features from all modalities are stored in aligned order.
- Labels are converted to 0-based indexing (action - 1).
- Subject IDs are extracted and stored.
- Cached arrays are written to data/processed:
  - skeleton_features.npy
  - inertial_features.npy
  - depth_features.npy
  - rgb_features.npy
  - labels.npy
  - subjects.npy
  - sample_prefixes.npy

### 5) Evaluation split selection

There are two evaluation modes in [main.py](main.py):

1. `subject` default mode
  - Uses the original cross-subject holdout split.
  - Default test subjects are 1, 3, 5, 7.
  - Train and test subject overlap is disallowed.
  - This is the best final reporting choice for UTD-MHAD because it tests generalization to unseen people.
2. `kfold` optional mode
  - Runs stratified k-fold cross-validation on the cached samples.
  - Each sample becomes a test sample exactly once.
  - Produces averaged comparison metrics across folds.
  - This is useful for a more stable development estimate, but it can be less strict than subject holdout.

Important note:
- K-fold often uses more of the data for training in each fold, so it can give a stronger or smoother estimate.
- It does not guarantee higher real-world accuracy, especially if the deployment setting is cross-subject.
- For the final project result, subject split is the safer default.

### 6) Data-Level fusion

- Concatenate full feature vectors from all modalities into one large vector.
- Train SVM (RBF) with StandardScaler in a pipeline.
- Evaluate on test split.
- Save confusion matrix plot.

### 7) Feature-Level fusion

- For each modality, compute class-wise variance using train data only.
- Select top features above percentile threshold (default 75).
- Concatenate selected modality features.
- Scale, train SVM (RBF), evaluate.
- Save confusion matrix plot.

### 8) Decision-Level fusion

- Train unimodal SVM classifiers (with probability outputs) per modality.
- Build two late-fusion outputs:
  - Vote-style score fusion (sum probabilities, argmax)
  - Learned combiner network (PyTorch MLP over concatenated modality scores)
- Evaluate and save confusion matrix plots.

### 9) Comparison and ranking

- Compute metrics for each fusion strategy:
  - accuracy
  - precision_macro
  - recall_macro
  - f1_macro
- Save comparison table and rank by accuracy then f1.
- Save strategy comparison bar plot.

## Exact Output Artifacts

After end-to-end run, expected key artifacts include:

- results/tables/step9_fusion_comparison.csv
- results/tables/step9_fusion_ranking.csv
- results/plots/step9_fusion_accuracy_comparison.png
- results/plots/step9_data_level_confusion_matrix.png
- results/plots/step9_feature_level_confusion_matrix.png
- results/plots/step9_decision_level_vote_confusion_matrix.png
- results/plots/step9_decision_level_learned_confusion_matrix.png

## How To Run The Entire Project (Windows, from start)

Run these from the repository root.

### A) Environment setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### B) Build processed cache from raw data

```powershell
@'
from pathlib import Path
from utils.preprocessing import PreprocessConfig, build_dataset, save_feature_cache

features, labels, subjects, prefixes = build_dataset(
    raw_root=Path("data/raw"),
    modalities=("skeleton", "inertial", "depth", "rgb"),
    config=PreprocessConfig(),
)

save_feature_cache(
    output_dir=Path("data/processed"),
    features=features,
    labels=labels,
    subjects=subjects,
    prefixes=prefixes,
)

print(f"Cached {labels.shape[0]} aligned samples to data/processed")
for name, arr in features.items():
    print(name, arr.shape)
'@ | .\venv\Scripts\python.exe -
```

### C) Run full end-to-end fusion comparison

```powershell
.\venv\Scripts\python.exe .\main.py
```

This now runs the subject holdout comparison by default.

### D) Run k-fold instead

```powershell
.\venv\Scripts\python.exe .\main.py --eval-mode kfold --cv-folds 5
```

## Minimal Re-Run (if cache already exists)

If `data/processed` already contains all required `.npy` files, run only:

```powershell
.\venv\Scripts\python.exe .\main.py
```

## Quick Validation Checklist

1. `data/processed` contains modality features + labels + subjects.
2. `main.py` completes without missing-file error.
3. `results/tables` contains comparison and ranking csv files, plus fold summaries in k-fold mode.
4. `results/plots` contains 4 confusion matrices + 1 comparison chart.

## Notes

- The project remains CPU-runnable.
- Deterministic behavior is used where applicable in late-fusion training.
- Feature selection is train-only to prevent leakage.
