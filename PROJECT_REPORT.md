# Project Report — Multimodal Action Recognition (UTD-MHAD)

## 1. Executive Summary
This project implements and compares three fusion strategies for human action recognition on the UTD-MHAD dataset: Data-Level (early fusion), Feature-Level (intermediate fusion) and Decision-Level (late fusion: vote and learned combiner). The default evaluation follows the cross-subject holdout (test subjects: 1,3,5,7). Results show Decision-Level Vote as the best-performing method for our run.

## 2. Dataset
- Dataset: UTD-MHAD (University of Texas at Dallas Multimodal Human Action Dataset)
- Modalities used: Skeleton, Inertial, Depth, RGB
- Subjects: 8
- Actions: 27 classes
- Trials: up to 4 per subject per action
- Processed cache: `data/processed/` contains feature arrays saved by the preprocessing pipeline: `skeleton_features.npy`, `inertial_features.npy`, `depth_features.npy`, `rgb_features.npy`, `labels.npy`, `subjects.npy`.

Notes: the original dataset has 864 samples (8 × 27 × 4). The processed cache may be smaller if some samples were skipped during preprocessing (e.g., 861 samples in a prior run).

## 3. Methodology (implementation details)
- Preprocessing (see `utils/preprocessing.py`):
  - `skeleton`: hip-centered, temporally resampled to 40 frames, flattened (20 joints × 3 × 40).
  - `inertial`: 6 channels z-score normalized, resampled to 40 frames, flattened.
  - `depth`: motion map (DMM-like) from frame differences, resized to 64×64, normalized, flattened.
  - `rgb`: sample up to 16 frames, resize 64×64, normalize and use mean frame.
- Feature caching: `save_feature_cache()` writes `.npy` arrays to `data/processed/`.
- Splits (see `utils/splits.py`):
  - Default evaluation: subject holdout (`test_subjects=1,3,5,7`), which ensures no subject overlap between train and test (recommended for UTD-MHAD final reporting).
  - Optional: stratified k-fold cross-validation (use `--eval-mode kfold --cv-folds K`) for development stability.
- Models (in `models/`):
  - Data-Level: concatenated modality vectors → `StandardScaler` + SVM (RBF).
  - Feature-Level: per-modality class-variance selection (train-only) at percentile threshold (default 75%), selected features concatenated → scaler + SVM (RBF).
  - Decision-Level:
    - Unimodal SVMs (probability outputs) for each modality.
    - Vote fusion: sum modality probability vectors and argmax.
    - Learned fusion: small MLP (PyTorch) trained on concatenated modality probability vectors.

Training hyperparameters used in the run:
- SVM: RBF, C=10.0, gamma="scale"
- Feature selection percentile: 75.0 (default)
- Learned late-fusion: default `epochs=120`, `lr=1e-3` (adjustable via CLI)

## 4. Experimental setup
- Run command (subject holdout default):

```powershell
.\venv\Scripts\python.exe .\main.py
```

- To run k-fold instead:

```powershell
.\venv\Scripts\python.exe .\main.py --eval-mode kfold --cv-folds 5
```

- Outputs (subject holdout) are stored under `results/tables/` and `results/plots/`.

## 5. Results (subject holdout run)
The primary comparison table (saved at `results/tables/step9_fusion_comparison.csv`) contains:

| Method | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---:|---:|---:|---:|
| Data-Level | 0.3016 | 0.4767 | 0.3015 | 0.3144 |
| Feature-Level | 0.4107 | 0.5974 | 0.4100 | 0.4112 |
| Decision-Level Vote | 0.8329 | 0.8632 | 0.8330 | 0.8353 |
| Decision-Level Learned | 0.7100 | 0.8110 | 0.7094 | 0.7126 |


### Confusion matrices and accuracy comparison

- Accuracy comparison chart: `results/plots/step9_fusion_accuracy_comparison.png`

![](results/plots/step9_fusion_accuracy_comparison.png)

- Confusion matrices (plots):
  - Data-Level: `results/plots/step9_data-level_confusion_matrix.png`

![](results/plots/step9_data-level_confusion_matrix.png)

  - Feature-Level: `results/plots/step9_feature-level_confusion_matrix.png`

![](results/plots/step9_feature-level_confusion_matrix.png)

  - Decision-Level Vote: `results/plots/step9_decision-level_vote_confusion_matrix.png`

![](results/plots/step9_decision-level_vote_confusion_matrix.png)

  - Decision-Level Learned: `results/plots/step9_decision-level_learned_confusion_matrix.png`

![](results/plots/step9_decision-level_learned_confusion_matrix.png)

## 6. Discussion
- The Decision-Level Vote method outperforms the other fusion strategies on the subject-holdout evaluation (Accuracy ≈ 0.833). This suggests unimodal classifiers are complementary and score summation is an effective, low-risk fusion strategy for this dataset and these model choices.
- Feature-Level improves over naive Data-Level fusion in our run, likely because selecting discriminative features per modality reduces noise and helps the SVM.
- The learned late fusion shows reasonable performance (Accuracy ≈ 0.71) but is sensitive to training configuration; with limited training data and an MLP that has many parameters, it can underperform if not tuned or regularized.

## 7. Overfitting checks and recommendations
- Check per-fold vs out-of-fold variance (use `--eval-mode kfold`) to detect instability.
- If k-fold scores are substantially higher than subject-holdout scores, prefer the subject-holdout as the final reported metric.
- For the learned combiner: reduce capacity, add weight decay, or use early stopping.

## 8. Comparison with prior work
I reviewed the two provided research papers and extracted the numeric comparison points that are explicitly visible in the text you shared.

### Paper 1: UTD-MHAD dataset paper
- Dataset size: 27 actions, 8 subjects, 4 trials per action, 861 valid sequences after removing corrupted samples.
- The paper presents a multimodal fusion example using depth and inertial signals and reports that fusion improves accuracy by **more than 11%** over the corresponding single-modality baseline in that experiment.
- The key point for our project is the evaluation protocol: the paper is centered on a **cross-subject / subject-independent** setting, which matches our default subject-holdout evaluation choice.

### Paper 2: Fusion of Video and Inertial Sensing for Deep Learning-Based Human Action Recognition
- Evaluation protocol: **leave-one-subject-out cross validation** on UTD-MHAD.
- Reported average accuracies:
  - Video only: **76.0%**
  - Inertial only: **90.5%**
  - Feature-level fusion: **94.1%**
  - Decision-level fusion: **95.6%**

### Comparison against our project

| Method / Study | Protocol | Reported Accuracy |
|---|---|---:|
| Our Data-Level fusion | Subject holdout | 30.16% |
| Our Feature-Level fusion | Subject holdout | 41.07% |
| Our Decision-Level Vote | Subject holdout | 83.29% |
| Our Decision-Level Learned | Subject holdout | 71.00% |
| Paper 2 Video only | Leave-one-subject-out | 76.0% |
| Paper 2 Inertial only | Leave-one-subject-out | 90.5% |
| Paper 2 Feature-level fusion | Leave-one-subject-out | 94.1% |
| Paper 2 Decision-level fusion | Leave-one-subject-out | 95.6% |

### Interpretation
- Our best method is **Decision-Level Vote (83.29%)**, which is strong for a lightweight classical-machine-learning pipeline but still below the deep-learning fusion results in Paper 2.
- The gap is expected because Paper 2 uses dedicated deep models for each modality (3D CNN for video and 2D CNN for inertial), while our current implementation uses cached handcrafted/summary features with SVM classifiers and a small MLP combiner.
- The fact that the fusion-based methods in Paper 2 improve from 76.0% / 90.5% to 94.1% / 95.6% supports the same overall conclusion seen in our project: **late fusion is usually the strongest strategy when modalities are complementary**.
- Our Data-Level baseline is much weaker than the fusion methods, which is consistent with the paper trend that simple concatenation is often not enough when modalities have very different structure and scale.

### Relation to our implementation
- The UTD-MHAD dataset paper supports the dataset choice and the subject-independent protocol.
- The deep fusion paper provides a strong external benchmark showing that fusion can reach the mid-90% range when modality-specific deep models are used.
- Our project is therefore a good semester-level reproduction and comparison study, but it is intentionally lighter-weight than the deep-learning benchmark.

## 9. References
- Chen, C., Jafari, R., and Kehtarnavaz, N. UTD-MHAD: A Multimodal Dataset for Human Action Recognition Utilizing a Depth Camera and a Wearable Inertial Sensor.
- Wei, H., Jafari, R., and Kehtarnavaz, N. Fusion of Video and Inertial Sensing for Deep Learning-Based Human Action Recognition.


---

*Report generated from project artifacts in this repository. Figures are in `results/plots/` and tables in `results/tables/`.*
