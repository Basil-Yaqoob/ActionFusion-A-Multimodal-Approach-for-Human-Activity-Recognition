# ActionFusion — Project Report

## Abstract

This report summarizes the multimodal fusion experiments implemented in this repository on the UTD-MHAD dataset. We compare three fusion strategies: data-level (early) fusion, feature-level fusion, and decision-level fusion (both majority-vote and a learned late-fusion network). Evaluation is performed with the original subject holdout split and with Leave-One-Subject-Out (LOSO). The report includes tables and plots generated during experiments and a short comparison to two reference papers on UTD-MHAD.

## Methodology

- Preprocess each modality to a fixed-size feature vector (skeleton, inertial, depth, RGB).
- Implement three fusion strategies:
  - Data-level (early) fusion: concatenate all modality features and train an SVM.
  - Feature-level fusion: per-modality feature selection (variance across classes on train set) then concatenate selected features and train an SVM.
  - Decision-level fusion: train per-modality SVMs producing class probability vectors; combine with (a) majority vote, and (b) a small neural network (LateFusionNet) trained on stacked modality probability vectors.
- Evaluate under multiple protocols: fixed subject holdout (subjects 1,3,5,7) and LOSO.

## Dataset

We use the UTD-MHAD dataset (UTD Multimodal Human Action Dataset). Key facts:

- 27 actions, 8 subjects, each action repeated multiple times (original dataset ~861 sequences after cleaning).
- Modalities: RGB video, depth video, skeleton joint positions, wearable inertial sensor signals.
- Data are temporally synchronized and stored as `.avi` (RGB) and `.mat` files (depth, skeleton, inertial).

See the original paper `UTD-MHAD: A Multimodal Dataset for Human Action Recognition` (Chen et al.) for details on sensors, subjects, and action taxonomy.

## Implementation Details

- Preprocessing (utils/preprocessing.py):
  - Skeleton: center on hip, resample to fixed temporal length, flatten.
  - Inertial: z-score per-channel, resample to fixed length, flatten.
  - Depth: compute Depth Motion Map (DMM) from temporal differences, resize to fixed image size, flatten.
  - RGB: sample fixed number of frames, resize, average frames to mean image, flatten.
- Caching: processed arrays saved to `data/processed/` as `.npy` files.
- Models (summary):
  - Data-level: `SVC` pipeline with `StandardScaler`.
  - Feature-level: train-only variance-based feature selection per modality, then `SVC` on fused selected-features.
  - Decision-level: per-modality `SVC` (probabilities), then majority vote or `LateFusionNet` (PyTorch) trained with cross-entropy.

## Evaluation Protocols

- Subject holdout (default): test subjects `1,3,5,7` vs rest for training.
- Leave-One-Subject-Out (LOSO): each subject held out once (strong subject-independent test).

All metric calculations use accuracy, macro precision, macro recall, and macro F1.

## Results

### Overall Comparison (Subject holdout)

The table below shows the overall fusion comparison for the subject holdout split (from `results/tables/step9_fusion_comparison.csv`):

| Method                 | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
| ---------------------- | -------: | ----------------: | -------------: | ---------: |
| Data-Level             |   0.3016 |            0.4767 |         0.3015 |     0.3144 |
| Feature-Level          |   0.4107 |            0.5974 |         0.4100 |     0.4112 |
| Decision-Level Vote    |   0.8190 |            0.8621 |         0.8190 |     0.8231 |
| Decision-Level Learned |   0.7285 |            0.8164 |         0.7278 |     0.7273 |

Full CSV: [results/tables/step9_fusion_comparison.csv](results/tables/step9_fusion_comparison.csv)

### LOSO Results

LOSO overall comparison (from `results/tables/step9_loso_fusion_comparison.csv`):

| Method                 | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
| ---------------------- | -------: | ----------------: | -------------: | ---------: |
| Data-Level             |   0.3751 |            0.5533 |         0.3752 |     0.4047 |
| Feature-Level          |   0.5134 |            0.6528 |         0.5138 |     0.5397 |
| Decision-Level Vote    |  0.86295 |            0.8700 |        0.86305 |     0.8645 |
| Decision-Level Learned |  0.80023 |            0.8747 |        0.80048 |    0.81327 |

Full CSV: [results/tables/step9_loso_fusion_comparison.csv](results/tables/step9_loso_fusion_comparison.csv)

### Per-subject (LOSO) Fold Metrics

We provide per-subject fold metrics in `results/tables/step9_loso_fold_metrics.csv`. Example excerpt (subjects 1–8):

| fold | held_out_subject | method              | accuracy | precision_macro | recall_macro | f1_macro |
| ---: | ---------------: | ------------------- | -------: | --------------: | -----------: | -------: |
|    1 |                1 | Decision-Level Vote |   0.9533 |          0.9611 |       0.9537 |   0.9514 |
|    2 |                2 | Decision-Level Vote |   0.9352 |          0.9481 |       0.9352 |   0.9347 |
|    3 |                3 | Decision-Level Vote |   0.8426 |          0.8641 |       0.8426 |   0.8345 |
|    4 |                4 | Decision-Level Vote |   0.8056 |          0.7683 |       0.8056 |   0.7696 |
|    5 |                5 | Decision-Level Vote |   0.8704 |          0.9041 |       0.8704 |   0.8619 |

Full CSV: [results/tables/step9_loso_fold_metrics.csv](results/tables/step9_loso_fold_metrics.csv)

### Plots and Confusion Matrices

Accuracy comparison (subject holdout):

![Subject holdout accuracy comparison](results/plots/step9_fusion_accuracy_comparison.png)

Accuracy comparison (LOSO):

![LOSO accuracy comparison](results/plots/step9_loso_fusion_accuracy_comparison.png)

Confusion matrices (subject holdout):

![Data-level confusion matrix](results/plots/step9_data-level_confusion_matrix.png)
![Feature-level confusion matrix](results/plots/step9_feature-level_confusion_matrix.png)
![Decision-level vote confusion matrix](results/plots/step9_decision-level_vote_confusion_matrix.png)
![Decision-level learned confusion matrix](results/plots/step9_decision-level_learned_confusion_matrix.png)

Confusion matrices (LOSO):

![LOSO Data-level confusion matrix](results/plots/step9_loso_data-level_confusion_matrix.png)
![LOSO Feature-level confusion matrix](results/plots/step9_loso_feature-level_confusion_matrix.png)
![LOSO Decision-level vote confusion matrix](results/plots/step9_loso_decision-level_vote_confusion_matrix.png)
![LOSO Decision-level learned confusion matrix](results/plots/step9_loso_decision-level_learned_confusion_matrix.png)

## Discussion

- Decision-level fusion (majority vote) produced the best performance in our experiments (Accuracy ~81.9% subject holdout, ~86.3% LOSO).
- Learned late fusion performed well but lower than simple vote on the subject holdout split; on LOSO it is closer to the vote but still behind (~80.0%). Possible reasons:
  - Per-modality SVM outputs are already highly informative; a shallow neural combiner can overfit if training data for the combiner is limited.
  - Majority vote is robust and requires no training, while learned fusion requires careful regularization and validation.
- Feature-level fusion improves over naive early fusion but is still substantially lower than decision-level fusion, indicating the value of modality-specific modeling.

## Related Work & Comparison

We compare our results with two relevant papers on UTD-MHAD:

1. Chen et al., "UTD-MHAD: A Multimodal Dataset for Human Action Recognition" — dataset description and baseline results using DMM+inertial fusion (classical methods). (See dataset description and earlier baselines in that paper.)
2. Wei et al., "Fusion of Video and Inertial Sensing for Deep Learning-Based Human Action Recognition" — deep models (3D CNN for video, 2D CNN for inertial), evaluated with LOSO. Reported accuracies (from paper):

| Modality / Fusion     | Reported LOSO Accuracy (paper) |
| --------------------- | -----------------------------: |
| Video only            |                          76.0% |
| Inertial only         |                          90.5% |
| Feature-level fusion  |                          94.1% |
| Decision-level fusion |                **95.6%** |

Comparison notes:

- Our decision-level LOSO (vote) = 86.3% vs reported 95.6% in Wei et al. Differences likely due to:

  - Wei et al. use deep CNNs (3D/2D) for modality-specific modeling which are more expressive than SVMs on handcrafted features.
  - Their preprocessing converts video volumes and inertial signals into formats suitable for CNNs (e.g., video volumes 320×240×32, inertial as small 2D images), then trains end-to-end.
  - Our pipeline uses classical features (DMM for depth, averaged RGB frames, flattened skeleton/inertial vectors) and SVMs, so lower capacity but faster training and much smaller data requirements.
- Nevertheless, our results show decision-level fusion (even with classical SVMs) strongly outperforms data- or feature-level SVM baselines, which qualitatively agrees with the finding that combining modality-specific strengths improves recognition.

## References

- Chen C., Jafari R., Kehtarnavaz N., "UTD-MHAD: A Multimodal Dataset for Human Action Recognition", University of Texas at Dallas.
- Wei H., Jafari R., Kehtarnavaz N., "Fusion of Video and Inertial Sensing for Deep Learning-Based Human Action Recognition".
