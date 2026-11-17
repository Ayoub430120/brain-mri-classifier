# Brain MRI Tumor Classification with Transfer Learning

Train, validate, evaluate, and interpret a **MobileNetV2** image classifier for 2-D brain MRI slices, using TensorFlow / Keras in Google Colab.

> **Educational use only.** This project is not a medical device and must not be used to diagnose, screen, or guide treatment. Metrics and Grad-CAM maps do not establish clinical reliability or causal explanations. See [Limitations](#limitations-and-responsible-use).

## Overview

- **Task:** classify brain MRI slices into `glioma`, `meningioma`, `pituitary`, and `notumor`. The notebook works with any image dataset arranged as one folder per class.
- **Approach:** two-stage transfer learning with MobileNetV2 pre-trained on ImageNet:
  1. freeze the backbone and train a small classification head;
  2. unfreeze the top backbone layers and fine-tune with a much smaller learning rate.
- **Training details:** class-weighted cross-entropy, mild geometric augmentation (rotation, zoom, translation), dropout, Adam optimizer, early stopping and checkpointing on validation loss.
- **Evaluation:** held-out test split evaluated only after the checkpoint is chosen, with a wide set of metrics and Grad-CAM visual explanations.

```
Drive images → integrity + duplicate checks → stratified train/validation split
   → tf.data (decode, resize, cache) → augmentation + MobileNetV2 (frozen) → head training
   → fine-tune top layers → reload best checkpoint → evaluate on held-out test
   → ROC/PR, calibration, error review, Grad-CAM → artifacts saved to Drive
```

## Features

- Dataset discovery and validation (class consistency, empty-class checks, hidden-file filtering)
- Exact-duplicate detection (SHA-256) within and across splits, with optional removal
- Stratified validation split when no `validation/` folder exists
- `tf.data` pipeline with optional in-memory caching
- Metrics: accuracy, balanced accuracy, per-class precision / sensitivity / specificity / F1, ROC-AUC, average precision, MCC, Cohen's kappa, log loss, Brier score, expected calibration error, top-2 accuracy
- 95 % bootstrap confidence intervals
- Training curves, confusion matrices, ROC / PR curves, reliability diagram, error review
- Grad-CAM heat maps
- Single-image prediction helper
- Reproducibility: fixed seeds, full run configuration and library versions saved

## Dataset

Suggested: the [Brain Tumor MRI Dataset](https://doi.org/10.34740/KAGGLE/DSV/2645886) on Kaggle (CC0 licence, 7,023 images: about 5,712 for training and 1,311 for testing). It combines several public sources, including the [figshare brain tumor dataset](https://doi.org/10.6084/m9.figshare.1512427.v8), so exact duplicates across the official Training and Testing folders are possible. The notebook checks for them.

Expected layout in Google Drive:

```
MyDrive/brain_mri_project/
├── Training/            (or train/)
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/             (or test/)   ← same class folders
```

An optional `validation/` folder (next to `train/` and `test/`) is used as-is; otherwise a stratified validation split is taken from the training folder.

**Privacy:** upload only data you are authorized to share. Do not upload identifiable or sensitive MRI data without appropriate authorization. The dataset itself is not included in this repository.

## How to run

1. Create `MyDrive/brain_mri_project/` in Google Drive and place the dataset folders inside it.
2. Open `brain_mri_classifier.ipynb` in Google Colab (upload it to Drive, or use **File → Upload notebook**).
3. Select **Runtime → Change runtime type → GPU**.
4. Review the settings in the *Configuration* section (`PROJECT_DIR`, `IMAGE_SIZE`, `BATCH_SIZE`, `EPOCHS`, `FINE_TUNE_EPOCHS`, etc.).
5. Run **Runtime → Run all** and keep the runtime connected until the last cell finishes.

Results are written to `artifacts/<run id>/` (plus a `.zip`) in your Drive project folder.

The notebook can also run locally (it falls back to `./brain_mri_project` when not in Colab); a GPU is strongly recommended.

## Requirements

Installed automatically if missing:

- Python 3.10+
- TensorFlow >= 2.16, < 3
- opencv-python-headless >= 4.9
- scikit-learn >= 1.4
- matplotlib >= 3.8
- pandas >= 2
- pillow

## Configuration highlights

| Setting | Meaning |
| --- | --- |
| `PROJECT_DIR` | Drive folder containing the dataset folders |
| `IMAGE_SIZE`, `BATCH_SIZE` | Input resolution (square) and batch size |
| `VALIDATION_SPLIT` | Share of training data used for validation (if no `validation/` folder) |
| `CACHE_IN_MEMORY` | Cache decoded images in RAM (about 150 KB per image) |
| `CHECK_DUPLICATES`, `DROP_DUPLICATES` | Find and optionally drop exact duplicate images |
| `EPOCHS`, `FINE_TUNE_EPOCHS`, `FINE_TUNE_LAYERS` | Stage 1 length, stage 2 length, number of unfrozen top backbone layers |
| `WEIGHTS` | `"imagenet"` or `None` (train from scratch) |
| `SEED` | Random seed |

## Output files

| File | Contents |
| --- | --- |
| `brain_mri_classifier.keras` | Best model checkpoint (lowest validation loss) |
| `class_names.json`, `run_config.json` | Class order, all settings, split sizes, library versions |
| `training_history.csv` | Per-epoch loss and accuracy for both stages |
| `metrics.json`, `validation_metrics.json` | All metrics for test and validation splits |
| `test_predictions.csv`, `validation_predictions.csv` | Per-image predictions with class probabilities |
| `test_probabilities.npz`, `validation_probabilities.npz` | Raw probabilities for re-analysis |
| `*.png` | Training curves, confusion matrices, ROC / PR, calibration, error examples, Grad-CAM |

## Reusing the saved model

```python
import json
import tensorflow as tf

model = tf.keras.models.load_model("brain_mri_classifier.keras")
class_names = json.load(open("class_names.json"))  # never assume alphabetical order
```

The model accepts raw 0-255 RGB images of size `IMAGE_SIZE × IMAGE_SIZE` (preprocessing is built in).

## Limitations and responsible use

- **Not a clinical tool.** Trained on a small public dataset of 2-D slices, with no regulatory clearance.
- **Possible patient-level leakage.** Public datasets rarely include patient IDs, so slices from one patient may land in different splits and inflate scores. Exact-duplicate checks do not remove this risk.
- **No external validation.** Different scanners, protocols, and populations can change performance drastically.
- **Label quality** was not re-verified.
- **Class imbalance and subgroups** can be hidden by aggregate metrics.
- **Calibration:** softmax confidence is often over-confident.
- **Interpretability:** Grad-CAM maps are coarse and can look plausible for the wrong reasons.
- **Reproducibility:** seeds are fixed, but GPU operations are not bit-for-bit deterministic.
- **Data protection:** respect licences, consent, and regulations (e.g. HIPAA, GDPR) before using real patient data.

## References

1. Sandler et al. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. CVPR. arXiv:1801.04381.
2. Deng et al. (2009). ImageNet: A large-scale hierarchical image database. CVPR.
3. Selvaraju et al. (2017). Grad-CAM. ICCV. arXiv:1610.02391.
4. Guo et al. (2017). On calibration of modern neural networks. ICML. arXiv:1706.04599.
5. Nickparvar (2021). Brain Tumor MRI Dataset. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/2645886
6. Cheng (2017). brain tumor dataset. figshare. https://doi.org/10.6084/m9.figshare.1512427.v8
7. Kingma & Ba (2015). Adam. ICLR. arXiv:1412.6980.
8. Srivastava et al. (2014). Dropout. JMLR 15.
9. Yosinski et al. (2014). How transferable are features in deep neural networks? NeurIPS.
10. Kapoor & Narayanan (2023). Leakage and the reproducibility crisis in ML-based science. Patterns 4(9).

The notebook lists the full reference set (18 entries).

## License

Add a license of your choice (e.g. MIT) for the code. The dataset is distributed separately under its own licence (CC0 for the suggested Kaggle dataset).
