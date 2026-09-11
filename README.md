# Fake Currency Detection (PKR) 🇵🇰

A deep learning system that classifies Pakistani Rupee (PKR) currency notes as **Real**, **Fake**, or **Not A Note**, using a CNN with transfer learning, explainability (Grad-CAM), an OCR-based text-verification safety net, and a live interactive dashboard.

> **Status:** Complete — Capstone project for AI/ML Internship (Sept 2026)

---

## 1. Problem Statement

Counterfeit currency detection is traditionally done manually or with specialized hardware. This project explores whether a CNN can learn to distinguish real vs. fake PKR notes from a photo alone — while being honest about where a photo-only approach structurally cannot succeed, and layering in additional techniques (a rejection class, OCR text-checking) to cover some of those gaps.

## 2. Dataset

- **Source:** Kaggle — "Real and Fake Currency Pakistanis Dataset" (~1.23 GB, real-world casual photos)
- **Classes:**
  - Real Notes: 950 images
  - Fake Notes: 650 images
  - Not A Note: 165 images (Fashion-MNIST samples + supplementary images, added after discovering the out-of-distribution problem — see Section 6)
- **Total:** 1,765 images, split 80/20 (train/val), stratified by class

## 3. Approach

1. **EDA** — class balance, image size variation (300px–4160px), visual inspection of real vs. fake samples
2. **Preprocessing** — resize to 224×224, normalize, augment (rotation, brightness, zoom — no flips, since notes have a fixed orientation)
3. **Modeling** — baseline CNN from scratch, then transfer learning (MobileNetV2) for comparison
4. **Evaluation** — accuracy, precision, recall, F1-score, confusion matrix, ROC-AUC, manual misclassified-image review
5. **Explainability** — Grad-CAM heatmaps to visualize model attention
6. **Iteration** — two real failure modes were discovered through live testing and fixed (see Section 6)
7. **Deployment** — Streamlit dashboard with upload/camera input, manual cropping, and OCR-based safety checks

## 4. Repository Structure

```
fake-currency-detection/
├── data/
│   ├── raw/                 # Original downloaded dataset (Fake Notes / Real Notes / Not A Note)
│   └── processed/           # train/val split, ready for training
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_cnn.ipynb
│   ├── 04_transfer_learning.ipynb
│   ├── 05_evaluation.ipynb
│   ├── 06_gradcam.ipynb
│   └── 07_add_not_a_note_class.ipynb
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── gradcam.py
├── models/
│   ├── baseline_cnn.keras
│   ├── mobilenetv2_fake_currency.keras       # original 2-class model
│   └── mobilenetv2_3class_currency.keras     # final 3-class model (used in app)
├── app/
│   └── streamlit_app.py
├── reports/
│   ├── figures/
│   └── EDA_report.md
├── tests/
│   └── test_preprocessing.py
├── requirements.txt
└── README.md
```

## 5. How to Run

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd fake-currency-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install Tesseract OCR (separate program, not just a pip package)
#    Download from: github.com/UB-Mannheim/tesseract/wiki
#    Add the Urdu language pack (urd.traineddata) to the tessdata folder
#    for the OCR safety-net feature to work on Urdu-script note text

# 4. Launch the dashboard
cd app
streamlit run streamlit_app.py
```

## 6. Results & Model Evolution

This project went through two major iterations after real failures were found during live testing — this section documents that process honestly, since it's the most technically meaningful part of the work.

### 6.1 Baseline vs. Transfer Learning (2-Class)

| Model | Test Accuracy | Test Loss | Notes |
|---|---|---|---|
| Baseline CNN (from scratch) | ~82% | ~0.44 (unstable) | Noisy validation loss, weaker generalization |
| MobileNetV2 (Transfer Learning) | ~93.4% | ~0.16–0.19 | Smooth training, ROC-AUC = 0.985 |

**2-class evaluation (Fake vs. Real):**

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Fake Notes | 0.90 | 0.95 | 0.92 |
| Real Notes | 0.96 | 0.93 | 0.94 |

Manual review of misclassified images revealed two patterns:
- Fake notes misclassified as real were often high-quality counterfeits with sharp printing, visually convincing even to a human eye.
- Real notes misclassified as fake were often worn, faded, or photographed in poor lighting.

### 6.2 Problem #1: Out-of-Distribution Inputs (Not A Note)
**3-class model performance (exact validation results):**

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Fake Notes | 0.83 | 1.00 | 0.91 |
| Not A Note | 1.00 | 0.97 | 0.98 |
| Real Notes | 0.99 | 0.86 | 0.92 |

**Overall Accuracy: 92.35%** (27/353 misclassified)

Notably, the model achieves perfect recall (100%) on Fake Notes — it never misses
an actual counterfeit in the validation set — at the cost of lower precision
(83%), meaning some genuine notes are occasionally flagged as fake. This is a
safety-conscious trade-off appropriate for a currency verification tool: a false
alarm on a real note is far less costly than a missed counterfeit.

### 6.3 Problem #2: Novelty Notes with Disclaimer Text

**Discovery:** Further testing found that "Bachon Ka Khel" notes — officially printed children's play-money notes designed to closely visually mimic genuine currency, distinguished only by small Urdu-script disclaimer text and an all-zero serial number — were misclassified as "Real Note."

**Root cause:** A CNN judges visual style (color, texture, layout), not text meaning. The disclaimer text is the *only* feature distinguishing these notes from genuine currency, and it is invisible to a purely visual classifier.

**Fix:** An OCR-based safety-net layer was added, running independently alongside the CNN:
1. Tesseract OCR (English + Urdu) extracts any visible text from the image
2. The extracted text is checked against a list of known fake/novelty indicator phrases, in both Roman and Urdu script (e.g., "بچوں کا کھیل", "specimen," "replica")
3. A check for all-zero serial numbers (a common specimen/novelty marker) is also run
4. If any match is found, **the CNN's decision is overridden to "Fake Notes,"** since explicit text evidence is more reliable than visual pattern-matching for this specific failure case

This is a two-stage verification design: CNN handles visual authenticity classification, OCR provides a targeted textual safety check — rather than asking either system to solve the whole problem alone.

### 6.4 Grad-CAM Explainability Findings

Grad-CAM analysis across multiple samples revealed that model attention correlates with how much of the frame the note occupies:
- When notes fill most of the image, activation concentrates on legitimate security features (watermark, denomination).
- When notes are smaller or photographed at odd angles with more visible background, activation leaks significantly into background elements (lighting, table surfaces, fabric).

This directly motivated the manual crop-before-analyze feature in the dashboard, which lets users frame just the note before classification, reducing background influence.

## 7. Dashboard Features

- **Upload or live camera input** — works on both desktop and phone browsers
- **Manual crop tool** — user frames just the note before analysis, addressing the background-bias finding above
- **3-class prediction** — Real / Fake / Not A Note, with full confidence breakdown
- **Confidence-threshold warning** — predictions below 85% confidence trigger an explicit caution message recommending official verification
- **OCR text-verification safety net** — flags known fake/novelty phrases and all-zero serial numbers, overriding the CNN when triggered
- **Grad-CAM visualization** — shows what the model focused on, skipped for "Not A Note" predictions where it isn't meaningful

## 8. Limitations

- **Photo-based detection has a fundamental ceiling.** This is not unique to this model — it reflects a limitation of any photo-based classification approach. Real-world currency verification relies on physical security layers unavailable to a camera (UV-reactive ink, embedded security threads, watermarks visible under light, raised print texture). This tool is a first-pass screening aid, not a replacement for official multi-layered verification.
- **The "Not A Note" class has limited training diversity.** Trained primarily on Fashion-MNIST and a small supplementary set, it does not generalize to all possible non-currency images — testing showed a visually complex, colorful image (a movie poster) was still misclassified as "Fake Notes" rather than correctly rejected. A larger, more diverse negative-class dataset would improve this.
- **OCR accuracy depends on image quality.** Angle, lighting, and print size all affect text extraction reliability; the OCR safety net will not catch every novelty note, especially in poor-quality photos.
- **Background influence, while reduced by manual cropping, is not fully eliminated** for notes that still include significant surrounding context.

## 9. Future Improvements

- [ ] Expand the "Not A Note" dataset with more diverse real-world non-currency images
- [ ] Automatic note detection/cropping (replacing manual cropping) via a trained object detector
- [ ] Combine CNN confidence + OCR results + specific security-feature detection into a single weighted authenticity score
- [ ] Support multiple denominations with per-denomination validation
- [ ] Deploy as a mobile app with on-device inference

## 10. Tech Stack

Python · TensorFlow/Keras · OpenCV · Tesseract OCR (pytesseract) · Streamlit · streamlit-cropper · scikit-learn · Matplotlib/Seaborn

---

**Author:** Malik Faisal Mukhtar
**Intern ID:** ZYNVEX-CERT-1130
**Program:** AI/ML Internship — Capstone Project