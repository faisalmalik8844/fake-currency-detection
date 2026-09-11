# EDA Report — PKR Fake Currency Dataset

## 1. Dataset Overview
- **Original dataset:** 1,600 images (Real Notes, Fake Notes)
- **Final dataset (after Section 6.2 fix, see main README):** 1,765 images across 3 classes
- Image format(s): mixed (jpg/png)
- Source: Kaggle — "Real and Fake Currency Pakistanis Dataset", supplemented with
  ~165 "Not A Note" images (Fashion-MNIST samples) added later after an
  out-of-distribution failure was discovered during live testing (see main README, Section 6.2)

## 2. Class Distribution

| Class | Image Count | Percentage |
|---|---|---|
| Real Notes | 950 | 53.8% |
| Fake Notes | 650 | 36.8% |
| Not A Note | 165 | 9.4% |

> Mild imbalance across all three classes. The Real/Fake ratio (~60/40) was
> present from the original dataset; "Not A Note" was added afterward as a
> deliberately smaller supporting class, sufficient to teach the model to
> reject non-currency inputs without needing to match the size of the two
> main classes.

## 3. Image Properties (Original Real/Fake Dataset)
- Width range: 300px – 4160px (avg ~2050px)
- Height range: 168px – 4160px (avg ~1975px)
- Significant size variation across the dataset, requiring standardized
  resizing (224x224) before training. Since most images are much larger
  than the target size, resizing mostly downsizes images, preserving
  quality well. A few smaller images (near 300px) may lose some detail
  when resized, which is noted as a limitation.
- "Not A Note" images (Fashion-MNIST) are natively small (28x28 grayscale)
  and were upscaled to 224x224 RGB to match — a quality trade-off accepted
  given the time constraints of adding this class late in the project
  (see main README, Limitations).

## 4. Visual Inspection

Sample real vs. fake images were plotted side by side (see `notebooks/01_eda.ipynb`).
Images are casual, real-world photos with varied backgrounds, angles, and lighting
rather than clean scans — this increases realism for deployment but risks
background-based spurious correlations, which was later confirmed via Grad-CAM
analysis (see main README, Section 6.4).

**Note:** the misclassification patterns below were identified during the
*evaluation* phase (after training the first model), not during this initial
EDA pass — included here for completeness since they stem from the same
visual inspection process.

Observations:
- Manual inspection of misclassified images revealed two dominant patterns:
  (1) several "fake" notes misclassified as real were high-quality counterfeits
  with sharp printing that visually resemble genuine notes even to human
  inspection, suggesting possible dataset labeling ambiguity or genuinely
  sophisticated fakes; (2) several "real" notes misclassified as fake were
  worn, faded, or photographed in poor lighting, suggesting the model may
  associate certain texture degradation with counterfeit characteristics.
- This indicates the model's errors are not random, but correlate with note
  condition and photo quality — a limitation with real-world deployment
  implications, further explored via Grad-CAM (see main README, Section 6.4).

## 5. Data Quality Issues Found
- Corrupted images: 0 (verified via OpenCV load check across all 1,600 original images)
- Duplicate images: not checked
- Mislabeled images: not manually verified, though see the labeling-ambiguity
  observation in Section 4 above regarding high-quality counterfeits

## 6. Preprocessing Decisions
- Resize target: 224x224 (to match MobileNetV2 input)
- Normalization: pixel values scaled to [0, 1]
- Augmentation applied: rotation (±10°), brightness (±20%), zoom (±10%)
- Augmentation NOT used: horizontal/vertical flips (currency notes have
  a fixed, meaningful orientation — flipping would create unrealistic
  training examples)
- Class imbalance: not corrected via class weights in the final model:
  the 3-class evaluation results (see main README, Section 6.2) show the
  model still achieves strong per-class recall despite the imbalance,
  though this remains a possible future improvement

## 7. Train/Validation Split
- Split ratio: 80/20, applied consistently across all three classes
- Stratified: yes — preserves each class's ratio in both splits
- Final split sizes: 1,412 train / 353 validation (see main README for
  exact per-class evaluation results on this validation set)