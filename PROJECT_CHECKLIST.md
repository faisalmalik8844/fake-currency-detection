# Project Checklist — Fake Currency Detection

**Target completion:** Sept 18, 2026 | **Hard deadline:** Sept 21, 2026

## Phase 1: Finish GenAI labs (Sept 3–8)
- [ ] Complete remaining Week 4 labs (LLMs, prompt engineering, RAG, agents, APIs)
- [ ] Download PKR dataset(s) from Kaggle, just skim structure (no coding yet)

## Phase 2: Project Build (Sept 9–18)

### Sept 9 — Setup + EDA
- [ ] Download dataset into `data/raw/`
- [ ] Run `inspect_dataset()` and `check_image_integrity()` from `preprocessing.py`
- [ ] Fill in `notebooks/01_eda.ipynb` — class balance, sample images, resolution check
- [ ] Fill in `reports/EDA_report.md`
- [ ] Initial `git init` + first commit

### Sept 10 — Data Pipeline
- [ ] Finalize `CLASS_NAMES` and `IMG_SIZE` in `preprocessing.py` to match actual dataset
- [ ] Test `get_data_generators()` produces correct batches
- [ ] Move/organize images into `data/processed/` (train-ready structure)
- [ ] Fill in `notebooks/02_preprocessing.ipynb`

### Sept 11 — Baseline CNN
- [ ] Run `train.py --model baseline`
- [ ] Record baseline accuracy/loss
- [ ] Fill in `notebooks/03_baseline_cnn.ipynb`

### Sept 12 — Transfer Learning
- [ ] Confirm `LAST_CONV_LAYER` name via `model.summary()`
- [ ] Run `train.py --model transfer`
- [ ] Compare against baseline — write down why one performs better
- [ ] Fill in `notebooks/04_transfer_learning.ipynb`

### Sept 13 — Tuning
- [ ] Try 2-3 dropout/architecture variations
- [ ] Watch for overfitting (train vs val accuracy gap)
- [ ] If needed: fine-tune base model with `--fine-tune` flag

### Sept 14 — Full Evaluation
- [ ] Run `evaluate.py --model-path <best_model>`
- [ ] Review confusion matrix, ROC-AUC, misclassified images
- [ ] Fill in `notebooks/05_evaluation.ipynb`
- [ ] Write "Limitations" section in README honestly

### Sept 15 — Grad-CAM
- [ ] Confirm Grad-CAM heatmaps look reasonable (focus on note details, not background)
- [ ] Generate 4-6 example heatmaps (real + fake) for the README
- [ ] Fill in `notebooks/06_gradcam.ipynb`

### Sept 16 — Dashboard
- [ ] Update `MODEL_PATH` and `LAST_CONV_LAYER` in `app/streamlit_app.py`
- [ ] Test upload flow end-to-end
- [ ] Add input validation (reject non-images, handle errors gracefully)

### Sept 17 — Polish + Documentation
- [ ] Finish full README (results table, Grad-CAM images, limitations, future work)
- [ ] Clean up notebooks (remove dead code, add markdown explanations)
- [ ] Make sure `requirements.txt` is accurate (`pip freeze` check)

### Sept 18 — Final Submission Prep
- [ ] Record short demo video/GIF of the dashboard in action
- [ ] Final GitHub push — check repo looks clean and professional
- [ ] Prepare presentation slides / talking points
- [ ] Buffer time: Sept 19–21 for last-minute fixes

## Notes / Blockers
_(use this space to jot down anything that comes up during the build)_
