# Project Setup Summary
## 30-Day Hospital Readmission Prediction using MIMIC-IV

**Date Created:** October 20, 2025  
**Status:** Infrastructure Complete - Ready for Data Processing and Model Development

---

## 📋 Project Overview

This project implements a comprehensive machine learning pipeline for predicting 30-day hospital readmissions using the MIMIC-IV database. The goal is to compare multiple models (Logistic Regression, Random Forest, XGBoost, LSTM, Transformer) and determine which architecture performs best for this clinical prediction task.

**Key Research Question:** Can Transformer-based models significantly outperform LSTM and traditional ML approaches in predicting 30-day readmissions using temporal clinical data?

---

## ✅ Completed Components

### 1. Documentation (100% Complete)

| File | Status | Description |
|------|--------|-------------|
| `README.md` | ✅ Complete | Full project documentation with setup instructions |
| `QUICKSTART.md` | ✅ Complete | Step-by-step guide to run the pipeline in 30 minutes |
| `docs/PROBLEM_STATEMENT.md` | ✅ Complete | Research question, hypotheses, evaluation metrics, timeline |

### 2. Configuration (100% Complete)

| File | Status | Description |
|------|--------|-------------|
| `config.yaml` | ✅ Complete | Comprehensive configuration with all parameters (cohort criteria, features, models, evaluation) |
| `requirements.txt` | ✅ Complete | Python dependencies with pinned versions |

### 3. Data Processing Scripts (50% Complete)

| File | Status | Description |
|------|--------|-------------|
| `src/data/01_cohort_selection.py` | ✅ Complete | Filters admissions (adults, valid times, exclude deaths/AMA) → `cohort.csv` |
| `src/data/02_label_generation.py` | ✅ Complete | Generates 30-day readmission labels with competing risk handling → `readmission_labels.csv` |
| `src/data/03_missingness_report.py` | ✅ Complete | Analyzes feature coverage and missing data → `missingness_report.csv` + plots |
| `src/data/04_feature_extraction.py` | ⏳ **TODO** | Extract time-series features (vitals, labs) from last 48h → `time_series_tensors/` |
| `src/data/05_static_features.py` | ⏳ **TODO** | Generate static features and comorbidity scores (Charlson, Elixhauser) → `static_features.csv` |
| `src/data/06_train_val_test_split.py` | ⏳ **TODO** | Patient-level split (70/15/15) → `split_train.txt`, `split_val.txt`, `split_test.txt` |
| `src/data/run_pipeline.py` | ✅ Complete | Master script to run all data processing steps |

### 4. Directory Structure (100% Complete)

```
yuchen_testing_10.19/
├── docs/                      ✅ Created
├── src/
│   ├── data/                  ✅ Created (3/6 scripts done)
│   ├── models/                ✅ Created (empty)
│   ├── experiments/           ✅ Created (empty)
│   └── utils/                 ✅ Created (empty)
├── mimic_data/
│   └── processed_data/        ✅ Created (ready for outputs)
├── experiments/
│   └── configs/               ✅ Created (empty)
└── results/
    ├── plots/                 ✅ Created (empty)
    └── interpretation/        ✅ Created (empty)
```

---

## ⏳ Remaining Tasks

### Priority 1: Complete Data Processing Pipeline (Critical)

#### Task 4: Time-Series Feature Extraction
**File:** `src/data/04_feature_extraction.py`

**Requirements:**
- Read `chartevents.csv` and `labevents.csv`
- Filter to last 48 hours before discharge (configurable)
- Map itemids to clinical features using config (heart rate, BP, labs, etc.)
- Temporal binning: hourly aggregation (mean/median)
- Create mask tensors for missing values
- Save as NPZ files: `{hadm_id}.npz` containing:
  - `X`: Time-series array (48, num_features)
  - `mask`: Boolean array (48, num_features)
  - `times`: Relative time from discharge
  
**Estimated Effort:** 4-6 hours (most complex script)

#### Task 5: Static Features and Comorbidity Scores
**File:** `src/data/05_static_features.py`

**Requirements:**
- Extract demographics: age, gender from cohort
- Calculate comorbidity indices:
  - Charlson Comorbidity Index (CCI) from ICD codes
  - Elixhauser score (optional)
- Add admission features: admission_type, length_of_stay
- Merge medications (antibiotics, vasopressors, diuretics) as binary flags
- Output: `static_features.csv` with columns:
  - `hadm_id`, `subject_id`, `age`, `gender`, `charlson_score`, `elixhauser_score`, `los_hours`, etc.

**Estimated Effort:** 2-3 hours

#### Task 6: Train/Val/Test Split
**File:** `src/data/06_train_val_test_split.py`

**Requirements:**
- Load valid labels
- Split by `subject_id` (patient-level, no leakage)
- Stratify by `readmit_30d` label
- Proportions: 70% train, 15% val, 15% test
- Save lists of `hadm_id` to:
  - `split_train.txt`
  - `split_val.txt`
  - `split_test.txt`
- Print statistics (size, readmission rate per split)

**Estimated Effort:** 1-2 hours

---

### Priority 2: Model Implementations (Next Phase)

#### Baseline Models
**Files to create:**
- `src/models/baseline_lr.py` - Logistic Regression (static features only)
- `src/models/baseline_tree.py` - Random Forest + XGBoost (static + aggregated time-series)
- `src/experiments/train_baselines.py` - Training script for all baselines

**Requirements:**
- Load static features and labels
- For tree models: aggregate time-series (mean, std, min, max of last 48h)
- Cross-validation on training set
- Hyperparameter tuning
- Save models and predictions
- Compute metrics: AUROC, AUPRC, F1, Precision, Recall

**Estimated Effort:** 3-4 hours

#### Deep Learning Models
**Files to create:**
- `src/models/lstm_model.py` - LSTM architecture
- `src/models/transformer_model.py` - Transformer architecture
- `src/models/gru_d.py` - GRU with Decay (mask-aware)
- `src/experiments/train_lstm.py` - LSTM training script
- `src/experiments/train_transformer.py` - Transformer training script

**Requirements:**
- PyTorch implementations
- DataLoader for time-series + static features
- Handle variable-length sequences (padding/masking)
- Early stopping, learning rate scheduling
- TensorBoard logging
- Save best model checkpoints

**Estimated Effort:** 6-8 hours

---

### Priority 3: Evaluation and Analysis

#### Utilities
**Files to create:**
- `src/utils/metrics.py` - Metric calculation functions (AUROC, AUPRC, calibration)
- `src/utils/visualization.py` - ROC/PR curves, calibration plots
- `src/utils/statistical_tests.py` - Bootstrap CI, DeLong test
- `src/utils/config.py` - Config loading helpers

**Estimated Effort:** 2-3 hours

#### Evaluation Script
**Files to create:**
- `src/experiments/evaluate_models.py` - Comprehensive evaluation on test set
- `src/experiments/fairness_analysis.py` - Subgroup performance analysis
- `src/utils/generate_report.py` - Auto-generate markdown report

**Requirements:**
- Load all trained models
- Compute all metrics on test set
- Bootstrap 95% confidence intervals
- DeLong test for model comparison
- Calibration analysis (Brier score, ECE, reliability plots)
- Fairness metrics by gender/age/race
- Generate `results/evaluation_summary.md`

**Estimated Effort:** 3-4 hours

---

### Priority 4: Interpretability

**Files to create:**
- `src/experiments/shap_analysis.py` - SHAP values for tree models
- `src/experiments/attention_visualization.py` - Attention weights for Transformer
- `src/experiments/feature_importance.py` - Aggregate feature importance

**Estimated Effort:** 2-3 hours

---

### Priority 5: Ablation Studies

**Files to create:**
- `src/experiments/ablation_time_windows.py` - Test 24h, 48h, 72h windows
- `src/experiments/ablation_features.py` - Test different feature combinations
- `src/experiments/subgroup_analysis.py` - Performance by age/diagnosis

**Estimated Effort:** 3-4 hours

---

## 📊 Implementation Timeline (Suggested)

| Week | Tasks | Deliverables |
|------|-------|--------------|
| **Week 1** | Complete data processing (Tasks 4-6) | Processed datasets ready for modeling |
| **Week 2** | Baseline models (LR, RF, XGB) | Initial performance benchmarks |
| **Week 3** | Deep learning models (LSTM) | LSTM results, comparison with baselines |
| **Week 4** | Transformer model | Complete model comparison |
| **Week 5** | Evaluation, calibration, fairness | `evaluation_summary.md`, statistical tests |
| **Week 6** | Interpretability, ablation studies | SHAP plots, attention maps, ablation results |
| **Week 7** | Final report, documentation | Thesis-ready documentation |

**Total Estimated Effort:** 30-40 hours of focused development

---

## 🚀 How to Proceed

### Immediate Next Steps (Today):

1. **Review Configuration**
   ```bash
   # Make sure MIMIC-IV path is correct
   nano config.yaml
   # Update data.raw_data_path if needed
   ```

2. **Test Completed Scripts**
   ```bash
   cd src/data
   # Test cohort selection (should take 2-5 minutes)
   python 01_cohort_selection.py
   
   # Test label generation (should take 1-2 minutes)
   python 02_label_generation.py
   
   # Test missingness report (may take 10-20 minutes due to sampling)
   python 03_missingness_report.py
   ```

3. **Implement Remaining Data Scripts** (Tasks 4-6)
   - Start with `04_feature_extraction.py` (most critical and complex)
   - Then `05_static_features.py`
   - Finally `06_train_val_test_split.py`

### This Week:

4. **Complete Data Pipeline**
   - Finish tasks 4-6
   - Run `python run_pipeline.py --all`
   - Verify all outputs in `mimic_data/processed_data/`

5. **Start Baseline Models**
   - Implement logistic regression
   - Implement random forest
   - Create training script

---

## 📝 Key Design Decisions Made

1. **Time Window:** 48 hours before discharge (can ablate 24h, 72h later)
2. **Temporal Resolution:** Hourly bins (1-hour aggregation)
3. **Missing Data Strategy:** Mask-aware (preserve masks for DL models)
4. **Cohort Criteria:** Adults (≥18), alive at discharge, LOS ≥24h, valid admission types
5. **Label Handling:** Exclude last admissions, track competing risks (death within 30d)
6. **Split Strategy:** Patient-level (70/15/15), stratified by readmission
7. **Features:**
   - **Vitals:** HR, BP, RR, SpO2, Temp
   - **Labs:** WBC, Hgb, Plt, Na, K, Cr, BUN, Glucose, Lactate
   - **Static:** Age, gender, Charlson score, LOS, admission type
8. **Evaluation:** AUROC (primary), AUPRC, F1, Recall@Top10%, Brier score, 95% CI

---

## 📚 Reference Materials Created

1. **PROBLEM_STATEMENT.md** - Scientific framing of the research
2. **README.md** - Complete technical documentation
3. **QUICKSTART.md** - Hands-on tutorial for running the pipeline
4. **config.yaml** - Centralized configuration (edit once, use everywhere)
5. **This file (PROJECT_SUMMARY.md)** - Implementation roadmap

---

## ✨ What Makes This Pipeline Good

1. **Reproducible:** Fixed seeds, version control, documented configs
2. **Clinical Validity:** Proper cohort selection, competing risk handling
3. **Modular:** Each step is independent, can re-run individually
4. **Configurable:** Single config file for all experiments
5. **Scalable:** Chunk processing for large datasets
6. **Well-Documented:** Clear documentation at every level
7. **Research-Grade:** Follows best practices for medical ML research

---

## 🎯 Success Criteria Checklist

Before submitting your final project, ensure:

- [ ] All data processing scripts complete and tested
- [ ] Cohort size is 40,000-100,000 admissions
- [ ] Readmission rate is 15-20%
- [ ] No data leakage between train/val/test
- [ ] All baseline models trained and evaluated
- [ ] At least one deep learning model (LSTM or Transformer) working
- [ ] AUROC ≥ 0.75 on test set
- [ ] 95% confidence intervals reported
- [ ] Model comparison with statistical significance testing
- [ ] Calibration analysis (Brier score, reliability plots)
- [ ] Fairness analysis across demographic groups
- [ ] Interpretability analysis (SHAP or attention)
- [ ] At least one ablation study (time window or features)
- [ ] Complete evaluation_summary.md generated
- [ ] All code committed to git with clear commit messages

---

## 💡 Tips for Success

1. **Start Simple:** Get logistic regression working before deep learning
2. **Validate Early:** Check data quality after each pipeline step
3. **Debug with Samples:** Test on 1,000 admissions before running full dataset
4. **Monitor Resources:** Time-series extraction may need 30-60 minutes
5. **Version Everything:** Git commit after each working script
6. **Document Decisions:** Add comments explaining "why" not just "what"
7. **Ask for Help:** If stuck >2 hours, consult documentation or ask

---

**Project Status:** 🟡 In Progress - Foundation Complete, Ready for Core Development

**Next Milestone:** Complete data processing pipeline (Tasks 4-6) by end of week

**Contact:** Yuchen Zhou | Duke CS526 | Fall 2025

---

*Last Updated: October 20, 2025*
