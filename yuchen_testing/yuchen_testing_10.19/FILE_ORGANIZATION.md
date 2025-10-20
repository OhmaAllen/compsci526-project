# Project File Organization Guide

## 📊 Training Status (Current)

### Active Training Processes
- **LSTM** (PID: 70764): Epoch 7/25 (28%), Val AUROC: 0.6089 ✅
- **Transformer** (PID: 76300): Epoch 3/25 (12%), Val AUROC: 0.5982 ✅

---

## ✅ KEEP - Essential Production Files

### Data Pipeline (src/data/)
```
01_cohort_selection.py          ← Stage 1: Select eligible admissions
02_label_generation.py          ← Stage 2: Generate 30-day readmission labels
03_missingness_report.py        ← Stage 3: Analyze data quality
04_feature_extraction.py        ← Stage 4: Extract time-series lab values
05_static_features.py           ← Stage 5: Create static features
06_train_val_test_split.py      ← Stage 6: Split data into train/val/test
run_pipeline.py                 ← Master script to run all stages
```

### Model Code (src/models/)
```
dataloader.py                   ← DataLoader for time-series + static features ⭐
lstm_model.py                   ← LSTM architecture (22K params) ⭐
transformer_model_simple.py     ← Stable Transformer (56K params) ⭐
train_lstm_fixed.py             ← LSTM training script (RUNNING) ⭐⭐
train_transformer_stable.py     ← Transformer training script (RUNNING) ⭐⭐
train_baselines.py              ← Baseline models (LR, RF, XGB) ⭐
```

### Documentation
```
TRANSFORMER_FIX_SUMMARY.md      ← Technical documentation of Transformer fixes
TRAINING_STATUS_FINAL.md        ← Current training status and monitoring
FILE_ORGANIZATION.md            ← This file
README.md                       ← Project overview
```

---

## ❌ DELETE - Obsolete/Debug Files

### Old/Failed Transformer Versions
```
transformer_model.py            ← DELETE: Original with NaN issues
transformer_model_fixed.py      ← DELETE: Partial fix, still has NaN
train_transformer.py            ← DELETE: Uses unstable model
```

### Debug/Test Scripts
```
test_transformer_real.py        ← DELETE: Was for debugging
diagnose_transformer.py         ← DELETE: Diagnostic tool (no longer needed)
test_training_debug.py          ← DELETE: Debug script
train_deep_learning.py          ← DELETE: Failed combined training script
train_simple.py                 ← DELETE: Old training script
04_feature_extraction_test.py   ← DELETE: Test version
```

### Duplicate/Old Preprocessing
```
preprocessing/generate_readmission_features_v1.py    ← DELETE: Old version
jiaqi_preprocessing/generate_readmission_features_v1.py  ← DELETE: Duplicate
clean_testing/LG+RM+GB.py       ← DELETE: Old baseline test
```

---

## 📁 Recommended Directory Structure

```
git_proj/
├── README.md                                    ⭐ Keep
├── milestone_report.tex                         ⭐ NEW (to create)
│
├── yuchen_testing/yuchen_testing_10.19/
│   ├── FILE_ORGANIZATION.md                     ⭐ Keep
│   ├── TRANSFORMER_FIX_SUMMARY.md               ⭐ Keep
│   ├── TRAINING_STATUS_FINAL.md                 ⭐ Keep
│   │
│   ├── src/
│   │   ├── data/                                ⭐ Keep entire folder
│   │   │   ├── 01_cohort_selection.py
│   │   │   ├── 02_label_generation.py
│   │   │   ├── 03_missingness_report.py
│   │   │   ├── 04_feature_extraction.py
│   │   │   ├── 05_static_features.py
│   │   │   ├── 06_train_val_test_split.py
│   │   │   ├── run_pipeline.py
│   │   │   └── mimic_data/                      (Data files)
│   │   │
│   │   └── models/                              ⭐ Keep essential files only
│   │       ├── dataloader.py                    ⭐⭐
│   │       ├── lstm_model.py                    ⭐⭐
│   │       ├── transformer_model_simple.py      ⭐⭐
│   │       ├── train_lstm_fixed.py              ⭐⭐
│   │       ├── train_transformer_stable.py      ⭐⭐
│   │       └── train_baselines.py               ⭐⭐
│   │
│   └── results/                                 ⭐ Keep
│       └── models/
│           ├── baselines/
│           │   └── baseline_results.json
│           ├── lstm/
│           │   ├── lstm_best.pt                 (After training)
│           │   └── lstm_results.json            (After training)
│           └── transformer/
│               ├── transformer_best.pt          (After training)
│               └── transformer_results.json     (After training)
│
└── [Other folders to delete]
    ├── preprocessing/                           ❌ DELETE
    ├── jiaqi_preprocessing/                     ❌ DELETE
    ├── clean_testing/                           ❌ DELETE
    ├── XiChen_testing/                          ❌ DELETE (if not used)
    └── visualization/                           ⚠️ REVIEW (keep if has useful plots)
```

---

## 🗑️ Files to Delete (Commands)

```bash
# Navigate to project root
cd /Users/yuchenzhou/Documents/duke/compsci526/final_proj/git_proj

# Delete old transformer versions
rm yuchen_testing/yuchen_testing_10.19/src/models/transformer_model.py
rm yuchen_testing/yuchen_testing_10.19/src/models/transformer_model_fixed.py
rm yuchen_testing/yuchen_testing_10.19/src/models/train_transformer.py

# Delete debug/test scripts
rm yuchen_testing/yuchen_testing_10.19/src/models/test_transformer_real.py
rm yuchen_testing/yuchen_testing_10.19/src/models/diagnose_transformer.py
rm yuchen_testing/yuchen_testing_10.19/src/models/test_training_debug.py
rm yuchen_testing/yuchen_testing_10.19/src/models/train_deep_learning.py
rm yuchen_testing/yuchen_testing_10.19/src/models/train_simple.py

# Delete old preprocessing files
rm -rf preprocessing/
rm -rf jiaqi_preprocessing/
rm -rf clean_testing/

# Delete test version
rm yuchen_testing/yuchen_testing_10.19/src/data/04_feature_extraction_test.py

# Optional: Delete other testing folders (check first!)
# rm -rf XiChen_testing/
```

---

## 📋 File Purposes Summary

### Essential Production Files (12 files)

| File | Purpose | Lines | Status |
|------|---------|-------|--------|
| **Data Pipeline (7 files)** |
| 01_cohort_selection.py | Select admissions from ICU stays | ~200 | ✅ |
| 02_label_generation.py | Generate readmission labels | ~150 | ✅ |
| 03_missingness_report.py | Analyze data quality | ~100 | ✅ |
| 04_feature_extraction.py | Extract time-series features | ~300 | ✅ |
| 05_static_features.py | Create static features | ~150 | ✅ |
| 06_train_val_test_split.py | Split into train/val/test | ~100 | ✅ |
| run_pipeline.py | Master orchestration script | ~200 | ✅ |
| **Model Code (6 files)** |
| dataloader.py | Load time-series + static data | ~180 | ✅ |
| lstm_model.py | LSTM architecture | ~200 | ✅ |
| transformer_model_simple.py | Stable Transformer | ~350 | ✅ |
| train_lstm_fixed.py | LSTM training (ACTIVE) | ~200 | 🏃 |
| train_transformer_stable.py | Transformer training (ACTIVE) | ~220 | 🏃 |
| train_baselines.py | Baseline models | ~250 | ✅ |

### Files to Delete (11+ files)

| File | Reason to Delete |
|------|------------------|
| transformer_model.py | Original with NaN issues |
| transformer_model_fixed.py | Partial fix, still unstable |
| train_transformer.py | Uses unstable transformer |
| test_transformer_real.py | Debug script |
| diagnose_transformer.py | Diagnostic tool |
| test_training_debug.py | Debug script |
| train_deep_learning.py | Failed combined training |
| train_simple.py | Old training script |
| 04_feature_extraction_test.py | Test version |
| preprocessing/ folder | Old duplicate code |
| jiaqi_preprocessing/ folder | Old duplicate code |
| clean_testing/ folder | Old baseline tests |

---

## 🎯 Key Insights

### Code Quality
- **12 essential files** covering full pipeline + models
- **11+ obsolete files** can be safely deleted
- Total reduction: ~50% fewer files
- All production code is well-documented

### Model Status
- **Baseline models**: Complete (RF: 0.610 AUROC)
- **LSTM**: Training epoch 7/25, current best 0.6089
- **Transformer**: Training epoch 3/25, current best 0.5982
- Both models running stably without NaN issues

### Project Cleanliness
After cleanup:
- Clear separation of data pipeline vs models
- No duplicate/obsolete code
- Only production-ready files remain
- Easy to understand and maintain

---

## 📝 Post-Cleanup Checklist

- [ ] Backup current state (git commit)
- [ ] Delete obsolete files (use commands above)
- [ ] Test that essential files still work
- [ ] Update GitHub repository
- [ ] Create milestone report (LaTeX)
- [ ] Wait for training completion
- [ ] Run final evaluation
- [ ] Submit milestone report

---

**Last Updated**: October 20, 2025, 14:00
**Status**: Both models training successfully ✅
