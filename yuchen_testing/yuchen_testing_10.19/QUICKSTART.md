# QUICKSTART GUIDE
# 30-Day Hospital Readmission Prediction Pipeline

This guide will help you get started with the complete pipeline in 30 minutes.

## Prerequisites

1. **MIMIC-IV Access**: You need credentialed access to MIMIC-IV v3.1
   - PhysioNet account with completed training
   - Data downloaded to: `/Users/yuchenzhou/Documents/duke/compsci526/final_proj/mimic-iv-3.1`

2. **Python Environment**: Python 3.9+

3. **Computational Resources**:
   - ~50GB disk space for processed data
   - 16GB+ RAM recommended
   - GPU optional but recommended for deep learning models

---

## Step-by-Step Setup

### 1. Environment Setup (5 minutes)

```bash
# Navigate to project directory
cd /Users/yuchenzhou/Documents/duke/compsci526/final_proj/git_proj/yuchen_testing/yuchen_testing_10.19

# Create virtual environment
conda create -n readmit_pred python=3.9 -y
conda activate readmit_pred

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import pandas, numpy, torch, sklearn, yaml; print('✓ All packages installed')"
```

### 2. Configuration Check (2 minutes)

```bash
# Review configuration
cat config.yaml

# Key settings to verify:
# - data.raw_data_path: Should point to your MIMIC-IV directory
# - features.time_window_hours: Default is 48 (last 48h before discharge)
# - cohort.min_age: Default is 18 (adult patients only)
```

**Important**: Edit `config.yaml` if your MIMIC-IV data is in a different location!

### 3. Run Data Processing Pipeline (20-60 minutes)

The pipeline has 6 main steps. You can run them all at once or individually.

#### Option A: Run Complete Pipeline

```bash
cd src/data
python run_pipeline.py --all
```

This will execute:
1. Cohort selection (filter valid admissions)
2. Label generation (30-day readmission labels)
3. Missingness report (data quality analysis)
4. Feature extraction (time-series + static features)
5. Static features (comorbidity scores)
6. Train/val/test split (patient-level split)

#### Option B: Run Steps Individually

```bash
# Step 1: Create cohort (adults, valid admissions, exclude deaths)
python 01_cohort_selection.py
# Expected output: mimic_data/processed_data/cohort.csv
# ~50,000-100,000 admissions after filtering

# Step 2: Generate readmission labels
python 02_label_generation.py
# Expected output: mimic_data/processed_data/readmission_labels.csv
# Expect ~15-20% readmission rate

# Step 3: Missingness analysis
python 03_missingness_report.py
# Expected outputs: 
#   - missingness_report.csv (coverage statistics)
#   - missingness_plots.png (visualizations)

# Step 4: Extract time-series features (TAKES LONGEST - 30-60 min)
python 04_feature_extraction.py
# Expected output: time_series_tensors/ (NPZ files per admission)

# Step 5: Generate static features and comorbidity scores
python 05_static_features.py
# Expected output: static_features.csv

# Step 6: Create train/val/test split (patient-level)
python 06_train_val_test_split.py
# Expected outputs: split_train.txt, split_val.txt, split_test.txt
```

### 4. Verify Data Processing (2 minutes)

```bash
# Check that all files were created
cd ../../mimic_data/processed_data
ls -lh

# Expected files:
# - cohort.csv
# - readmission_labels.csv
# - readmission_labels_valid.csv
# - missingness_report.csv
# - itemid_map.csv
# - static_features.csv
# - split_train.txt
# - split_val.txt
# - split_test.txt
# - time_series_tensors/ (directory with many .npz files)

# Quick validation
python -c "
import pandas as pd
labels = pd.read_csv('readmission_labels_valid.csv')
print(f'Total admissions: {len(labels):,}')
print(f'Readmission rate: {labels[\"readmit_30d\"].mean():.1%}')
print(f'Unique patients: {labels[\"subject_id\"].nunique():,}')
"
```

---

## Next Steps: Model Training

### Train Baseline Models (15 minutes)

```bash
cd ../../src/experiments

# Train logistic regression, random forest, XGBoost
python train_baselines.py --config ../experiments/configs/baseline_config.json

# Expected output:
# - Trained models saved to experiments/models/
# - Metrics saved to results/baseline_metrics.json
# - ROC/PR curves saved to results/plots/
```

### Train Deep Learning Models (30-60 minutes each)

```bash
# LSTM model
python train_lstm.py --config ../experiments/configs/lstm_config.json

# Transformer model
python train_transformer.py --config ../experiments/configs/transformer_config.json
```

### Generate Evaluation Report

```bash
cd ../utils
python generate_report.py --output ../../results/evaluation_summary.md
```

---

## Common Issues and Solutions

### Issue 1: MIMIC-IV path not found
**Error**: `FileNotFoundError: .../mimic-iv-3.1/hosp/admissions.csv`

**Solution**: Edit `config.yaml` and update `data.raw_data_path` to your MIMIC-IV location

### Issue 2: Out of memory during feature extraction
**Error**: `MemoryError` or system freezes

**Solution**: 
- Process admissions in batches (modify `04_feature_extraction.py`)
- Reduce `sample_size` in missingness analysis
- Use a machine with more RAM

### Issue 3: Missing package
**Error**: `ModuleNotFoundError: No module named 'yaml'`

**Solution**: 
```bash
pip install pyyaml
# Or reinstall all dependencies:
pip install -r requirements.txt
```

### Issue 4: Very low readmission rate
**Problem**: Readmission rate < 5% (expected is 15-20%)

**Solution**: Check these settings in `config.yaml`:
- `labels.readmission_days: 30` (not 7 or 14)
- `labels.exclude_last_admission: true` (should be true)
- `cohort.exclude_deaths: true` (should be true)

---

## Project Structure Overview

```
yuchen_testing_10.19/
├── config.yaml              # Main configuration file - EDIT THIS FIRST
├── requirements.txt         # Python dependencies
├── README.md               # Full documentation
│
├── docs/
│   └── PROBLEM_STATEMENT.md # Research question, hypotheses, metrics
│
├── src/
│   ├── data/               # Data processing scripts
│   │   ├── 01_cohort_selection.py
│   │   ├── 02_label_generation.py
│   │   ├── 03_missingness_report.py
│   │   ├── 04_feature_extraction.py    # TO BE CREATED
│   │   ├── 05_static_features.py       # TO BE CREATED
│   │   ├── 06_train_val_test_split.py  # TO BE CREATED
│   │   └── run_pipeline.py
│   │
│   ├── models/             # Model implementations
│   ├── experiments/        # Training scripts
│   └── utils/              # Helper functions
│
├── mimic_data/
│   └── processed_data/     # Generated data files
│
├── experiments/
│   ├── configs/            # Model hyperparameters
│   ├── logs/               # Training logs
│   └── models/             # Saved model weights
│
└── results/
    ├── plots/              # Visualizations
    ├── interpretation/     # SHAP, attention plots
    └── evaluation_summary.md  # Main results
```

---

## Expected Timeline

| Phase | Duration | Description |
|-------|----------|-------------|
| Setup | 5-10 min | Environment, dependencies |
| Data Processing | 30-90 min | Cohort selection through train/test split |
| Baseline Models | 15-30 min | LR, RF, XGBoost |
| Deep Learning | 2-4 hours | LSTM, Transformer training |
| Evaluation | 30 min | Metrics, calibration, fairness |
| **Total** | **4-6 hours** | Complete pipeline first run |

**Subsequent runs**: Once data is processed, you can iterate on models in minutes.

---

## Validation Checklist

Before proceeding to model training, verify:

- [ ] Cohort size is reasonable (40,000-100,000 admissions)
- [ ] Readmission rate is 15-20%
- [ ] Train/val/test split is ~70/15/15%
- [ ] No patient appears in multiple splits
- [ ] Key features have >50% coverage (check missingness_report.csv)
- [ ] Time-series tensors directory contains many .npz files
- [ ] Static features file has comorbidity scores

---

## Getting Help

1. **Check logs**: `experiments/logs/pipeline_YYYYMMDD_HHMMSS.log`
2. **Review documentation**: `docs/PROBLEM_STATEMENT.md`, `README.md`
3. **Common issues**: See "Common Issues" section above
4. **Contact**: [your.email@duke.edu]

---

## Next: Advanced Topics

Once the basic pipeline is working:

1. **Ablation Studies**: Test different time windows (24h, 48h, 72h)
2. **Feature Selection**: Remove low-coverage features
3. **Hyperparameter Tuning**: Grid search for optimal model parameters
4. **Interpretability**: SHAP values, attention visualization
5. **Fairness Analysis**: Performance across demographic groups
6. **Clinical Validation**: Subgroup analysis by diagnosis

See `docs/METHODOLOGY.md` for detailed guidance on each topic.

---

**Ready to start? Run this:**

```bash
cd /Users/yuchenzhou/Documents/duke/compsci526/final_proj/git_proj/yuchen_testing/yuchen_testing_10.19
conda activate readmit_pred
cd src/data
python run_pipeline.py --all
```

Good luck! 🚀
