# 30-Day Hospital Readmission Prediction - MIMIC-IV

A comprehensive machine learning pipeline for predicting 30-day hospital readmissions using temporal clinical data from the MIMIC-IV database.

## Project Overview

This project implements and evaluates multiple machine learning models (Logistic Regression, Random Forest, XGBoost, LSTM, Transformer) for predicting whether a patient will be readmitted to the hospital within 30 days of discharge. The models leverage both static patient features and time-series data from the last 48 hours before discharge.

**Research Question:** Can Transformer-based models significantly outperform LSTM and traditional ML approaches in predicting 30-day readmissions?

See [docs/PROBLEM_STATEMENT.md](docs/PROBLEM_STATEMENT.md) for detailed research objectives, hypotheses, and evaluation metrics.

---

## Project Structure

```
yuchen_testing_10.19/
├── docs/                           # Documentation
│   ├── PROBLEM_STATEMENT.md        # Research question, hypotheses, metrics
│   ├── DATA_DICTIONARY.md          # Feature descriptions
│   └── METHODOLOGY.md              # Detailed methods
├── src/                            # Source code
│   ├── data/                       # Data processing scripts
│   │   ├── 01_cohort_selection.py
│   │   ├── 02_label_generation.py
│   │   ├── 03_missingness_report.py
│   │   ├── 04_feature_extraction.py
│   │   ├── 05_static_features.py
│   │   ├── 06_train_val_test_split.py
│   │   └── run_pipeline.py
│   ├── models/                     # Model implementations
│   │   ├── baseline_lr.py
│   │   ├── baseline_tree.py
│   │   ├── lstm_model.py
│   │   ├── transformer_model.py
│   │   └── gru_d.py
│   ├── experiments/                # Training scripts
│   │   ├── train_baselines.py
│   │   ├── train_lstm.py
│   │   └── train_transformer.py
│   └── utils/                      # Utility functions
│       ├── metrics.py
│       ├── visualization.py
│       ├── calibration.py
│       └── config.py
├── mimic_data/                     # Data files (not in repo)
│   └── processed_data/             # Processed datasets
│       ├── cohort.csv
│       ├── readmission_labels.csv
│       ├── static_features.csv
│       ├── missingness_report.csv
│       ├── itemid_map.csv
│       ├── split_train.txt
│       ├── split_val.txt
│       ├── split_test.txt
│       └── time_series_tensors/    # Serialized time-series data
├── experiments/                    # Experiment configs and logs
│   ├── configs/                    # Hyperparameter configs (JSON)
│   └── logs/                       # Training logs
├── results/                        # Results and outputs
│   ├── plots/                      # Performance plots
│   ├── interpretation/             # SHAP, attention visualizations
│   ├── evaluation_summary.md       # Main results
│   ├── fairness_report.md
│   └── ablation_table.csv
├── requirements.txt                # Python dependencies
├── environment.yml                 # Conda environment (optional)
├── config.yaml                     # Main configuration file
└── README.md                       # This file
```

---

## Setup Instructions

### 1. Prerequisites

- Python 3.9+
- Access to MIMIC-IV v3.1 dataset (requires credentialed PhysioNet access)
- ~50GB disk space for processed data

### 2. Environment Setup

```bash
# Create virtual environment
conda create -n readmit_pred python=3.9
conda activate readmit_pred

# Install dependencies
pip install -r requirements.txt

# Or use conda environment file
conda env create -f environment.yml
```

### 3. Data Setup

Place the MIMIC-IV raw CSV files in the appropriate directory:

```bash
# The pipeline expects data at:
# /Users/yuchenzhou/Documents/duke/compsci526/final_proj/mimic-iv-3.1/

# Required files:
# - hosp/admissions.csv
# - hosp/patients.csv
# - hosp/diagnoses_icd.csv
# - hosp/procedures_icd.csv
# - hosp/labevents.csv
# - hosp/prescriptions.csv
# - icu/chartevents.csv
# - icu/d_items.csv
# - hosp/d_labitems.csv
```

---

## Quick Start

### Step 1: Run Data Processing Pipeline

```bash
# Run the complete preprocessing pipeline
cd src/data
python run_pipeline.py --config ../../config.yaml

# Or run steps individually:
python 01_cohort_selection.py
python 02_label_generation.py
python 03_missingness_report.py
python 04_feature_extraction.py
python 05_static_features.py
python 06_train_val_test_split.py
```

### Step 2: Train Baseline Models

```bash
cd ../experiments
python train_baselines.py --config ../experiments/configs/baseline_config.json
```

### Step 3: Train Deep Learning Models

```bash
# LSTM
python train_lstm.py --config ../experiments/configs/lstm_config.json

# Transformer
python train_transformer.py --config ../experiments/configs/transformer_config.json
```

### Step 4: Evaluate and Generate Reports

```bash
# Generate evaluation report
python ../utils/generate_report.py --output ../../results/evaluation_summary.md
```

---

## Data Processing Pipeline

### 1. Cohort Selection (`01_cohort_selection.py`)

**Filters:**
- Age ≥ 18 at admission
- Valid admission and discharge times
- Exclude deaths during hospitalization (separate label)
- Exclude observation stays (LOS < 24h)

**Output:** `mimic_data/processed_data/cohort.csv`

### 2. Label Generation (`02_label_generation.py`)

**Rules:**
- `readmit_30d = 1` if next admission occurs within 0-30 days after discharge
- Handle competing risks (death within 30 days)
- Flag last admissions (cannot observe readmission)

**Output:** `mimic_data/processed_data/readmission_labels.csv`

### 3. Missingness Analysis (`03_missingness_report.py`)

**Analysis:**
- Missing rate per feature across all admissions
- Coverage statistics (% of admissions with each feature)
- Top-20 most common lab/vital measurements

**Output:** `mimic_data/processed_data/missingness_report.csv` + plots

### 4. Time-Series Feature Extraction (`04_feature_extraction.py`)

**Time Window:** Last 48 hours before discharge (default)

**Features:**
- **Vital Signs:** Heart rate, BP (sys/dias), RR, SpO2, temperature
- **Labs:** WBC, Hgb, Platelets, Na, K, Creatinine, BUN, Glucose, Lactate
- **Medications:** Antibiotics, vasopressors, diuretics (binary flags)

**Temporal Alignment:**
- Hourly bins (48 bins total)
- Aggregation: mean per hour
- Preserve missingness with mask tensors

**Output:** `mimic_data/processed_data/time_series_tensors/` (NPZ files per admission)

### 5. Static Features (`05_static_features.py`)

**Features:**
- Demographics: age, gender
- Comorbidity scores: Charlson Comorbidity Index (CCI), Elixhauser
- Admission characteristics: admission_type, length_of_stay

**Output:** `mimic_data/processed_data/static_features.csv`

### 6. Train/Val/Test Split (`06_train_val_test_split.py`)

**Strategy:**
- Split by `subject_id` (patient-level)
- Proportions: 70% train, 15% validation, 15% test
- Stratified by `readmit_30d` label

**Output:** `split_train.txt`, `split_val.txt`, `split_test.txt`

---

## Model Architectures

### Baseline Models

1. **Logistic Regression:** Static features only (demographics + comorbidities)
2. **Random Forest:** Static + aggregated time-series (mean, std, min, max of last 48h)
3. **XGBoost:** Same features as Random Forest

### Deep Learning Models

4. **LSTM:** 
   - Input: Time-series (48, F) + static features
   - Architecture: 2-layer LSTM (hidden=128), dropout=0.2
   - Output: Binary classifier

5. **GRU-D (GRU with Decay):**
   - Handles irregular sampling and missingness explicitly
   - Input: Time-series + mask + time-since-last-observation

6. **Transformer:**
   - Input: Time-series with positional encoding + static features
   - Architecture: 4 heads, 2 layers, d_model=128
   - Self-attention over temporal sequence

---

## Evaluation Framework

### Metrics (see docs/PROBLEM_STATEMENT.md for details)

- **Discrimination:** AUROC, AUPRC
- **Classification:** F1, Precision, Recall
- **Clinical Utility:** Recall@Top10%, Precision@Top10%
- **Calibration:** Brier score, Expected Calibration Error (ECE)
- **Statistical:** 95% CI (bootstrap), DeLong test for AUROC comparison

### Fairness Analysis

Evaluate performance separately across:
- Gender (male/female)
- Age quartiles
- Race/ethnicity (if available)
- Primary diagnosis category (HF, COPD, Sepsis, etc.)

**Metrics:** Equalized odds, demographic parity, calibration by group

### Interpretability

- **SHAP:** For tree-based models (global and per-patient)
- **Attention Visualization:** For Transformer (identify important time points)
- **Feature Importance:** Aggregate across models

---

## Configuration

Edit `config.yaml` to customize:

```yaml
data:
  raw_data_path: "/Users/yuchenzhou/Documents/duke/compsci526/final_proj/mimic-iv-3.1"
  processed_data_path: "mimic_data/processed_data"
  time_window_hours: 48  # Can test 24, 48, 72
  
cohort:
  min_age: 18
  min_los_hours: 24
  exclude_deaths: true
  
features:
  vitals: ["heart_rate", "sbp", "dbp", "resp_rate", "spo2", "temperature"]
  labs: ["wbc", "hemoglobin", "platelets", "sodium", "potassium", "creatinine", 
         "bun", "glucose", "lactate"]
  
models:
  random_seed: 42
  test_size: 0.15
  val_size: 0.15
```

---

## Reproducibility

All experiments use **fixed random seeds**:
- Data split: `seed=42`
- Model training: `seed=2024`
- Bootstrap resampling: `seed=123`

Dependencies are pinned in `requirements.txt`.

---

## Results

Results will be saved in `results/`:

- `evaluation_summary.md`: Main performance table across all models
- `plots/`: ROC curves, PR curves, calibration plots
- `interpretation/`: SHAP summary plots, attention heatmaps
- `ablation_table.csv`: Ablation study results (time windows, feature sets)

---

## Citation

If you use this code, please cite:

```
@misc{yuchen2025readmission,
  title={Predicting 30-Day Hospital Readmission using Transformer Models on MIMIC-IV},
  author={Yuchen Zhou},
  year={2025},
  institution={Duke University, CS526}
}
```

MIMIC-IV dataset:
```
Johnson, A., Bulgarelli, L., Pollard, T., Horng, S., Celi, L. A., & Mark, R. (2023). 
MIMIC-IV (version 2.2). PhysioNet. https://doi.org/10.13026/6mm1-ek67
```

---

## License

This project is for academic research purposes. The code is licensed under MIT License. MIMIC-IV data usage must comply with PhysioNet credentialed data use agreement.

---

## Contact

**Author:** Yuchen Zhou  
**Email:** [your.email@duke.edu]  
**Course:** CS526 - Machine Learning for Health  
**Semester:** Fall 2025

For questions or issues, please open a GitHub issue or contact via email.
