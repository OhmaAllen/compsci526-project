#!/usr/bin/env python3
"""
Baseline Models: Logistic Regression, Random Forest, XGBoost

Train classical ML models on static features only as baselines.
These establish the performance floor that deep learning models should beat.

Models:
1. Logistic Regression (LR): Linear baseline
2. Random Forest (RF): Non-linear tree ensemble
3. XGBoost (XGB): Gradient boosted trees

All models use only static features:
- age, gender_binary, los_days, admission_type_cat, charlson_score, num_diagnoses
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, 
    precision_recall_curve, roc_curve,
    f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
import xgboost as xgb
import pickle
import json
from datetime import datetime

# Set random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Paths
BASE_DIR = Path(__file__).parent.parent / 'data'
PROCESSED_DIR = BASE_DIR / 'mimic_data' / 'processed_data'
RESULTS_DIR = Path(__file__).parent.parent.parent / 'results'
MODELS_DIR = RESULTS_DIR / 'models'

# Create directories
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("BASELINE MODELS: LR / RF / XGBoost")
print("="*70)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Random seed: {RANDOM_SEED}")
print()

# ============================================================================
# Load Data
# ============================================================================
print("Loading data...")
print("-"*70)

# Load static features
static_df = pd.read_csv(PROCESSED_DIR / 'static_features.csv')
print(f"Static features loaded: {len(static_df):,} admissions")

# Load splits
train_ids = pd.read_csv(PROCESSED_DIR / 'split_train.txt', header=None)[0].tolist()
val_ids = pd.read_csv(PROCESSED_DIR / 'split_val.txt', header=None)[0].tolist()
test_ids = pd.read_csv(PROCESSED_DIR / 'split_test.txt', header=None)[0].tolist()

print(f"Train: {len(train_ids):,} admissions")
print(f"Val:   {len(val_ids):,} admissions")
print(f"Test:  {len(test_ids):,} admissions")
print()

# Filter to splits
train_df = static_df[static_df['hadm_id'].isin(train_ids)].copy()
val_df = static_df[static_df['hadm_id'].isin(val_ids)].copy()
test_df = static_df[static_df['hadm_id'].isin(test_ids)].copy()

# Convert categorical admission_type to numeric
admission_type_map = {'emergency': 0, 'elective': 1, 'observation': 2, 'other': 3}
for df in [train_df, val_df, test_df, static_df]:
    df['admission_type_cat'] = df['admission_type_cat'].map(admission_type_map)

# Define features
feature_cols = ['age', 'gender_binary', 'los_days', 'admission_type_cat', 
                'charlson_score', 'num_diagnoses']
target_col = 'readmit_30d'

print(f"Features used: {feature_cols}")
print(f"Target: {target_col}")
print()

# Extract X and y
X_train = train_df[feature_cols].values
y_train = train_df[target_col].values

X_val = val_df[feature_cols].values
y_val = val_df[target_col].values

X_test = test_df[feature_cols].values
y_test = test_df[target_col].values

print("Class distribution:")
print(f"  Train: {y_train.mean():.3%} readmissions")
print(f"  Val:   {y_val.mean():.3%} readmissions")
print(f"  Test:  {y_test.mean():.3%} readmissions")
print()

# ============================================================================
# Preprocessing
# ============================================================================
print("Preprocessing...")
print("-"*70)

# Standardize features (important for LR)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"Feature means (train): {X_train.mean(axis=0)}")
print(f"Feature stds (train):  {X_train.std(axis=0)}")
print()

# Save scaler
scaler_path = MODELS_DIR / 'scaler.pkl'
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"✓ Saved scaler to {scaler_path.name}")
print()

# ============================================================================
# Model 1: Logistic Regression
# ============================================================================
print("="*70)
print("MODEL 1: Logistic Regression")
print("="*70)

print("Training LR...")
lr_model = LogisticRegression(
    random_state=RANDOM_SEED,
    max_iter=1000,
    class_weight='balanced',  # Handle class imbalance
    solver='lbfgs'
)
lr_model.fit(X_train_scaled, y_train)

# Predictions
lr_val_proba = lr_model.predict_proba(X_val_scaled)[:, 1]
lr_test_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

# Evaluate
lr_val_auroc = roc_auc_score(y_val, lr_val_proba)
lr_val_auprc = average_precision_score(y_val, lr_val_proba)
lr_test_auroc = roc_auc_score(y_test, lr_test_proba)
lr_test_auprc = average_precision_score(y_test, lr_test_proba)

print(f"✓ Training complete")
print(f"Val  AUROC: {lr_val_auroc:.4f}")
print(f"Val  AUPRC: {lr_val_auprc:.4f}")
print(f"Test AUROC: {lr_test_auroc:.4f}")
print(f"Test AUPRC: {lr_test_auprc:.4f}")

# Feature importance
lr_coefs = lr_model.coef_[0]
print("\nFeature coefficients (importance):")
for feat, coef in zip(feature_cols, lr_coefs):
    print(f"  {feat:20s}: {coef:+.4f}")

# Save model
lr_model_path = MODELS_DIR / 'logistic_regression.pkl'
with open(lr_model_path, 'wb') as f:
    pickle.dump(lr_model, f)
print(f"\n✓ Saved model to {lr_model_path.name}")
print()

# ============================================================================
# Model 2: Random Forest
# ============================================================================
print("="*70)
print("MODEL 2: Random Forest")
print("="*70)

print("Training RF...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=50,
    min_samples_leaf=20,
    random_state=RANDOM_SEED,
    class_weight='balanced',
    n_jobs=-1,  # Use all cores
    verbose=1
)
rf_model.fit(X_train, y_train)  # RF doesn't need scaling

# Predictions
rf_val_proba = rf_model.predict_proba(X_val)[:, 1]
rf_test_proba = rf_model.predict_proba(X_test)[:, 1]

# Evaluate
rf_val_auroc = roc_auc_score(y_val, rf_val_proba)
rf_val_auprc = average_precision_score(y_val, rf_val_proba)
rf_test_auroc = roc_auc_score(y_test, rf_test_proba)
rf_test_auprc = average_precision_score(y_test, rf_test_proba)

print(f"✓ Training complete")
print(f"Val  AUROC: {rf_val_auroc:.4f}")
print(f"Val  AUPRC: {rf_val_auprc:.4f}")
print(f"Test AUROC: {rf_test_auroc:.4f}")
print(f"Test AUPRC: {rf_test_auprc:.4f}")

# Feature importance
rf_importances = rf_model.feature_importances_
print("\nFeature importances:")
for feat, imp in sorted(zip(feature_cols, rf_importances), key=lambda x: x[1], reverse=True):
    print(f"  {feat:20s}: {imp:.4f}")

# Save model
rf_model_path = MODELS_DIR / 'random_forest.pkl'
with open(rf_model_path, 'wb') as f:
    pickle.dump(rf_model, f)
print(f"\n✓ Saved model to {rf_model_path.name}")
print()

# ============================================================================
# Model 3: XGBoost
# ============================================================================
print("="*70)
print("MODEL 3: XGBoost")
print("="*70)

print("Training XGBoost...")

# Calculate scale_pos_weight for class imbalance
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Scale pos weight: {scale_pos_weight:.2f}")

xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_SEED,
    eval_metric='auc',
    n_jobs=-1
)

# Train with early stopping
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=10
)

# Predictions
xgb_val_proba = xgb_model.predict_proba(X_val)[:, 1]
xgb_test_proba = xgb_model.predict_proba(X_test)[:, 1]

# Evaluate
xgb_val_auroc = roc_auc_score(y_val, xgb_val_proba)
xgb_val_auprc = average_precision_score(y_val, xgb_val_proba)
xgb_test_auroc = roc_auc_score(y_test, xgb_test_proba)
xgb_test_auprc = average_precision_score(y_test, xgb_test_proba)

print(f"✓ Training complete")
print(f"Val  AUROC: {xgb_val_auroc:.4f}")
print(f"Val  AUPRC: {xgb_val_auprc:.4f}")
print(f"Test AUROC: {xgb_test_auroc:.4f}")
print(f"Test AUPRC: {xgb_test_auprc:.4f}")

# Feature importance
xgb_importances = xgb_model.feature_importances_
print("\nFeature importances:")
for feat, imp in sorted(zip(feature_cols, xgb_importances), key=lambda x: x[1], reverse=True):
    print(f"  {feat:20s}: {imp:.4f}")

# Save model
xgb_model_path = MODELS_DIR / 'xgboost.pkl'
with open(xgb_model_path, 'wb') as f:
    pickle.dump(xgb_model, f)
print(f"\n✓ Saved model to {xgb_model_path.name}")
print()

# ============================================================================
# Summary Table
# ============================================================================
print("="*70)
print("BASELINE MODELS SUMMARY")
print("="*70)

results = {
    'Logistic Regression': {
        'val_auroc': lr_val_auroc,
        'val_auprc': lr_val_auprc,
        'test_auroc': lr_test_auroc,
        'test_auprc': lr_test_auprc
    },
    'Random Forest': {
        'val_auroc': rf_val_auroc,
        'val_auprc': rf_val_auprc,
        'test_auroc': rf_test_auroc,
        'test_auprc': rf_test_auprc
    },
    'XGBoost': {
        'val_auroc': xgb_val_auroc,
        'val_auprc': xgb_val_auprc,
        'test_auroc': xgb_test_auroc,
        'test_auprc': xgb_test_auprc
    }
}

# Print table
print("\n{:<25} {:>12} {:>12} {:>12} {:>12}".format(
    "Model", "Val AUROC", "Val AUPRC", "Test AUROC", "Test AUPRC"))
print("-"*70)
for model_name, metrics in results.items():
    print("{:<25} {:>12.4f} {:>12.4f} {:>12.4f} {:>12.4f}".format(
        model_name,
        metrics['val_auroc'],
        metrics['val_auprc'],
        metrics['test_auroc'],
        metrics['test_auprc']
    ))

# Best model
best_model = max(results.items(), key=lambda x: x[1]['val_auroc'])
print(f"\n✓ Best model (by val AUROC): {best_model[0]} ({best_model[1]['val_auroc']:.4f})")
print()

# Save results
results_path = RESULTS_DIR / 'baseline_results.json'
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"✓ Saved results to {results_path}")

# Save predictions
predictions = {
    'hadm_id': test_df['hadm_id'].tolist(),
    'y_true': y_test.tolist(),
    'lr_proba': lr_test_proba.tolist(),
    'rf_proba': rf_test_proba.tolist(),
    'xgb_proba': xgb_test_proba.tolist()
}
predictions_df = pd.DataFrame(predictions)
predictions_path = RESULTS_DIR / 'baseline_predictions.csv'
predictions_df.to_csv(predictions_path, index=False)
print(f"✓ Saved predictions to {predictions_path}")

print()
print("="*70)
print("BASELINE TRAINING COMPLETE!")
print("="*70)
print(f"Models saved to: {MODELS_DIR}")
print(f"Results saved to: {RESULTS_DIR}")
print()
