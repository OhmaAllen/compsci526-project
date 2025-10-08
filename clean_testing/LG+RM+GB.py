"""
Fast MIMIC-IV Readmission Prediction Pipeline
使用方案1：transform()加速时序特征构建（适合50万+数据）
"""

import warnings
import numpy as np
import pandas as pd
import random
from pathlib import Path
import time

warnings.filterwarnings("ignore")

# ============== CONFIG ==============
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

CLEANED_CSV = "/Users/xiaohei/Documents/Duke/Courses/526 Data Science/Project/compsci526-project/readmission_features_30d_v2_cleaned_20251008_1229.csv"
ID_COLS = ["subject_id", "hadm_id"]
TIME_COL = "admittime"
TARGET = "readmit_label"
ROLL_WINDOW = 3

POOL_NUM_COLS_KEYWORDS = [
    "WBC_", "Hemoglobin_", "PlateletCount_",
    "Sodium_", "Potassium_", "Creatinine_",
    "UreaNitrogen_", "Glucose_",
    "length_of_stay", "num_transfers", "ed_los_hours",
    "days_since_prev_discharge"
]

# ============ IMPORTS ============
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, precision_recall_curve, confusion_matrix, classification_report
)

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.combine import SMOTETomek
    USE_SMOTE = True
except ImportError:
    print("⚠️ imbalanced-learn not installed. Using class_weight instead.")
    USE_SMOTE = False

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============ UTILITY FUNCTIONS ============
def pick_cols_for_pool(df):
    """选择用于建模的列"""
    cols = set(ID_COLS + [TIME_COL, TARGET])
    for c in df.columns:
        if any(kw in c for kw in POOL_NUM_COLS_KEYWORDS):
            cols.add(c)
        if any(p in c for p in ["gender_M", "marital_status_", "insurance_", 
                                "age_group_", "los_category_", "anchor_age",
                                "ed_duration_category_", "comorbidity_score_", 
                                "admission_season_", "high_risk_patient"]):
            cols.add(c)
    return list(cols)

def build_temporal_fast(df_input, pool_cols):
    """
    快速构建时序池化特征 - 使用vectorized操作
    50万条记录预计5-15分钟
    """
    print(f"   Processing {len(df_input)} records with {len(pool_cols)} features...")
    start_time = time.time()
    
    df = df_input.copy()
    df = df.sort_values(["subject_id", TIME_COL]).reset_index(drop=True)
    
    # 为每个数值列创建时序特征
    for i, col in enumerate(pool_cols, 1):
        if i % 5 == 0:
            elapsed = time.time() - start_time
            print(f"      [{i}/{len(pool_cols)}] {col} ({elapsed:.1f}s elapsed)")
        
        # Step 1: shift(1) - 避免数据泄露
        shifted = df.groupby("subject_id")[col].shift(1)
        
        # Step 2: rolling统计量
        # 关键：使用 groupby().rolling() 确保不会跨患者计算
        grouped = shifted.groupby(df["subject_id"])
        
        # 计算rolling mean
        df[f"{col}_rollmean"] = grouped.transform(
            lambda x: x.rolling(window=ROLL_WINDOW, min_periods=1).mean()
        )
        
        # 计算rolling min
        df[f"{col}_rollmin"] = grouped.transform(
            lambda x: x.rolling(window=ROLL_WINDOW, min_periods=1).min()
        )
        
        # 计算rolling max
        df[f"{col}_rollmax"] = grouped.transform(
            lambda x: x.rolling(window=ROLL_WINDOW, min_periods=1).max()
        )
        
        # 上一次的值
        df[f"{col}_prev"] = shifted
        
        # Step 3: 填充NaN（首次入院没有历史）
        for suffix in ['_rollmean', '_rollmin', '_rollmax', '_prev']:
            new_col = f"{col}{suffix}"
            # 用当前值填充（不引入未来信息）
            df[new_col] = df[new_col].fillna(df[col])
    
    total_time = time.time() - start_time
    print(f"   ✅ Completed in {total_time:.1f}s ({total_time/60:.1f} min)")
    
    return df

# ============ 1) LOAD & PREPROCESS ============
print("="*60)
print("📂 Loading data...")
df = pd.read_csv(CLEANED_CSV)

if TIME_COL in df.columns:
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")

use_cols = pick_cols_for_pool(df)
df = df[use_cols].copy()
df = df.sort_values([TIME_COL, "subject_id"]).reset_index(drop=True)

print(f"✅ Loaded {len(df):,} records, {len(df.columns)} features")
print(f"   Target distribution: {df[TARGET].value_counts().to_dict()}")
print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

# ============ 2) TIME-BASED SPLIT (BEFORE ROLLING) ============
print("\n" + "="*60)
print("🔪 Splitting train/test by time (80/20)...")
cut_idx = int(len(df) * 0.8)
train_df = df.iloc[:cut_idx].copy()
test_df = df.iloc[cut_idx:].copy()

print(f"   Train: {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
print(f"   Test:  {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)")

# ============ 3) BUILD TEMPORAL FEATURES - FAST VERSION ============
print("\n" + "="*60)
print("⏱️ Building temporal pooling features (FAST VERSION)...")

# 确定需要池化的列
all_num_cols = train_df.select_dtypes(include=["float", "int"]).columns.tolist()
for drop_c in ID_COLS + [TARGET]:
    if drop_c in all_num_cols:
        all_num_cols.remove(drop_c)

pool_cols = [c for c in all_num_cols if any(kw in c for kw in POOL_NUM_COLS_KEYWORDS)]
print(f"   Selected {len(pool_cols)} numeric columns for pooling")
print(f"   Rolling window: {ROLL_WINDOW} records")

# 构建训练集特征
print("\n📊 Processing TRAIN set:")
train_df = build_temporal_fast(train_df, pool_cols)

# 构建测试集特征
print("\n📊 Processing TEST set:")
test_df = build_temporal_fast(test_df, pool_cols)

print("\n✅ Temporal features successfully built!")

# ============ 4) FEATURE MATRIX ============
print("\n" + "="*60)
print("🔧 Preparing feature matrices...")

drop_cols = set(ID_COLS + [TIME_COL, TARGET])
X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])
y_train = train_df[TARGET].astype(int)
y_test = test_df[TARGET].astype(int)

# 确保训练/测试集列一致
missing_in_test = set(X_train.columns) - set(X_test.columns)
missing_in_train = set(X_test.columns) - set(X_train.columns)

if missing_in_test:
    print(f"   ⚠️ Adding {len(missing_in_test)} missing cols to test set")
    for col in missing_in_test:
        X_test[col] = 0

if missing_in_train:
    print(f"   ⚠️ Adding {len(missing_in_train)} missing cols to train set")
    for col in missing_in_train:
        X_train[col] = 0

X_test = X_test[X_train.columns]

print(f"   Feature matrix: {X_train.shape}")
print(f"   Train target: {dict(zip(*np.unique(y_train, return_counts=True)))}")
print(f"   Test target:  {dict(zip(*np.unique(y_test, return_counts=True)))}")

# 标准化
print("\n   Standardizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("   ✅ Standardization complete")

# ============ 5) HANDLE IMBALANCE ============
print("\n" + "="*60)
print("⚖️ Handling class imbalance...")

if USE_SMOTE:
    print("   Using SMOTETomek (hybrid sampling)...")
    smt = SMOTETomek(random_state=RANDOM_SEED)
    X_train_bal, y_train_bal = smt.fit_resample(X_train_scaled, y_train)
    print(f"   After resampling: {dict(zip(*np.unique(y_train_bal, return_counts=True)))}")
else:
    X_train_bal = X_train_scaled
    y_train_bal = y_train
    print("   Using class_weight='balanced' in models")

# ============ 6) MODEL TRAINING & EVALUATION ============
print("\n" + "="*60)
print("🤖 Training and evaluating models...")
print("="*60)

# 定义模型
models = {
    "LogisticRegression": LogisticRegression(
        max_iter=200, 
        class_weight='balanced' if not USE_SMOTE else None,
        random_state=RANDOM_SEED,
        n_jobs=-1
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=20,
        class_weight='balanced' if not USE_SMOTE else None,
        random_state=RANDOM_SEED,
        n_jobs=-1
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=RANDOM_SEED
    )
}

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

for name, clf in models.items():
    print(f"\n{'='*60}")
    print(f"📊 {name}")
    print('='*60)
    
    # 交叉验证
    print("   Running 5-fold cross-validation...")
    cv_start = time.time()
    cv_scores = cross_val_score(clf, X_train_bal, y_train_bal, 
                                cv=cv, scoring='roc_auc', n_jobs=-1)
    cv_time = time.time() - cv_start
    print(f"   CV AUROC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f} ({cv_time:.1f}s)")
    
    # 训练最终模型
    print("   Training final model on full training set...")
    train_start = time.time()
    clf.fit(X_train_bal, y_train_bal)
    train_time = time.time() - train_start
    print(f"   Training completed in {train_time:.1f}s")
    
    # 预测
    y_prob = clf.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    
    # 计算指标
    auroc = roc_auc_score(y_test, y_prob)
    auprc = average_precision_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    results[name] = {
        'CV_AUROC': cv_scores.mean(),
        'CV_AUROC_std': cv_scores.std(),
        'Test_AUROC': auroc,
        'Test_AUPRC': auprc,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1': f1
    }
    
    # 打印结果
    print(f"\n   📈 Test Set Performance:")
    print(f"      AUROC:     {auroc:.4f}")
    print(f"      AUPRC:     {auprc:.4f}")
    print(f"      Accuracy:  {acc:.4f}")
    print(f"      Precision: {prec:.4f}")
    print(f"      Recall:    {rec:.4f}")
    print(f"      F1-Score:  {f1:.4f}")
    
    print(f"\n   📊 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"      ┌─────────────┬─────────┬─────────┐")
    print(f"      │             │ Pred: 0 │ Pred: 1 │")
    print(f"      ├─────────────┼─────────┼─────────┤")
    print(f"      │ Actual: 0   │ {cm[0,0]:7,} │ {cm[0,1]:7,} │")
    print(f"      │ Actual: 1   │ {cm[1,0]:7,} │ {cm[1,1]:7,} │")
    print(f"      └─────────────┴─────────┴─────────┘")
    
    # 绘制曲线
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    precs, recs, _ = precision_recall_curve(y_test, y_prob)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(fpr, tpr, linewidth=2, label=f'AUROC={auroc:.3f}')
    axes[0].plot([0,1], [0,1], '--', alpha=0.5, color='gray', label='Random')
    axes[0].set_xlabel('False Positive Rate', fontsize=11)
    axes[0].set_ylabel('True Positive Rate', fontsize=11)
    axes[0].set_title(f'{name} - ROC Curve', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)
    
    axes[1].plot(recs, precs, linewidth=2, label=f'AUPRC={auprc:.3f}')
    axes[1].axhline(y=y_test.mean(), linestyle='--', alpha=0.5, color='gray', label='Baseline')
    axes[1].set_xlabel('Recall', fontsize=11)
    axes[1].set_ylabel('Precision', fontsize=11)
    axes[1].set_title(f'{name} - Precision-Recall Curve', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    filename = f'results_{name.lower().replace(" ", "_")}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   💾 Plot saved: {filename}")

# ============ 7) SUMMARY ============
print("\n" + "="*60)
print("📈 FINAL RESULTS SUMMARY")
print("="*60)

summary_df = pd.DataFrame(results).T
print(summary_df.round(4).to_string())

# 保存结果
summary_df.to_csv('model_results_summary.csv')
print(f"\n✅ Results saved to: model_results_summary.csv")
print(f"✅ Plots saved as: results_*.png")

# 打印最佳模型
best_model = summary_df['Test_AUROC'].idxmax()
best_auroc = summary_df.loc[best_model, 'Test_AUROC']
print(f"\n🏆 Best Model: {best_model} (Test AUROC: {best_auroc:.4f})")

print("\n" + "="*60)
print("✨ Pipeline completed successfully!")
print("="*60)
