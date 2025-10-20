# Problem Statement: 30-Day Hospital Readmission Prediction using MIMIC-IV

**Date:** October 20, 2025  
**Project:** Master's Thesis - CS526 Final Project

---

## 1. Research Question

**Primary Research Question:**

> In MIMIC-IV, can a Transformer-based model using static features combined with the last 48 hours of time-series data before discharge significantly outperform LSTM and traditional machine learning models in predicting 30-day hospital readmission, as measured by AUROC?

**Sub-questions:**

1. What is the optimal time window (24h, 48h, or 72h before discharge) for capturing predictive signals of readmission?
2. Which features (vital signs, laboratory values, medications, or static demographics) contribute most to prediction performance?
3. How do different models perform across patient subgroups (age, gender, primary diagnosis)?
4. Are the models well-calibrated for clinical decision support?

---

## 2. Research Hypotheses

### Hypothesis 1: Model Performance
**H1:** Transformer models will achieve significantly higher AUROC (≥0.02 improvement) compared to LSTM models when predicting 30-day readmission using the same feature set and time window.

**Justification:** Transformers' self-attention mechanism can better capture long-range dependencies and irregular temporal patterns in clinical data compared to LSTM's sequential processing.

### Hypothesis 2: Time Window Optimization
**H2:** The 48-hour pre-discharge window will yield better predictive performance than shorter (24h) or longer (72h) windows.

**Justification:** The 48h window balances recency of clinical deterioration signals with sufficient data density, while avoiding excessive noise from earlier admission periods.

### Hypothesis 3: Feature Importance
**H3:** Laboratory values (e.g., creatinine, BUN, lactate) combined with vital signs will have higher predictive importance than static demographic features alone.

**Justification:** Dynamic physiological measurements reflect the patient's clinical trajectory and readiness for discharge better than static characteristics.

### Hypothesis 4: Subgroup Performance
**H4:** Model performance will be consistent (AUROC variance < 0.05) across gender and age groups, but may vary by primary diagnosis category.

**Justification:** Readmission patterns may differ by disease etiology (e.g., heart failure vs. COPD), but should not systematically differ by demographics in a well-designed model.

---

## 3. Evaluation Metrics

### Primary Metrics

| Metric | Target/Threshold | Clinical Interpretation |
|--------|------------------|------------------------|
| **AUROC** | ≥ 0.75 (good); ≥ 0.80 (excellent) | Overall discrimination ability between readmitted and not readmitted patients |
| **AUPRC** | ≥ 0.40 (given class imbalance) | Performance on positive class (readmissions), important given expected 15-20% base rate |

### Secondary Metrics

| Metric | Target/Threshold | Purpose |
|--------|------------------|---------|
| **F1 Score** | Maximize at optimal threshold | Balance between precision and recall |
| **Recall@Top10%** | ≥ 0.40 | Proportion of readmissions captured if we intervene on top 10% highest risk patients |
| **Precision@Top10%** | ≥ 0.35 | Positive predictive value in high-risk group |
| **Brier Score** | < 0.15 (well-calibrated) | Calibration quality for probability estimates |
| **Expected Calibration Error (ECE)** | < 0.10 | Calibration across probability bins |

### Statistical Testing

- **95% Confidence Intervals:** Bootstrap resampling (1000 iterations) for all metrics on test set
- **Model Comparison:** DeLong test for AUROC comparison between models
- **Significance Level:** α = 0.05 (two-tailed)
- **Effect Size:** Cohen's h for proportion differences

### Fairness Metrics (by subgroup: gender, age quartiles, race)

- **Equalized Odds Difference:** |FPR_group1 - FPR_group2| and |TPR_group1 - TPR_group2|
- **Demographic Parity Difference:** |P(ŷ=1|group1) - P(ŷ=1|group2)|
- **Calibration by Group:** Separate Brier scores and reliability plots

---

## 4. Success Criteria

**Minimum Viable Success:**
- AUROC ≥ 0.75 on held-out test set
- AUPRC ≥ 0.35
- 95% CI does not include 0.50 (random chance)
- Model is calibrated (Brier score < 0.20)

**Target Success:**
- Transformer achieves AUROC ≥ 0.78 and significantly outperforms LSTM (p < 0.05)
- Recall@Top10% ≥ 0.40 (actionable for clinical intervention)
- Fairness metrics show no systematic bias (equalized odds difference < 0.10)

**Stretch Goals:**
- AUROC ≥ 0.80
- Interpretable attention patterns align with clinical knowledge
- Model generalizes across different hospital wards and primary diagnoses

---

## 5. Clinical Context and Impact

**Clinical Problem:** 
Hospital readmissions within 30 days are associated with poor patient outcomes and represent a significant healthcare cost burden. Identifying high-risk patients before discharge enables targeted interventions (e.g., intensive follow-up, medication reconciliation, home health services).

**Target Users:** 
- Hospital discharge planners
- Care transition teams
- Primary care physicians receiving handoffs

**Deployment Consideration:**
The model should produce calibrated probability estimates, not just binary classifications, to support risk-stratified intervention strategies.

---

## 6. Data and Cohort Specifications

**Dataset:** MIMIC-IV v3.1  
**Population:** Adult ICU and general ward patients (age ≥ 18)  
**Inclusion Criteria:**
- Complete admission and discharge times
- Alive at discharge (death is a competing outcome, not readmission)
- Valid time-series data in the pre-discharge window

**Exclusion Criteria:**
- Patients discharged against medical advice (AMA)
- Admissions with length of stay < 24 hours (observation only)
- Transfers to other acute care facilities (cannot observe readmission)

**Expected Cohort Size:** ~50,000-100,000 admissions (after exclusions)

---

## 7. Reproducibility Statement

All experiments will be conducted with:
- Fixed random seeds (seed = 42 for train/val/test split, 2024 for model initialization)
- Documented software versions (requirements.txt)
- Version-controlled code (Git)
- Saved model checkpoints and training logs
- Detailed hyperparameter configurations (JSON files)

---

## 8. Timeline and Milestones

| Phase | Deliverable | Target Date |
|-------|-------------|-------------|
| Phase 1 | Cohort selection, label generation, missingness report | Week 1 |
| Phase 2 | Feature engineering, time-series extraction, train/val/test split | Week 2 |
| Phase 3 | Baseline models (LR, RF, XGBoost) | Week 3 |
| Phase 4 | Deep learning models (LSTM, Transformer) | Week 4 |
| Phase 5 | Statistical evaluation, calibration, fairness | Week 5 |
| Phase 6 | Interpretability (SHAP, attention), ablation studies | Week 6 |
| Final | Documentation, final report, presentation | Week 7 |

---

**Document Version:** 1.0  
**Last Updated:** October 20, 2025
