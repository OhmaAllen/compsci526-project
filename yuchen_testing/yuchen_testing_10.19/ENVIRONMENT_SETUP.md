# Environment Setup Instructions

## Problem: NumPy Compatibility Issues

You're experiencing NumPy 2.x vs 1.x compatibility issues with Python 3.12. The solution is to create a **clean, dedicated conda environment** with compatible package versions.

---

## 🚀 Quick Setup (Recommended - 5 minutes)

### Option 1: Automated Setup (Easiest)

```bash
cd /Users/yuchenzhou/Documents/duke/compsci526/final_proj/git_proj/yuchen_testing/yuchen_testing_10.19

# Make script executable
chmod +x setup_environment.sh

# Run setup script (creates environment automatically)
./setup_environment.sh
```

This will:
- ✓ Create a new `readmit_pred` conda environment with Python 3.11
- ✓ Install all compatible packages
- ✓ Verify installation
- ✓ Show you how to activate it

**Total time: ~3-5 minutes**

---

### Option 2: Manual Setup with environment.yml (Alternative)

```bash
cd /Users/yuchenzhou/Documents/duke/compsci526/final_proj/git_proj/yuchen_testing/yuchen_testing_10.19

# Create environment from YAML file
conda env create -f environment.yml

# Activate environment
conda activate readmit_pred

# Verify
python -c "import numpy, pandas, sklearn, torch; print('✓ All packages loaded successfully')"
```

**Total time: ~5-10 minutes** (conda resolves dependencies)

---

### Option 3: Step-by-Step Manual Install (If others fail)

```bash
# 1. Create clean environment with Python 3.11
conda create -n readmit_pred python=3.11 -y

# 2. Activate it
conda activate readmit_pred

# 3. Install core packages via conda (FAST)
conda install -y numpy=1.24.3 scipy=1.11.3 pandas=2.1.4 \
    scikit-learn=1.3.2 matplotlib=3.8.0 seaborn=0.13.0 \
    pyyaml=6.0.1 tqdm=4.66.1 -c conda-forge

# 4. Install PyTorch
conda install -y pytorch=2.1.0 torchvision=0.16.0 cpuonly -c pytorch

# 5. Install remaining packages via pip
pip install xgboost==2.0.3 statsmodels==0.14.0 \
    imbalanced-learn==0.11.0 shap==0.43.0

# 6. Verify
python -c "import numpy as np; print(f'NumPy version: {np.__version__}')"
```

---

## ✅ After Setup - Test Your Scripts

```bash
# Activate the environment
conda activate readmit_pred

# Navigate to your scripts
cd /Users/yuchenzhou/Documents/duke/compsci526/final_proj/git_proj/yuchen_testing/yuchen_testing_10.19/src/data

# Test cohort selection (should run clean, no warnings!)
python 01_cohort_selection.py

# Test label generation
python 02_label_generation.py

# Test missingness report (this was failing before)
python 03_missingness_report.py
```

**Expected result:** No NumPy warnings, clean execution! ✨

---

## 📝 Why This Fixes the Problem

### The Issue:
- You're using **Python 3.12** with the **base** environment
- Base environment has **NumPy 2.2.6** (too new)
- Other packages (pandas, scipy, seaborn) were compiled for **NumPy 1.x**
- Result: Binary incompatibility errors

### The Solution:
- Use **Python 3.11** (better package ecosystem support)
- Install **NumPy 1.24.3** (stable, widely supported)
- Install all packages together in a **clean environment**
- No conflicts between pre-installed packages!

---

## 🎯 Usage After Setup

### Always activate before working:
```bash
conda activate readmit_pred
```

### When done:
```bash
conda deactivate
```

### To verify which environment you're in:
```bash
conda env list
# The active environment will have a * next to it
```

### To check Python/NumPy versions:
```bash
python --version
python -c "import numpy; print(numpy.__version__)"
```

Expected output:
```
Python 3.11.x
1.24.3
```

---

## 🔧 Troubleshooting

### If setup script fails:
1. Make sure you're in the base environment first:
   ```bash
   conda deactivate  # Deactivate any active environment
   conda activate base
   ```

2. Update conda:
   ```bash
   conda update -n base conda
   ```

3. Try again:
   ```bash
   ./setup_environment.sh
   ```

### If you get "permission denied":
```bash
chmod +x setup_environment.sh
```

### To completely remove and retry:
```bash
conda env remove -n readmit_pred
# Then run setup again
```

---

## 📊 What Gets Installed

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.11 | Base language |
| NumPy | 1.24.3 | Numerical computing |
| Pandas | 2.1.4 | Data manipulation |
| SciPy | 1.11.3 | Scientific computing |
| Scikit-learn | 1.3.2 | Machine learning |
| PyTorch | 2.1.0 | Deep learning |
| XGBoost | 2.0.3 | Gradient boosting |
| Matplotlib | 3.8.0 | Plotting |
| Seaborn | 0.13.0 | Statistical visualization |
| SHAP | 0.43.0 | Model interpretability |

**Total size:** ~2-3 GB

---

## 💡 Pro Tips

1. **Always use this environment for this project:**
   - Add `conda activate readmit_pred` to your workflow
   - Consider adding it to your shell profile for auto-activation in project directory

2. **VS Code Integration:**
   - If using VS Code, select the `readmit_pred` Python interpreter
   - Command Palette → "Python: Select Interpreter" → Choose `readmit_pred`

3. **Jupyter Integration:**
   ```bash
   conda activate readmit_pred
   python -m ipykernel install --user --name readmit_pred --display-name "Python (readmit_pred)"
   ```

4. **Freeze your environment later:**
   ```bash
   conda env export > environment_frozen.yml
   ```

---

## ⏱️ Installation Time Explained

### Why it takes time:
1. **Dependency resolution** (2-3 min): Conda checks compatibility of all packages
2. **Download packages** (1-2 min): ~2-3 GB of scientific libraries
3. **Extract and link** (1-2 min): Installing files

### Total: 5-10 minutes for initial setup

**But:** Once set up, it's instant to activate and use!

---

## 🎉 Success Criteria

After running the setup, you should be able to:

```bash
conda activate readmit_pred
cd /Users/yuchenzhou/Documents/duke/compsci526/final_proj/git_proj/yuchen_testing/yuchen_testing_10.19/src/data
python 03_missingness_report.py
```

**Expected:** Clean execution with no NumPy warnings! ✨

---

**Ready to start? Run:**

```bash
cd /Users/yuchenzhou/Documents/duke/compsci526/final_proj/git_proj/yuchen_testing/yuchen_testing_10.19
./setup_environment.sh
```
