#!/bin/bash
# Setup script for readmission prediction project
# This creates a clean conda environment with compatible packages

set -e  # Exit on error

echo "=========================================="
echo "Setting up Readmission Prediction Environment"
echo "=========================================="

# Environment name
ENV_NAME="readmit_pred"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

# Remove old environment if it exists
echo ""
echo "Checking for existing environment..."
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Removing existing ${ENV_NAME} environment..."
    conda env remove -n ${ENV_NAME} -y
fi

# Create new environment with Python 3.11 (better compatibility than 3.12)
echo ""
echo "Creating new conda environment: ${ENV_NAME}"
echo "Using Python 3.11 for better package compatibility..."
conda create -n ${ENV_NAME} python=3.11 -y

# Activate environment
echo ""
echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

# Install core scientific packages first
echo ""
echo "Installing core scientific packages..."
conda install -y numpy=1.24.3 scipy=1.11.3 -c conda-forge

# Install pandas and related
echo ""
echo "Installing pandas and data processing packages..."
conda install -y pandas=2.1.4 -c conda-forge

# Install visualization packages
echo ""
echo "Installing visualization packages..."
conda install -y matplotlib=3.8.0 seaborn=0.13.0 -c conda-forge

# Install ML packages
echo ""
echo "Installing machine learning packages..."
conda install -y scikit-learn=1.3.2 -c conda-forge
pip install xgboost==2.0.3

# Install PyTorch (CPU version for compatibility)
echo ""
echo "Installing PyTorch..."
conda install -y pytorch=2.1.0 torchvision=0.16.0 cpuonly -c pytorch

# Install other required packages
echo ""
echo "Installing additional packages..."
pip install pyyaml==6.0.1
pip install tqdm==4.66.1
pip install statsmodels==0.14.0
pip install imbalanced-learn==0.11.0
pip install shap==0.43.0

# Verify installation
echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="
python -c "
import sys
print(f'Python version: {sys.version}')
print()

packages = [
    'numpy', 'pandas', 'scipy', 'sklearn', 
    'matplotlib', 'seaborn', 'torch', 'yaml',
    'tqdm', 'statsmodels', 'xgboost'
]

print('Package versions:')
for pkg in packages:
    try:
        if pkg == 'sklearn':
            import sklearn
            print(f'  {pkg:15s} {sklearn.__version__}')
        elif pkg == 'yaml':
            import yaml
            print(f'  {pkg:15s} {yaml.__version__}')
        else:
            mod = __import__(pkg)
            print(f'  {pkg:15s} {mod.__version__}')
    except Exception as e:
        print(f'  {pkg:15s} ERROR: {e}')
"

echo ""
echo "=========================================="
echo "✓ Environment setup complete!"
echo "=========================================="
echo ""
echo "To use this environment:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "To test the pipeline:"
echo "  cd /Users/yuchenzhou/Documents/duke/compsci526/final_proj/git_proj/yuchen_testing/yuchen_testing_10.19/src/data"
echo "  python 01_cohort_selection.py"
echo ""
echo "To deactivate when done:"
echo "  conda deactivate"
echo ""
