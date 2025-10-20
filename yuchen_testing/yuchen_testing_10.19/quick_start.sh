#!/bin/bash
# Quick start script - Run this to activate environment and test pipeline

echo "🚀 Activating readmit_pred environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate readmit_pred

echo ""
echo "✓ Environment activated!"
echo "  Python: $(python --version)"
echo "  NumPy: $(python -c 'import numpy; print(numpy.__version__)')"
echo ""

# Navigate to data directory
cd /Users/yuchenzhou/Documents/duke/compsci526/final_proj/git_proj/yuchen_testing/yuchen_testing_10.19/src/data

echo "📊 Current location: $(pwd)"
echo ""
echo "Ready to run scripts! Try:"
echo "  python 01_cohort_selection.py    # Already done ✓"
echo "  python 02_label_generation.py    # Already done ✓"
echo "  python 03_missingness_report.py  # Run this next!"
echo ""
echo "Or run the complete pipeline:"
echo "  python run_pipeline.py --all"
echo ""

# Keep shell open in activated environment
$SHELL
