#!/usr/bin/env bash
# Run the expanded smoke test notebook headlessly and save outputs to a log and executed notebook.
set -euo pipefail

NOTEBOOK="yuchen_testing/preprocessing_readmit30_expanded.ipynb"
OUT_NOTEBOOK="yuchen_testing/preprocessing_readmit30_expanded.executed.ipynb"
LOGDIR="yuchen_testing/smoke_logs"
mkdir -p "$LOGDIR"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOGFILE="$LOGDIR/smoke_run_$TIMESTAMP.log"

echo "Running notebook: $NOTEBOOK" | tee "$LOGFILE"
echo "Output executed notebook will be: $OUT_NOTEBOOK" | tee -a "$LOGFILE"

# Ensure output directory exists and call nbconvert with --output-dir to avoid nested path issues
OUT_DIR=$(dirname "$OUT_NOTEBOOK")
OUT_NAME=$(basename "$OUT_NOTEBOOK")
mkdir -p "$OUT_DIR"

# Run with nbconvert; increase timeout if needed
jupyter nbconvert --to notebook --execute "$NOTEBOOK" --ExecutePreprocessor.timeout=1200 --output "$OUT_NAME" --output-dir "$OUT_DIR" 2>&1 | tee -a "$LOGFILE"

EXIT_CODE=${PIPESTATUS[0]:-0}
echo "nbconvert exit code: $EXIT_CODE" | tee -a "$LOGFILE"
if [ $EXIT_CODE -eq 0 ]; then
  echo "Completed successfully. Executed notebook and logs in $LOGDIR" | tee -a "$LOGFILE"
else
  echo "Notebook execution failed. See $LOGFILE for details." | tee -a "$LOGFILE"
fi

exit $EXIT_CODE
