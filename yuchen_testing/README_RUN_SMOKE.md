Run the expanded preprocessing smoke test locally

This directory contains a safe copy of the preprocessing notebook with an expanded smoke test that loads larger slices of the MIMIC CSVs (chartevents and labevents). Use the helper script below to execute the notebook headlessly and save outputs.

Files:
- preprocessing_readmit30_expanded.ipynb  -> Edited notebook (CE nrows=500k, LE nrows=200k, expanded itemids)
- run_smoke_test.sh                       -> Helper script to execute the notebook and save logs
- smoke_logs/                             -> Execution logs will be placed here

How to run (zsh):

1) Ensure you have the correct conda environment active with required packages (pandas, scikit-learn, jupyter).

2) Run the helper script from the project root:

```bash
chmod +x yuchen_testing/run_smoke_test.sh
./yuchen_testing/run_smoke_test.sh
```

3) After completion the executed notebook will be saved as `yuchen_testing/preprocessing_readmit30_expanded.executed.ipynb` and logs will be in `yuchen_testing/smoke_logs/`.

4) If you want to run interactively, open `preprocessing_readmit30_expanded.ipynb` in Jupyter and run cells manually; cell numbers are visible in the notebook UI.

Notes:
- The run may take several minutes and use multiple GB of RAM depending on the machine.
- If the script errors with memory problems, reduce `nrows` in the notebook or run the notebook interactively and increase `nrows` gradually.
