# Assignment 1 — End-to-End Machine Learning Pipeline

**Problem:** Predict online shopping session purchase intention (`Revenue`) from browsing and session attributes.  
**Track:** A — scikit-learn; four classifiers with stratified cross-validation and grid search.  
**Status:** Notebook executed successfully; results and plots are saved in this repository.

## Start here

- `progress_report.md` — detailed project log, methods, observed results, decisions, limitations, and next steps for the team.
- `assignment1_pipeline.ipynb` — complete runnable notebook with executed outputs.
- `assignment1_report.pdf` — short assignment report.
- `assignment1_notebook_code.pdf` — printable notebook source-code export.
- `output/model_comparison.csv` — model comparison table.
- `data/online_shoppers_intention.csv` — original UCI data; `data/dataset_sample_100_rows.csv` is a convenient sample.
- `figures/` — charts exported from the executed notebook.

## Reproduce

Python 3.9 or newer is recommended. From this directory:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
jupyter nbconvert --execute --to notebook --inplace assignment1_pipeline.ipynb
```

The notebook uses seed 42, a stratified 80/20 holdout, 5-fold stratified CV, and average precision (AP) for grid-search selection. All imputing, scaling, and encoding is fitted inside the model pipeline.

## Dataset citation

Sakar, C. O. & Kastro, Y. (2018). *Online Shoppers Purchasing Intention Dataset*. UCI Machine Learning Repository. DOI: [10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q). Dataset page: https://archive.ics.uci.edu/dataset/468/online%2Bshoppers%2Bpurchasing%2Bintention. Licensed under CC BY 4.0.
