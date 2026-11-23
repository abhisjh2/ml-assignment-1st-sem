# Assignment 1 — End-to-End Machine Learning Pipeline

**Problem:** Predict whether an online shopping session ends in a purchase (`Revenue`) from browsing behaviour and session context.
**Track:** A — scikit-learn; five model families (Logistic Regression, Decision Tree, Random Forest, Extra Trees, Histogram Gradient Boosting) tuned with grid search and 5-fold stratified cross-validation.
**Status:** Saved notebook outputs are from a macOS run on Python 3.13. The virtual lab run, screenshot, and final notebook PDF are pending.

## Start here

| File | What it is |
|---|---|
| `assignment1_pipeline.ipynb` | The complete, executed notebook (EDA, preprocessing, models, evaluation, interpretation). |
| `assignment1_pipeline.html` | Browser export of the notebook with saved outputs and plots. Download and open it locally. |
| `assignment1_report.pdf` | Five-page report with the virtual lab section marked pending. |
| `output/model_comparison.csv` | Metrics for every model plus the selected hyperparameters. |
| `output/summary.json` | Key numbers used by the report (selected model, PageValues sensitivity, top features). |
| `figures/` | All charts, exported by the notebook. |
| `data/online_shoppers_intention.csv` | Original UCI data; `data/dataset_sample_100_rows.csv` is a 100-row sample. |
| `make_pdfs.py` | Builds the report from `output/` and `figures/`. |
| `pdf_template/` | nbconvert template for wrapping long code lines in the notebook PDF. |
| `progress_report.md` | Revision history and submission checklist. |

## Results (models sorted by mean cross-validation average precision)

| Model | CV AP | Test AP | ROC-AUC | F1 |
|---|---:|---:|---:|---:|
| **Extra Trees (selected)** | 0.385 ± 0.028 | 0.369 | 0.770 | 0.417 |
| Random Forest | 0.383 ± 0.022 | 0.359 | 0.768 | 0.389 |
| Gradient Boosting | 0.370 ± 0.020 | 0.372 | 0.772 | 0.415 |
| Logistic Regression | 0.338 ± 0.013 | 0.326 | 0.744 | 0.380 |
| Decision Tree | 0.327 ± 0.014 | 0.305 | 0.709 | 0.367 |

`PageValues` is excluded as a target-leakage risk; including it would raise CV AP from 0.385 to 0.754.

## Reproduce

Python 3.9+ (tested on 3.9 and 3.13). From this directory:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter nbconvert --execute --to notebook --inplace assignment1_pipeline.ipynb   # ~1 minute
python make_pdfs.py                                                             # report
jupyter nbconvert --to webpdf --allow-chromium-download \
    --TemplateExporter.extra_template_basedirs=pdf_template --template wrapped_webpdf \
    assignment1_pipeline.ipynb                                                  # notebook PDF
```

Seed 42 is used everywhere; all imputation, scaling and encoding is fitted inside the model pipeline. The first notebook PDF export downloads a headless Chromium (about 100 MB).

## Virtual lab and final PDF (pending)

Run the notebook in the virtual lab, take a screenshot of the **Setup** cell output (it prints host, OS, Python and library versions), save it as `figures/virtual_lab_screenshot.png`, and re-run `python make_pdfs.py`. The report's "Use of virtual lab" section picks the screenshot up automatically. Export `assignment1_pipeline.pdf` from the lab run so it includes the lab outputs. The saved public notebook and HTML redact the original laptop name. Review the setup output before sharing any new public exports.

## Dataset citation

Sakar, C. O. & Kastro, Y. (2018). *Online Shoppers Purchasing Intention Dataset*. UCI Machine Learning Repository. DOI: [10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q). Licensed under CC BY 4.0.
