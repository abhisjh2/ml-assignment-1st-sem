# Assignment 1 — Team Progress and Methods Report

**Project:** End-to-End Machine Learning Pipeline  
**Problem:** Predict whether an online shopping session results in a purchase.  
**Track:** A — scikit-learn models  
**Current status:** Core pipeline implemented; notebook run top-to-bottom; comparison results and report exported.

## 1. Objective and motivation

The goal is to estimate purchase intent from session-level browsing behavior and context. A retailer could use a well-calibrated estimate to prioritize optional assistance. False positives can create unnecessary interventions, while false negatives can miss relevant opportunities. This is a supervised binary classification problem: `Revenue=True` is the positive class. The analysis is predictive and observational; it does not establish that a browsing behavior causes a purchase.

## 2. Dataset and source

We selected the UCI Online Shoppers Purchasing Intention Dataset, an open, non-toy dataset in the e-commerce domain. The data contains 12,330 sessions, 17 input columns and one outcome. Inputs cover administrative, informational and product pages and their durations; bounce/exit rates; page value; special-day proximity; browser, operating system, region, traffic type; visitor type; month; and weekend. The outcome has 1,908 purchases and 10,422 non-purchases (15.48% positive). There are no missing cells; 125 exact duplicate rows were identified. We retain duplicates because identical feature values may represent distinct sessions, and no session identifier is provided to establish they are redundant.

**Source:** Sakar, C. O. & Kastro, Y. (2018), *Online Shoppers Purchasing Intention Dataset*, UCI Machine Learning Repository, DOI [10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q), [dataset page](https://archive.ics.uci.edu/dataset/468/online%2Bshoppers%2Bpurchasing%2Bintention), CC BY 4.0. The repository includes the original CSV and a reproducible 100-row sample.

## 3. EDA and observations

The notebook includes class-balance counts, numeric summaries, boxplots for duration and rate variables, and purchase-rate comparisons by month and visitor type. The positive class is the minority (about one in six sessions), so accuracy alone would obscure purchase detection quality. Numeric session measures have skew and outlying values; these may reflect long or unusual sessions and were not dropped without evidence that they are data errors. Purchase rates vary across observed months and visitor types, which supports retaining these contextual categorical features. These are descriptive associations and may reflect confounding or dataset-specific behavior.

## 4. Cleaning and feature engineering

- Converted `Weekend` to categorical text and `Revenue` to a 0/1 target.
- Verified missing-value count; none was observed. Median numeric and most-frequent categorical imputers remain in the pipeline as safeguards if input data changes.
- Counted, but retained, 125 exact duplicate rows; there is no session key to determine whether they are repeated records or distinct sessions with identical attributes.
- Added `TotalPages`, `TotalDuration`, and `ProductTimePerPage` features.
- Applied `log1p` to nonnegative page-count and duration features to reduce skew. Bounded rate fields were left on their original scale.
- Excluded `PageValues` from predictive features because its definition can encode downstream purchase information and create target leakage.
- One-hot encoded categories and standardized numeric variables. Both transformations are inside the cross-validated pipeline so they are fitted only on each training fold.
- Used an 80/20 stratified train/test split with random seed 42. Class weights address imbalance during model fitting.

## 5. Models and tuning

The training set was evaluated with five-fold shuffled stratified cross-validation. Grid search maximized average precision (AP), emphasizing ranking quality for the minority purchase class. Four model families were trained: Logistic Regression, Decision Tree, Random Forest, and Extra Trees. Each has a small documented parameter grid in the notebook. The holdout set was kept outside model selection. The selected model is the one with the highest mean CV AP.

## 6. Results

| Model | CV AP | Test AP | Test ROC-AUC | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.3849 | 0.3574 | 0.7665 | 0.3229 | 0.4607 | 0.3797 | 0.7668 |
| Extra Trees | 0.3827 | 0.3634 | 0.7701 | 0.2936 | 0.6518 | 0.4049 | 0.7032 |
| Logistic Regression | 0.3331 | 0.3142 | 0.7359 | 0.2502 | 0.7487 | 0.3751 | 0.6135 |
| Decision Tree | 0.3218 | 0.2963 | 0.7102 | 0.2317 | 0.6885 | 0.3467 | 0.5981 |

CV AP selects Random Forest (0.3849), while Extra Trees has the highest holdout AP (0.3634), ROC-AUC (0.7701), recall (0.6518), and F1 (0.4049). The difference between CV and holdout rankings is a reminder that one holdout sample is uncertain. For an assignment-consistent, leakage-aware choice, Random Forest remains the final model because selection was pre-defined on training CV AP. If missing purchases is more costly, Extra Trees may be preferable at the default threshold due to higher recall, but precision is lower. Accuracy is not the selection metric given class imbalance. The notebook includes confusion matrix, classification report, precision-recall and ROC curves, plus a comparison plot.

## 7. Limitations and future work

- The dataset is older and from a single source; performance may not generalize to current shoppers or other retailers.
- A random split can place temporally similar sessions in both sets. A time-based validation would better approximate future deployment.
- The available fields omit product, price, campaign, consent, and other context. The model may capture correlations that are unstable or inappropriate for intervention.
- Class weighting and a default 0.5 threshold do not encode business costs. Calibrate probabilities and set the threshold using an explicit cost/benefit analysis.
- Review performance by visitor and device segments, privacy and consent constraints, and potential disparate impact before any operational use.
- Future work: time-separated validation; threshold selection based on documented intervention costs; probability calibration; feature stability and subgroup checks; comparison with gradient boosting if permitted by course scope.

## 8. Reproduction and team handoff

Run `pip install -r requirements.txt`, then `jupyter nbconvert --execute --to notebook --inplace assignment1_pipeline.ipynb`. Seed 42 is fixed in the notebook. Results are written to `output/model_comparison.csv`; notebook plots are included as executed outputs and exported under `figures/`. Review the notebook source and CSV if rerunning under a different library version, since model metrics can vary slightly across versions.

**Deliverables in this repository:** runnable notebook with outputs; short report PDF; printable notebook source PDF; raw dataset and sample; comparison CSV and figures; pinned direct dependencies. The project progress report is also maintained on the team Confluence page: [ML Assignment 1st sem](https://cisco-sbg.atlassian.net/wiki/spaces/~63b6ebe115d69a40aa1ac8b6/pages/1550485916/ML+Assignment+1st+sem).
