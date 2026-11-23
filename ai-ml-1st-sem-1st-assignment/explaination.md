# Step-by-Step Explanation

## Project in one sentence

This project uses session details from an online shop to estimate whether a session ends with a purchase. The answer is the `Revenue` column: `1` means a purchase and `0` means no purchase. The notebook walks through the full machine-learning process, from checking the data to comparing and interpreting models.

## Step 1: Define the prediction problem

The task is **binary classification** because there are two possible outcomes: purchase or no purchase. Each row represents a shopping session, and the model learns patterns in the session fields that are associated with the outcome.

The model finds patterns; it does not prove that a particular browsing behavior causes someone to buy. For example, a long product-page visit may be associated with purchases without being the reason for them.

## Step 2: Load the dataset and check its condition

The notebook uses the UCI Online Shoppers Purchasing Intention Dataset. It has 12,330 rows and 18 columns: 17 input fields and the `Revenue` target.

Before modeling, the notebook checks the data shape, missing values, and exact duplicates. It finds no missing cells and 125 exact duplicate rows. The duplicates are retained because the dataset has no session identifier that would prove they are accidental copies.

## Step 3: Record the software environment

The setup section imports the required Python libraries, sets random seed `42`, and prints the Python, operating system, machine, and library versions. That environment record makes it easier to identify where a run happened and which software versions it used.

A fixed seed makes random parts of the workflow repeatable in the same software setup, such as the train/test split and cross-validation folds. Different library versions can still produce small result differences.

## Step 4: Explore the data before building a model

Exploratory data analysis (EDA) helps reveal what is in the dataset and where modeling mistakes could happen. The notebook checks:

- how many sessions end in a purchase;
- how numeric values are distributed and whether they have extreme values;
- how numeric fields relate to each other; and
- how purchase rates vary across categories such as month and visitor type.

Only about 15% of sessions end in a purchase. This is called **class imbalance**: the purchase class is much smaller than the no-purchase class. A model that always predicts “no purchase” would be roughly 85% accurate while finding no buyers at all, so accuracy cannot be the only measure of success.

## Step 5: Check for target leakage

**Target leakage** happens when an input contains information that would not really be available at prediction time, or that partly gives away the answer. The notebook checks `PageValues`, a field that may reflect purchase activity recorded during or after a session.

The main model leaves `PageValues` out. Later, the notebook adds it back in a separate training-only sensitivity check to see how much it changes the score. This helps show why a suspiciously strong result can be misleading.

## Step 6: Prepare fields and create useful features

Some values are stored as numbers but represent labels. For example, browser code `2` is a browser category, not “twice as much browser” as code `1`. The notebook converts operating system, browser, region, traffic type, and weekend values into categories.

It also creates a few combined features:

- `TotalPages`: total administrative, informational, and product pages viewed;
- `TotalDuration`: total time on those page types;
- `ProductTimePerPage`: product-page time divided by product-page count; and
- `ProductDurationShare`: the share of total page time spent on product pages.

Page counts and durations are highly skewed: most sessions have smaller values, while a few have very large values. The notebook applies `log1p` to selected nonnegative counts and durations. This compresses large values while keeping zero values valid. Undefined per-page values are left for the imputer to handle.

## Step 7: Keep a final test set aside

The notebook splits the data into 80% training data and 20% test data. It uses a **stratified split**, which keeps about the same purchase rate in both parts, and random seed `42` so the split can be reproduced.

The training data is used to fit and compare models. The test data stays untouched until the final evaluation. This gives a fairer estimate of how a selected model might perform on sessions it has not seen.

## Step 8: Build preprocessing into the model pipeline

Machine-learning models need consistent input values. The notebook prepares numeric and categorical fields differently:

- Numeric fields: fill missing values with the training median, then standardize their scale.
- Categorical fields: fill missing values with the most common training value, then one-hot encode each category into indicator columns.

These steps are kept inside a scikit-learn `Pipeline`. During cross-validation, the preprocessing is fitted using only the training folds for that round. This prevents information from a validation fold or the final test set from influencing preprocessing. That is another form of leakage the pipeline is designed to avoid.

## Step 9: Compare five model types

The notebook compares:

1. **Logistic Regression** — a simple linear baseline.
2. **Decision Tree** — a set of branching rules.
3. **Random Forest** — many trees trained with randomness and combined.
4. **Extra Trees** — an ensemble of trees with additional randomness in how splits are chosen.
5. **Histogram Gradient Boosting** — trees added in sequence, with later trees correcting earlier errors.

Each model has a small **hyperparameter grid**. Hyperparameters are settings chosen before fitting, such as a tree's maximum depth. `GridSearchCV` tries the listed combinations rather than relying on one arbitrary setting. Class weights are used to account for the smaller purchase class.

## Step 10: Tune models with cross-validation

The training data is evaluated with **five-fold stratified cross-validation**. In each round, the training data is divided into five parts: four parts fit the model, and the remaining part checks it. Each part takes a turn as the validation fold. The five scores are averaged.

The notebook uses **average precision (AP)** to choose settings. AP summarizes the precision-recall trade-off across probability thresholds and is useful when the positive class is relatively rare. Cross-validation also reports a standard deviation, which shows how much the score varies across folds.

## Step 11: Evaluate the tuned models on the test set

After tuning, each best configuration is evaluated once on the held-out test data. The notebook reports several metrics because they answer different questions:

- **Test AP:** how well the model ranks purchase sessions ahead of non-purchase sessions across thresholds.
- **ROC-AUC:** another measure of ranking quality, comparing true positive and false positive rates.
- **Precision:** among sessions predicted as purchases, how many actually purchased?
- **Recall:** among actual purchases, how many did the model find?
- **F1:** a combined measure of precision and recall.
- **Accuracy:** the share of all predictions that are correct; it can hide poor purchase detection when classes are imbalanced.

Precision, recall, F1, and accuracy use the default probability threshold of `0.5`. AP and ROC-AUC evaluate ranking over a range of thresholds.

## Step 12: Select the final model using the rule set in advance

The rule is to select the model with the highest **mean cross-validation AP**, rather than whichever one happens to score highest on the test set. In the saved `output/model_comparison.csv`, **Extra Trees** is selected with CV AP `0.3849 ± 0.0277`; its test AP is `0.3694` and its F1 is `0.4165`.

Histogram Gradient Boosting has slightly higher test AP (`0.3719`), but choosing it because of that result would use the test set for model selection. Keeping the test set out of that decision preserves its role as a final check. The top three models are close, so the small ranking differences should not be treated as proof that one model is definitively better.

## Step 13: Interpret the model and revisit leakage

The notebook uses **permutation importance** to estimate which input fields the selected model relies on. It shuffles one field at a time and measures how much test AP falls. In the saved results, the leading fields are `Month`, `VisitorType`, `ExitRates`, `ProductRelated_Duration`, and `TrafficType`.

This is a measure of how useful a field is to this model on this test set. It does not show that the field causes purchases, and correlated fields can affect one another's importance scores.

The `PageValues` sensitivity check raises CV AP from about `0.3849` to about `0.7539` when the field is included. That large increase supports keeping it out of the main model because it may reveal information too closely tied to the purchase outcome.

## Step 14: Save and reproduce the work

The notebook saves the comparison table to `output/model_comparison.csv`, summary values to `output/summary.json`, and charts to `figures/`. The current folder contains the HTML export, dataset, results, and charts. The notebook file, PDF exports, short report, and virtual-lab screenshot are currently absent from the local working tree.

From this folder, install the dependencies and rerun the notebook before regenerating the short report:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter nbconvert --execute --to notebook --inplace assignment1_pipeline.ipynb
python make_pdfs.py
```

The notebook commands above assume `assignment1_pipeline.ipynb` has been restored to this folder. Rerunning it refreshes the generated metrics so the CSV, summary, and report use the same run.

## Main idea to remember

The assignment is not only about getting a high score. It is about defining a useful prediction target, checking the data, avoiding leakage, comparing models fairly, choosing a metric that fits the class imbalance, and explaining what the results can and cannot tell us.
