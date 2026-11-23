"""Build the short assignment report (assignment1_report.pdf).

All model numbers are read from the files the notebook writes, so run the notebook first:
    jupyter nbconvert --execute --to notebook --inplace assignment1_pipeline.ipynb
    python make_pdfs.py

The executed notebook itself is exported separately (with outputs) via:
    jupyter nbconvert --to webpdf --allow-chromium-download assignment1_pipeline.ipynb
"""
import ast
import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (Image, KeepTogether, ListFlowable, ListItem, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

ROOT = Path(__file__).parent
FIG = ROOT / 'figures'
NAVY = colors.HexColor('#17365D')
CONTENT_WIDTH = 174 * mm

# --- Inputs produced by the notebook ----------------------------------------------------------
results = pd.read_csv(ROOT / 'output' / 'model_comparison.csv')
summary = json.loads((ROOT / 'output' / 'summary.json').read_text())
best = results.iloc[0]                        # CSV is sorted by CV AP (the selection rule)
by_name = results.set_index('Model')
best_params = ast.literal_eval(best['Best params'])
boosting = by_name.loc['Gradient Boosting']

# --- Styles ----------------------------------------------------------------------------------
styles = getSampleStyleSheet()
styles.add(ParagraphStyle('TitleCustom', parent=styles['Title'], fontSize=19, leading=23,
                          alignment=TA_CENTER, textColor=NAVY, spaceAfter=4))
styles.add(ParagraphStyle('SubTitle', parent=styles['Normal'], alignment=TA_CENTER, fontSize=9.5,
                          textColor=colors.HexColor('#555555'), spaceAfter=10))
styles.add(ParagraphStyle('Section', parent=styles['Heading2'], fontSize=12.5, textColor=NAVY,
                          spaceBefore=7, spaceAfter=4))
styles.add(ParagraphStyle('Body', parent=styles['BodyText'], fontSize=9, leading=12, spaceAfter=5))
styles.add(ParagraphStyle('Caption', parent=styles['BodyText'], fontSize=8, leading=10,
                          textColor=colors.HexColor('#444444'), spaceAfter=8))
styles.add(ParagraphStyle('Cell', parent=styles['BodyText'], fontSize=7.6, leading=9.4))


def para(text):
    return Paragraph(text, styles['Body'])


def heading(text):
    return Paragraph(text, styles['Section'])


def bullets(items):
    return ListFlowable([ListItem(para(t), leftIndent=10) for t in items],
                        bulletType='bullet', start='•', leftIndent=10, bulletFontSize=8)


def figure(name, caption, width=CONTENT_WIDTH):
    """Image scaled to `width` with its true aspect ratio, kept together with its caption."""
    path = FIG / f'{name}.png'
    w, h = ImageReader(str(path)).getSize()
    return KeepTogether([Image(str(path), width=width, height=width * h / w),
                         Paragraph(caption, styles['Caption'])])


def table(rows, col_widths):
    """Grid table with a navy header row; every cell is a Paragraph so long text wraps."""
    cells = [[Paragraph(str(c), styles['Cell']) for c in r] for r in rows]
    t = Table(cells, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('GRID', (0, 0), (-1, -1), 0.35, colors.HexColor('#A6A6A6')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#EAF0F7')]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 3), ('BOTTOMPADDING', (0, 0), (-1, -1), 3)]))
    return t


def white(text):
    """Header-cell markup (Paragraph cells ignore the table's TEXTCOLOR)."""
    return f'<font color="white"><b>{text}</b></font>'


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(colors.grey)
    canvas.drawString(18 * mm, 12 * mm, 'Assignment 1 | Online Shoppers Purchase Intention')
    canvas.drawRightString(192 * mm, 12 * mm, str(doc.page))
    canvas.restoreState()


f3 = lambda x: f'{x:.3f}'
story = [
    Paragraph('End-to-End Machine Learning Pipeline', styles['TitleCustom']),
    Paragraph('Predicting Online Shoppers’ Purchase Intention | Assignment 1 | Track A (scikit-learn)',
              styles['SubTitle']),

    heading('1. Problem and motivation'),
    para('The task is to predict whether an e-commerce browsing session ends in a purchase (<i>Revenue</i>) '
         'from session behaviour and context. A retailer could use such a purchase-intent score to prioritise '
         'optional help (live chat, reminders, offers): false positives waste interventions, false negatives '
         f'miss sales. Only {summary["base_rate"]:.1%} of sessions convert, so accuracy is misleading; average '
         'precision (AP) is the primary metric, as precision-recall measures are more informative than ROC '
         'on imbalanced data [4], with ROC-AUC, precision, recall, F1 and accuracy reported '
         'alongside. The analysis is predictive, not causal.'),

    heading('2. Dataset and justification'),
    para('<b>UCI Online Shoppers Purchasing Intention Dataset</b> (Sakar &amp; Kastro, 2018; CC BY 4.0): 12,330 '
         'sessions from one year of a single retailer, 17 predictors and a binary outcome (1,908 purchases, '
         '10,422 non-purchases). Predictors: page counts and durations for administrative, informational and '
         'product pages; bounce and exit rates; <i>PageValues</i>; special-day proximity; integer-coded operating '
         'system, browser, region and traffic type; visitor type; month; weekend flag. It is not on the avoided '
         'list and needs real preparation: strong class imbalance, zero-inflated and heavily skewed counts with '
         'extreme outliers (one session lasts 17.8 h), four integer-coded nominal variables with rare levels '
         'plus three text categoricals, 125 exact duplicates, and a target-leakage question around '
         '<i>PageValues</i>. Source: https://doi.org/10.24432/C5F88Q.'),

    heading('3. Exploratory data analysis'),
    figure('eda_numeric_by_outcome', 'Figure 1. Numeric features by outcome (log1p scale for counts and '
           'durations). Buyers view more product pages and exit less; PageValues separates the classes almost '
           'perfectly.', width=140 * mm),
    bullets([
        '<b>Engagement:</b> buyers have a median of 29 product pages vs 16 for non-buyers, and 1,110 s vs 510 s of '
        'product-page time; median exit rate is 0.016 vs 0.029. Only 0.4% of sessions with the maximum bounce '
        'rate (0.2) buy.',
        '<b>Skew and outliers:</b> 79% of sessions view no informational page; skewness is 2.0–7.6 and the '
        'IQR rule flags 3–21% of rows per feature — a natural long tail rather than errors, so outliers '
        'are kept and log-transformed.',
        '<b>Redundancy:</b> each page count and its duration correlate strongly (Spearman ρ = 0.88–0.95; '
        'Figure 2). Apart from PageValues (ρ = 0.63), no feature correlates '
        'with the target above |0.25|.',
        '<b>Context (Figure 3):</b> strong seasonality (1.6% conversion in February, 25.4% in November); new '
        'visitors convert at 24.9% vs 13.9% for returning; traffic type 8 converts at 27.7% vs 5.8% for type 13; '
        'region barely matters (12.9–16.8%). Sessions near a special day convert less (3.4–8.3% vs '
        '16.5%), and the effect persists within May (6.4% vs 13.2%).',
        f'<b>Leakage:</b> PageValues alone gives ROC-AUC {summary["pagevalues_alone_auc"]:.3f}, with 56.3% '
        'conversion when non-zero vs 3.9% otherwise. It is attributed from pages preceding a completed '
        'transaction, so it is excluded from the features.',
    ]),
    figure('eda_correlation', 'Figure 2. Spearman correlations of numeric features and the target.',
           width=105 * mm),
    figure('eda_purchase_rate_by_category', 'Figure 3. Purchase rate by categorical context; dashed line = '
           'overall rate.'),

    heading('4. Cleaning, preprocessing and feature engineering'),
    table([[white('Issue'), white('Treatment')],
           ['No missing cells; engineered ratio undefined for 38 sessions with no product pages',
            'Median (numeric) / most-frequent (categorical) imputation inside the pipeline'],
           ['125 exact duplicates (all non-purchase single-page bounces)',
            'Retained — no session ID to prove they are repeats'],
           ['Skewed, zero-inflated counts and durations; outliers',
            'Kept; log1p transform; standardisation for the linear model'],
           ['OperatingSystems, Browser, Region, TrafficType are integer codes',
            'Treated as categories and one-hot encoded; levels with fewer than 20 training rows pooled as '
            '"infrequent"'],
           ['Month, VisitorType, Weekend', 'One-hot encoded'],
           ['PageValues leaks the outcome', 'Excluded (sensitivity check in Section 7)'],
           ['Class imbalance (15.5% positive)', 'Stratified split and folds; class-weighted models; AP for tuning'],
           ], [78 * mm, 96 * mm]),
    Spacer(1, 4),
    para('Four features were engineered: total pages, total duration, average time per product page, and the '
         'share of session time spent on product pages. All transformations live in a scikit-learn '
         '<i>Pipeline</i>/<i>ColumnTransformer</i>, so they are re-fitted on the training part of every CV fold. '
         'An 80/20 stratified hold-out (seed 42) is kept untouched until final evaluation.'),

    heading('5. Models, cross-validation and tuning'),
    para('Five algorithm families were tuned with <i>GridSearchCV</i> using 5-fold shuffled stratified CV on '
         'the training set, maximising AP. All use balanced class weights.'),
    table([[white('Model'), white('Grid searched'), white('Selected hyperparameters')]] +
          [[name, grid, ', '.join(f'{k}={v}' for k, v in ast.literal_eval(by_name.loc[name, 'Best params']).items())]
           for name, grid in [
               ('Logistic Regression', 'C ∈ {0.01, 0.1, 1, 10}; L1/L2 penalty'),
               ('Decision Tree', 'depth ∈ {4, 6, 8, 12}; min leaf ∈ {5, 20, 50}; gini/entropy'),
               ('Random Forest', '300 trees; depth ∈ {None, 16}; min leaf ∈ {2, 8}; max features ∈ {√p, 0.5}'),
               ('Extra Trees', 'same grid as Random Forest'),
               ('Gradient Boosting', 'HistGradientBoosting, early stopping; learning rate ∈ {0.05, 0.1}; '
                'leaves ∈ {15, 31}; min leaf ∈ {20, 50}; L2 ∈ {0, 1}')]],
          [34 * mm, 78 * mm, 62 * mm]),
    para('<b>Histogram Gradient Boosting.</b> The classifier adds trees one at a time to reduce prediction '
         'loss. It groups numeric values into bins to speed up finding splits. We compared 16 settings. '
         'Training used up to 300 boosting iterations and stopped early if the loss on an internal '
         'validation set failed to improve sufficiently for 20 consecutive iterations.'),

    heading('6. Results and model comparison'),
    table([[white(h) for h in ['Model', 'CV AP (± sd)', 'Test AP', 'ROC-AUC', 'Precision', 'Recall',
                               'F1', 'Accuracy']]] +
          [[r['Model'], f'{r["CV AP"]:.3f} ± {r["CV AP std"]:.3f}', f3(r['Test AP']), f3(r['Test ROC-AUC']),
            f3(r['Test Precision']), f3(r['Test Recall']), f3(r['Test F1']), f3(r['Test Accuracy'])]
           for _, r in results.iterrows()],
          [34 * mm, 26 * mm, 18 * mm, 19 * mm, 19 * mm, 18 * mm, 18 * mm, 22 * mm]),
    Paragraph('Table: models sorted by CV AP (the selection criterion). Test metrics are on the 20% hold-out; '
              'precision/recall/F1/accuracy use the 0.5 threshold.', styles['Caption']),
    figure('roc_pr_all_models', 'Figure 4. ROC and precision-recall curves of all five tuned models on the '
           'hold-out set, with chance baselines.'),
    figure('confusion_matrices', 'Figure 5. Hold-out confusion matrices at the default threshold.'),
    para(f'All models clearly beat chance (AP {results["Test AP"].min():.2f}–{results["Test AP"].max():.2f} '
         f'vs {summary["base_rate"]:.3f}). The three ensembles lie within '
         f'{results["CV AP"].iloc[0] - results["CV AP"].iloc[2]:.3f} CV AP of each other, less than their '
         'fold-to-fold standard deviation, and clearly ahead of logistic regression and the single tree, '
         'indicating useful non-linear interactions. The confusion matrices show different operating points '
         'rather than different ranking ability: the PR curves overlap, so the threshold should be set from '
         'intervention costs. Accuracy is highest for the model that predicts "buy" least often, which is why '
         'it is not used for selection.'),
    figure('model_comparison', 'Figure 6. Summary comparison across models (dotted line = AP of a random '
           'ranking).', width=120 * mm),

    heading('7. Final model and interpretation'),
    para(f'<b>Selected model: {best["Model"]}</b> '
         f'({", ".join(f"{k}={v}" for k, v in best_params.items())}). It has the highest mean CV AP '
         f'({best["CV AP"]:.3f} ± {best["CV AP std"]:.3f}), the criterion set before testing. Its hold-out '
         f'AP is {best["Test AP"]:.3f} and its F1 is {best["Test F1"]:.3f} at the 0.5 threshold.'),
    para(f'<b>Gradient Boosting comparison.</b> Mean CV AP was {boosting["CV AP"]:.3f}, with hold-out '
         f'AP of {boosting["Test AP"]:.3f} and ROC-AUC of {boosting["Test ROC-AUC"]:.3f}. At the 0.5 threshold '
         f'it found {boosting["Test Recall"]:.1%} of buyers, and {boosting["Test Precision"]:.1%} of its '
         'purchase predictions were correct. Model selection followed the mean CV AP ranking.'),
    para('Permutation importance (drop in hold-out AP when a column is shuffled) ranks '
         + ', '.join(f'{d["feature"]} ({d["mean AP drop"]:.3f})' for d in summary['top_features'])
         + ' highest, matching the EDA: seasonality, visitor type and engagement drive predictions. '
         f'<b>Sensitivity:</b> adding PageValues to the same model would raise CV AP from '
         f'{summary["cv_ap_without_pagevalues"]:.3f} to {summary["cv_ap_with_pagevalues"]:.3f}. Much of the '
         'high performance often quoted for this dataset therefore comes from that one leakage-prone field; '
         'our results are a more realistic estimate of what can be predicted before a transaction.'),
    figure('permutation_importance', f'Figure 7. Permutation importance for the selected {best["Model"]} '
           'model (mean ± sd over 10 shuffles).', width=95 * mm),

    heading('8. Conclusions, limitations and future work'),
    para(f'Without PageValues, session behaviour gives a useful but modest signal: the selected model ranks '
         f'sessions about {best["Test AP"] / summary["base_rate"]:.1f}× better than random (hold-out AP '
         f'{best["Test AP"]:.2f}) with ROC-AUC {best["Test ROC-AUC"]:.2f}.'),
    bullets([
        '<b>Limitations:</b> single retailer and one year of older data; a random split mixes months across '
        'train and test even though month is the strongest feature (January and April are absent); the 0.5 '
        'threshold is not cost-based and class weighting leaves probabilities uncalibrated; anonymised codes '
        'limit actionability; permutation importance splits credit between correlated features; compact grids '
        'cannot separate the ensembles statistically; associations are not causes.',
        '<b>Future work:</b> time-based validation; cost-based threshold selection on out-of-fold predictions '
        'and probability calibration; repeated or nested CV and wider search, including XGBoost, LightGBM or '
        'CatBoost; a leakage-free "page value so far" feature or sequence models on raw clickstreams; segment, '
        'privacy and consent review before any deployment.',
    ]),

    heading('9. Use of virtual lab'),
]

lab_shot = FIG / 'virtual_lab_screenshot.png'
if lab_shot.exists():
    story += [figure('virtual_lab_screenshot', 'Figure 8. The notebook executed in the virtual lab '
                     '(environment cell showing host, Python and library versions).', width=120 * mm)]
else:
    story += [para('<font color="#B00020"><b>[To be completed]</b> Run the notebook in the virtual lab and save a '
                   'screenshot of the environment cell (Section 2 of the notebook) as '
                   '<i>figures/virtual_lab_screenshot.png</i>, then re-run <i>python make_pdfs.py</i>.</font>')]

story += [
    heading('10. Reproducibility and references'),
    para('Deliverables: executed notebook (<i>assignment1_pipeline.ipynb</i> and its PDF export with outputs), '
         'this report, the full dataset and a 100-row sample, <i>output/model_comparison.csv</i>, '
         '<i>output/summary.json</i> and all figures. Install the pinned <i>requirements.txt</i> (Python 3.9 or '
         '3.13 tested), execute the notebook from the repository root, then run <i>python make_pdfs.py</i>. Seed '
         '42 throughout.'),
    para('[1] Sakar, C. O. &amp; Kastro, Y. (2018). <i>Online Shoppers Purchasing Intention Dataset</i>. UCI '
         'Machine Learning Repository. https://doi.org/10.24432/C5F88Q<br/>'
         '[2] Sakar, C. O., Polat, S. O., Katircioglu, M. &amp; Kastro, Y. (2019). Real-time prediction of online '
         'shoppers’ purchasing intention using multilayer perceptron and LSTM recurrent neural networks. '
         '<i>Neural Computing and Applications</i>, 31, 6893–6908.<br/>'
         '[3] Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. <i>JMLR</i>, 12, 2825–2830. '
         'Documentation for the version used (1.6.1): https://scikit-learn.org/1.6/<br/>'
         '[4] Saito, T. &amp; Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC '
         'plot when evaluating binary classifiers on imbalanced datasets. <i>PLOS ONE</i>, 10(3), e0118432.'),
]

doc = SimpleDocTemplate(str(ROOT / 'assignment1_report.pdf'), pagesize=A4, rightMargin=18 * mm,
                        leftMargin=18 * mm, topMargin=15 * mm, bottomMargin=20 * mm,
                        title='End-to-End ML Pipeline: Online Shoppers Purchase Intention')
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print('wrote assignment1_report.pdf')
