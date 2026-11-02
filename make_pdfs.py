"""Export the assignment report and notebook source to printable PDFs."""
from pathlib import Path
import json
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                Image, PageBreak, Preformatted, KeepTogether)

ROOT = Path(__file__).parent
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleCustom', parent=styles['Title'], fontSize=20, leading=24,
                          alignment=TA_CENTER, textColor=colors.HexColor('#17365D'), spaceAfter=8))
styles.add(ParagraphStyle(name='SubTitle', parent=styles['Normal'], alignment=TA_CENTER,
                          fontSize=10, textColor=colors.HexColor('#555555'), spaceAfter=16))
styles.add(ParagraphStyle(name='Section', parent=styles['Heading2'], textColor=colors.HexColor('#17365D'),
                          spaceBefore=5, spaceAfter=6))
styles.add(ParagraphStyle(name='BodySmall', parent=styles['BodyText'], fontSize=9, leading=12, spaceAfter=6))
styles.add(ParagraphStyle(name='CodeSmall', fontName='Courier', fontSize=6.2, leading=7.3))

def para(text): return Paragraph(text, styles['BodySmall'])
def heading(text): return Paragraph(text, styles['Section'])
def footer(canvas, doc):
    canvas.saveState(); canvas.setFont('Helvetica', 8); canvas.setFillColor(colors.grey)
    canvas.drawString(18*mm, 12*mm, 'Assignment 1 | Online Shoppers Purchase Intention')
    canvas.drawRightString(192*mm, 12*mm, str(doc.page)); canvas.restoreState()

# Short report: four intentional pages.
report = SimpleDocTemplate(str(ROOT/'assignment1_report.pdf'), pagesize=A4,
    rightMargin=18*mm, leftMargin=18*mm, topMargin=16*mm, bottomMargin=20*mm)
story = [Paragraph('End-to-End Machine Learning Pipeline', styles['TitleCustom']),
         Paragraph('Online Shoppers Purchasing Intention | Assignment 1 | Track A', styles['SubTitle']),
         heading('1. Problem and motivation'),
         para('This project predicts whether an online shopping session ends in a purchase (Revenue). A retailer could use purchase-intent estimates to prioritize optional assistance. False positives can waste interventions and false negatives can miss opportunities. This is an observational prediction task; model associations do not show that browsing behavior causes purchases.'),
         heading('2. Dataset and source'),
         para('The UCI Online Shoppers Purchasing Intention Dataset contains 12,330 sessions, 17 predictors and a binary outcome. Predictors cover browsing page counts and durations, bounce/exit rates, special-day proximity, device and traffic categories, visitor type, month, and weekend. There are 1,908 purchases (15.48%) and 10,422 non-purchases. The dataset has no missing cells and 125 exact duplicate rows. The duplicates are retained because the public data has no session identifier to prove they are redundant.'),
         para('Source: Sakar, C. O. & Kastro, Y. (2018), UCI Machine Learning Repository, DOI 10.24432/C5F88Q, CC BY 4.0. https://archive.ics.uci.edu/dataset/468/online%2Bshoppers%2Bpurchasing%2Bintention'),
         heading('3. Exploratory analysis'),
         para('The EDA examines class balance, numeric summaries and distributions, outlying session durations/rates, and purchase rates across month and visitor type. Purchase is the minority outcome, so accuracy alone is insufficient. Session counts and durations are skewed; extreme values are retained because they may describe real sessions. Differences across groups are descriptive and may reflect confounding.'),
         Image(str(ROOT/'figures/figure_1.png'), width=170*mm, height=57*mm),
         para('Figure 1. Class counts and numeric ranges used to inspect imbalance and extreme values.'),
         PageBreak(),
         heading('4. Preparation and feature engineering'),
         para('The target was converted to 0/1 and Weekend treated as categorical. Feature engineering added total pages, total duration, and product time per page. Nonnegative page-count and duration fields were log1p transformed. PageValues was excluded because it may act as a proxy for the outcome and introduce leakage. Exact duplicates were counted but retained. Median/mode imputers are included as pipeline safeguards even though this extract has no missing values.'),
         para('Numeric inputs were standardized and categorical inputs one-hot encoded. These operations live inside scikit-learn pipelines, so each cross-validation fold learns preprocessing only from its training partition. An 80/20 stratified split with seed 42 preserves class proportions. Models use class weights to account for imbalance.'),
         heading('5. Models and evaluation'),
         para('Four algorithm families were compared: Logistic Regression, Decision Tree, Random Forest, and Extra Trees. Hyperparameters were tuned with GridSearchCV over a documented compact grid and five-fold shuffled stratified cross-validation. Average precision (AP) was the tuning metric because it measures positive-class retrieval quality under imbalance. The final holdout set was not used during tuning.'),
         heading('6. Model comparison'),]
rows=[['Model','CV AP','Test AP','ROC-AUC','Precision','Recall','F1','Accuracy'],
      ['Random Forest','0.3849','0.3574','0.7665','0.3229','0.4607','0.3797','0.7668'],
      ['Extra Trees','0.3827','0.3634','0.7701','0.2936','0.6518','0.4049','0.7032'],
      ['Logistic Regression','0.3331','0.3142','0.7359','0.2502','0.7487','0.3751','0.6135'],
      ['Decision Tree','0.3218','0.2963','0.7102','0.2317','0.6885','0.3467','0.5981']]
t=Table(rows, colWidths=[36*mm,19*mm,20*mm,21*mm,20*mm,17*mm,16*mm,19*mm], repeatRows=1)
t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#17365D')),('TEXTCOLOR',(0,0),(-1,0),colors.white),
    ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),7.4),('ALIGN',(1,1),(-1,-1),'CENTER'),
    ('GRID',(0,0),(-1,-1),0.35,colors.HexColor('#A6A6A6')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#EAF0F7')]),
    ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
story += [t, Spacer(1,5), para('Random Forest has the strongest mean CV AP (0.3849), so it is selected using the pre-specified training-only criterion. Extra Trees has the highest holdout AP, ROC-AUC, recall, and F1. Its recall is higher (0.6518) but precision lower (0.2936), indicating a different threshold trade-off. Holdout ranking differences are expected from sampling variability; accuracy is not the model-selection criterion.'),
          Image(str(ROOT/'figures/figure_4.png'), width=170*mm, height=57*mm),
          para('Figure 2. Cross-validation and holdout ranking metrics alongside holdout F1.'), PageBreak(),
          heading('7. Interpretation, limitations and future work'),
          para('The results show useful ranking signal in session behavior, but the models do not establish purchase causes or justify automated interventions. A default classification threshold is only illustrative. Before operational use, calibrate probabilities and choose a threshold using documented intervention costs.'),
          para('The data is older and comes from one source. Random splitting can put sessions from similar periods in both train and test, so temporal generalization remains unknown. Product, price, campaign, and consent context are not represented. Performance may vary across devices or visitor segments, and correlations can change over time.'),
          para('Recommended next steps: use a time-separated validation set; compare calibration and threshold-dependent cost; inspect stability and performance by relevant segments; and confirm privacy, consent, and fairness requirements. A current, independently sourced dataset would be needed to support deployment claims.'),
          heading('8. Reproducibility and deliverables'),
          para('The repository includes the executed Jupyter notebook, pinned direct requirements, full UCI CSV, 100-row sample, comparison CSV, chart exports, this report, and a printable source-code PDF. Use Python 3.9+; install requirements and execute the notebook from the repository root. Random seed is 42. Results may shift slightly with dependency versions.'),
          para('Reference: Sakar, C. O. & Kastro, Y. (2018). Online Shoppers Purchasing Intention Dataset. UCI Machine Learning Repository. DOI: 10.24432/C5F88Q. CC BY 4.0.')]
report.build(story, onFirstPage=footer, onLaterPages=footer)

# Readable printable source export for the notebook deliverable.
nb=json.loads((ROOT/'assignment1_pipeline.ipynb').read_text())
code_lines=[]
for i,cell in enumerate(nb['cells'],1):
    code_lines.append(f'# CELL {i} — {cell["cell_type"].upper()}')
    src=''.join(cell.get('source',[])).splitlines()
    code_lines.extend(src or [''])
    code_lines.append('')
# Keep source monospaced and split long lines for the page width.
import textwrap
wrapped=[]
for line in code_lines:
    wrapped.extend(textwrap.wrap(line, width=105, subsequent_indent='    ', replace_whitespace=False, drop_whitespace=False) or [''])
code_doc=SimpleDocTemplate(str(ROOT/'assignment1_notebook_code.pdf'), pagesize=A4,
    rightMargin=12*mm,leftMargin=12*mm,topMargin=14*mm,bottomMargin=18*mm)
code_story=[Paragraph('Notebook Source Code',styles['TitleCustom']),
            para('Printable source export of assignment1_pipeline.ipynb. The notebook is the canonical runnable version and includes executed outputs.'),
            Preformatted('\n'.join(wrapped), styles['CodeSmall'], maxLineLength=110)]
code_doc.build(code_story,onFirstPage=footer,onLaterPages=footer)
print('wrote assignment1_report.pdf and assignment1_notebook_code.pdf')
