# MTech Assignments

This repository keeps MTech coursework organized by assignment. Each subject and assignment has its own folder at the repository root.

## Assignment index

| Subject | Semester | Assignment | Topic | Folder |
|---|---:|---:|---|---|
| DNN | 1st | 1st | Classifying cat and dog photos with a Dense neural network | [`dnn-1st-sem-1st-assignment/`](dnn-1st-sem-1st-assignment/readme.md) |
| AI/ML | 1st | 1st | Predicting online shoppers' purchase intention | [`ai-ml-1st-sem-1st-assignment/`](ai-ml-1st-sem-1st-assignment/README.md) |

## DNN — 1st Semester, 1st Assignment

This assignment uses Keras to classify cat and dog photos with a Dense neural network. The images are resized to 32 × 32, and the network has two hidden Dense layers. The selected learning rate is `0.01`; the final model reached **66.85% test accuracy** (3,338 of 4,993 images).

Read the [assignment overview](dnn-1st-sem-1st-assignment/readme.md) or the [step-by-step explanation](dnn-1st-sem-1st-assignment/explaination.md). The folder also contains the [notebook](dnn-1st-sem-1st-assignment/DL_assignment_Student2026CS05046.ipynb) and its [PDF export](dnn-1st-sem-1st-assignment/DL_assignment_student2026CS05046.pdf).

## AI/ML — 1st Semester, 1st Assignment

This assignment builds a purchase-intent classification pipeline using the UCI Online Shoppers Purchasing Intention dataset. It compares five scikit-learn models with grid search and five-fold stratified cross-validation. Extra Trees is selected by mean cross-validation average precision (`0.3849`); its saved test average precision is `0.3694`.

Read the [assignment overview](ai-ml-1st-sem-1st-assignment/README.md) and its [step-by-step explanation](ai-ml-1st-sem-1st-assignment/explaination.md). The folder includes an [HTML notebook export](ai-ml-1st-sem-1st-assignment/assignment1_pipeline.html), source data, saved metrics, and figures.

Add future subjects and assignments as separate folders at the repository root.
