# DNN Assignment 1 — Cats vs Dogs

This first-semester assignment trains a Dense neural network in Keras to classify photos as cats or dogs. The notebook reads the Microsoft PetImages dataset, removes duplicate images, and turns each photo into a row of 3,072 scaled RGB values.

The model has two hidden Dense layers with ReLU and a two-unit softmax output. It compares two learning rates using five-fold cross-validation, then trains the final model with SGD for 20 epochs. The saved run classified **3,338 of 4,993 held-out photos correctly (66.85%)**.

## Files

- [Jupyter notebook](DL_assignment_Student2026CS05046.ipynb) — Code, saved results, and charts.
- [PDF assignment](DL_assignment_student2026CS05046.pdf) — Printable version of the completed notebook.
- [Detailed explanation](explaination.md) — A step-by-step guide to the code and results.

## Run the notebook

Open the notebook in Jupyter or Colab and run the cells in order. For a local Python environment, install the dependencies first.

```bash
python -m pip install tensorflow scikit-learn pillow matplotlib
```

The notebook downloads the dataset on the first run. The archive is about 787 MiB.
