# DNN Assignment 1 - Cats vs Dogs

For my first DNN assignment, I used Keras to train a model that tells cats and dogs apart in photos. I resized the photos to 32 × 32 and gave their pixel values to a network with two hidden Dense layers.

I tried two learning rates using five-fold cross-validation. After picking `0.01`, I trained the final model for 20 epochs with SGD. It got **3,338 out of 4,993 test photos right (66.85%)**.

## Files

- [Notebook](DL_assignment_Student2026CS05046.ipynb) with the code, plots, and results.
- [PDF](DL_assignment_student2026CS05046.pdf) of the completed notebook.
- [Detailed explanation](explaination.md) of what I did and how I read the results.

## Running it

Open the notebook in Jupyter or Colab and run the cells in order. For a local setup, install the packages below first. The dataset download is about 787 MiB.

```bash
python -m pip install tensorflow scikit-learn pillow matplotlib
```
