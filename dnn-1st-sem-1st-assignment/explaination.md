# Cats vs Dogs DNN Assignment Explained

The assignment trains a neural network to tell whether a photograph shows a cat or a dog. Each photo goes in as a row of pixel values. The model gives back two scores, one for each class. The class with the higher score becomes the prediction.

The saved run got **3,338 out of 4,993 test photos right**, for **66.85% test accuracy**. It used a Dense network with two hidden layers, trained with stochastic gradient descent (SGD). Five-fold cross-validation helped choose the learning rate before the final test.

The notebook has the runnable code, the saved cell outputs, and the plots. The PDF contains the same completed assignment in a format you can read without opening Jupyter. [Jupyter notebook](DL_assignment_Student2026CS05046.ipynb) · [PDF version](DL_assignment_student2026CS05046.pdf)

## Contents

1. [What the model learns](#1-what-the-model-learns)
2. [Where the photos come from](#2-where-the-photos-come-from)
3. [How the photos become numbers](#3-how-the-photos-become-numbers)
4. [How the data is split](#4-how-the-data-is-split)
5. [How the Dense network works](#5-how-the-dense-network-works)
6. [How training changes the weights](#6-how-training-changes-the-weights)
7. [How cross-validation chooses the learning rate](#7-how-cross-validation-chooses-the-learning-rate)
8. [What the training curves show](#8-what-the-training-curves-show)
9. [What the test results mean](#9-what-the-test-results-mean)
10. [What the assignment shows and where it is limited](#10-what-the-assignment-shows-and-where-it-is-limited)
11. [How to run the notebook again](#11-how-to-run-the-notebook-again)

## 1. What the model learns

This is **supervised learning** because the training examples come with answers. The answer, or **label**, is the type of animal in the photo. The notebook uses two labels, so the task is **binary classification**.

During training, the network makes a prediction, compares it with the known label, and adjusts its weights. It repeats that process across many labelled photos. Once training is over, the network takes the pixels of a test photo and predicts a label. The evaluation code uses the known test label to score the prediction.

The model assigns one label to the whole image. For a photo with multiple animals, it still returns one of the two classes. It has no output for the animal's position or for images outside these two classes.

The assignment asks for a Dense neural network. Every trainable layer in this notebook is a Dense layer. The notebook resizes and flattens each photo before handing it to the network. Image loading, resizing, and label encoding happen in the data preparation cells.

## 2. Where the photos come from

The notebook reads Microsoft's `kagglecatsanddogs_5340.zip` archive, the source data for the Cats vs Dogs entry in the TensorFlow Datasets catalog. The archive puts the JPEGs in two folders. `PetImages/Cat/` supplies the cat labels, and `PetImages/Dog/` supplies the dog labels. The order of `class_names = ["Cat", "Dog"]` gives cats the integer `0` and dogs the integer `1`. [TensorFlow Datasets catalog](https://www.tensorflow.org/datasets/catalog/cats_vs_dogs) · [Microsoft dataset download](https://www.microsoft.com/en-us/download/details.aspx?id=54765)

### What the loading code does

1. `keras.utils.get_file()` finds the ZIP in Keras's local cache or downloads it from Microsoft. The `file_hash` argument gives Keras the expected SHA-256 checksum for the archive. A matching checksum verifies that the archive bytes match the expected file.
2. `ZipFile` lets the notebook open JPEGs directly inside the archive. The loops consider `.jpg` files under the two animal folders, in sorted filename order.
3. Pillow opens each photo, converts it to RGB, and resizes it to 32 × 32. `np.asarray(image)` turns the prepared photo into an array. The code appends its label only after a successful load.
4. The `try` block handles image-reading failures. It also treats Pillow `UserWarning`s raised inside the block as errors. Files caught by that block increase `skipped` and contribute neither pixels nor a label.
5. `np.stack(images)` collects the readable photos into one array. At this point the array has one axis for photos, two for image position, and one for color channels. The labels form a separate one-dimensional array in the same order.

### What came out of the loading step

| Count | Value |
| --- | ---: |
| JPEG files found in the archive | 25,000 |
| JPEG files skipped by the loading checks | 3 |
| Readable cat photos | 12,499 |
| Readable dog photos | 12,498 |
| Total readable photos | 24,997 |

The first plot uses `np.bincount(labels)` to count the two classes. It counts the readable photos **before** duplicate removal and splitting. One photo separates the two classes at this stage, so the bars are almost the same height. The counts here come from reading the original archive with Pillow in this notebook. The catalog describes the dataset produced by its own loading pipeline.

## 3. How the photos become numbers

### Give every photo the same shape

Original JPEGs can have different widths, heights, and color formats. A Dense layer needs the same number of inputs for each example. The loading code converts every readable image to RGB, so every pixel has a red, green, and blue value. It then resizes every photo to 32 × 32. That gives each photo 1,024 pixel positions and `32 × 32 × 3 = 3,072` channel values.

Using 32 × 32 keeps the first layer smaller and the training run short. The trade-off is detail. The resize can soften features such as whiskers and small differences in ear shape. It also puts every photo into a square, so it can change the proportions of a photo whose original shape was rectangular.

### Remove repeated images before making the split

The notebook uses `sha256(image.tobytes()).digest()` as a compact key for the **resized RGB pixels**. This is a separate use of SHA-256 from the download checksum. The download checksum checks the whole ZIP. The image key lets the notebook group photos that have the same prepared pixels.

The `seen` dictionary stores the index of the first image for each key. A repeated key with the same label needs only one representative. If a repeated key has a different label, `conflicting_images` records the key. During cleanup, `keep` excludes that entire group because the same input would otherwise have both answers.

The code reports **30 repeated copies** beyond the first representative for each key. This count includes any extra copies within conflicting groups. The code also drops the first representative of each of the **2 conflicting groups**. The count therefore goes from `24,997` readable photos to `24,997 − 30 − 2 = 24,965` usable examples.

Doing this before the split prevents matching prepared pixels from appearing on both sides. If an exact copy gets into both training and test data, the test score can give the model credit for recognizing a photo it already saw. Cropped, recolored, or slightly altered copies can still produce different pixel arrays. The notebook checks exact equality at the resized image level.

### Flatten and scale the pixels

The main conversion is `images[keep].reshape(len(keep), -1).astype("float32") / 255`. Each part does one job.

| Part | What it does |
| --- | --- |
| `images[keep]` | Selects the photos that survived cleanup. |
| `.reshape(len(keep), -1)` | Keeps one row per photo and flattens the color values into 3,072 columns. NumPy calculates the column count from the original shape. |
| `.astype("float32")` | Converts the pixel integers to floating-point values. |
| `/ 255` | Scales the pixel values from `0–255` to `0–1`. |

For example, a pixel with RGB values `[0, 128, 255]` becomes approximately `[0.0, 0.502, 1.0]`. The order of the channel values stays the same. The scale for every example comes from the known pixel range, so the notebook can apply the same fixed constant to every photo without estimating statistics from the data.

Flattening puts the RGB values in a consistent order. Each column still corresponds to a particular position and channel in the resized photo. The Dense model processes these columns as a long row and has no built-in operation that moves the same filter across neighboring pixels.

### Prepare the target labels

`labels = labels[keep]` applies the same selection to the labels, so photo `X[i]` and label `labels[i]` still belong together. `keras.utils.to_categorical(labels, num_classes=2)` then converts the integers to one-hot labels.

| Class | Integer label | One-hot label | Output index |
| --- | ---: | --- | ---: |
| Cat | `0` | `[1, 0]` | `0` |
| Dog | `1` | `[0, 1]` | `1` |

One-hot encoding gives the true class a `1` and the other class a `0`. It matches the two output scores that the model will produce. The notebook keeps the integer labels too. They are convenient for stratifying the splits and comparing the predicted class with the actual class.

The final shapes before splitting are `X.shape = (24965, 3072)` and `y.shape = (24965, 2)`. `np.isfinite(X).all()` checks that every prepared feature is finite. The stored output shows that the scaled values range from `0.0` to `1.0`.

## 4. How the data is split

The notebook builds a list of photo indices with `np.arange(len(y))` and splits that list with `train_test_split()`. It uses the resulting indices on the pixel arrays and both forms of the labels. This keeps each photo matched with its answer through the split.

The split uses `test_size=0.2`, `stratify=labels`, and `random_state=12`. **Stratification** keeps the cat-to-dog ratio close to the ratio in the full cleaned dataset. The random seed makes the same choice of indices reproducible when the input and software behavior match. An assertion checks that the training and test index lists have zero overlap.

| Partition | Photos | Role |
| --- | ---: | --- |
| Training pool | 19,972 | Supplies all the training and validation folds. After choosing the learning rate, the final model trains on this whole pool. |
| Held-out test set | 4,993 | Supplies the final reported results. Its labels play no part in selecting the learning rate or adjusting the weights. |

The term **validation set** refers to a held-out part of the training pool inside a cross-validation run. A different part takes that role in each fold. The **test set** is a separate group of photos that stays fixed for the whole assignment. The model selection code uses validation results, so the test set can serve as a separate check once those decisions have been made.

## 5. How the Dense network works

`build_model()` returns a Keras `Sequential` model. Sequential means the layers run in order. Each layer passes its output to the layer that follows it.

```text
3,072 scaled RGB values
    → Dense layer with 128 units and ReLU
    → Dense layer with 64 units and ReLU
    → Dense layer with 2 units and softmax
    → Cat or Dog, using the larger score
```

### What a Dense unit calculates

A Dense unit uses a weight for every number coming into it. It multiplies each input by its weight, adds the products, and adds one bias. The result before activation is a weighted sum. The activation function then transforms that sum before the network passes it on.

For a toy unit with two inputs, suppose the inputs are `0.8` and `0.2`, the weights are `0.5` and `-0.4`, and the bias is `0.1`. The weighted sum is `0.8 × 0.5 + 0.2 × (-0.4) + 0.1 = 0.42`. A real unit in the first hidden layer does this with 3,072 inputs. Different units have their own sets of weights and can respond to different combinations of pixels.

The hidden layers use **ReLU**, which returns `max(0, z)` for an incoming sum `z`. The toy value `0.42` would pass through as `0.42`, and `-0.42` would become `0`. ReLU adds a nonlinear step. Without a nonlinear activation between the hidden Dense layers, stacking those weighted sums and biases would still produce a single affine transformation.

The final layer uses **softmax**. Softmax converts the two output scores before activation, often called *logits*, into two values that sum to 1. In a hypothetical output of `[0.27, 0.73]`, index `1` is larger, so the prediction is Dog. The first output always corresponds to Cat because the notebook uses the same class order for encoding and predictions.

### Where the 401,730 parameters come from

Each Dense layer needs one weight for every input-to-unit connection plus one bias per unit. For a layer with `m` inputs and `n` units, the total is `m × n + n`.

| Layer | Inputs per unit | Units | Weights | Biases | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| First hidden layer | 3,072 | 128 | 393,216 | 128 | 393,344 |
| Second hidden layer | 128 | 64 | 8,192 | 64 | 8,256 |
| Output layer | 64 | 2 | 128 | 2 | 130 |
| **Total** | | | **401,536** | **194** | **401,730** |

The `keras.Input()` declaration tells Keras to expect 3,072 values per photo. It introduces zero trainable parameters. Most of the weights belong to the first hidden layer because each of its 128 units takes all 3,072 input values. The next layer narrows the representation to 64 values. The final layer needs two units because there are two possible answers.

## 6. How training changes the weights

### Loss and accuracy measure different things

The notebook compiles the model with **categorical cross-entropy** as the loss and **accuracy** as a metric. The loss gives SGD a quantity to reduce. Accuracy counts correct predictions and makes the result easier to read.

For a one-hot label, categorical cross-entropy for one example reduces to `−log(p_true)`, where `p_true` is the softmax value assigned to the actual class. If the photo is a cat and the scores are `[0.8, 0.2]`, the loss is `−log(0.8)`, about `0.223`. The output `[0.2, 0.8]` gives the true cat class only `0.2`, so the loss is about `1.609`.

Two predictions can both get the label right and have different losses. If the actual class is Cat, `[0.51, 0.49]` and `[0.90, 0.10]` both count as correct. The second has a lower cross-entropy because it gives the actual class a larger score. The test loss therefore adds information beyond the percentage of correct labels.

### One batch, one epoch, and the role of SGD

A **batch** is a small group of training photos. The notebook sets the batch size to 256. For each batch, Keras performs this sequence.

1. Pass the batch through the layers to get predictions. This is the **forward pass**.
2. Compare those predictions with the one-hot labels and calculate the batch loss.
3. Work backwards through the layers to calculate how the loss changes when each weight or bias changes. This is **backpropagation**. The resulting values are called **gradients**.
4. Use SGD to update the parameters in a direction intended to reduce the loss. Repeat with the next batch.

An **epoch** is one pass through the training photos for that run. Every fold model trains for 20 epochs, and the final model also trains for 20 epochs.

### Which settings were chosen and which were learned

| Setting | Value in the notebook | What it controls |
| --- | --- | --- |
| Optimizer | SGD | Uses gradients from batches to update the weights and biases. |
| Learning rate | `0.01` or `0.03` during the comparison | Scales the parameter updates. A larger value makes larger steps when the other settings stay the same. |
| Momentum | `0.9` | Carries some of the previous update direction forward. Repeated gradients in the same direction can build momentum, while opposing directions can partly cancel. |
| Weight decay | `0.0005` | Applies a small shrinkage to the parameters during optimization. This is the regularization used by the notebook. |
| Batch size | `256` | Sets how many photos the model handles in a full training batch. |
| Epochs | `20` per fit | Sets how many times each fit passes through its training examples. |

The learning rate, layer sizes, batch size, and other settings in the table are **hyperparameters**. They come from the experiment setup. The values inside the weight matrices and bias vectors are **parameters**. SGD learns those values from the labelled photos. The notebook compares two learning rates and holds the other settings fixed, so it can see how that one change affects validation accuracy under this setup.

## 7. How cross-validation chooses the learning rate

A single validation split gives a score on one group of photos. The notebook uses **five-fold stratified cross-validation** so every photo in the training pool contributes to the validation score once per learning rate.

`StratifiedKFold(n_splits=5, shuffle=True, random_state=12)` forms five almost equal parts with similar class proportions. Each validation fold has 3,994 or 3,995 photos. In each run, the other four folds supply 15,977 or 15,978 training photos.

| Run | Folds used to train | Fold used to validate |
| --- | --- | --- |
| 1 | 2, 3, 4, 5 | 1 |
| 2 | 1, 3, 4, 5 | 2 |
| 3 | 1, 2, 4, 5 | 3 |
| 4 | 1, 2, 3, 5 | 4 |
| 5 | 1, 2, 3, 4 | 5 |

For each learning rate, the notebook repeats all five runs. Before training each fold, it clears Keras's previous model state, sets the random seed to `12 + fold`, and calls `build_model()` again. This starts each fold with a fresh model. The two learning rates use the same split and seed for a given fold, which makes their comparison more consistent.

`model.fit()` receives the four fitting folds as its training data and the remaining fold as `validation_data`. The validation examples produce accuracy and loss measurements after each epoch. Only the fitting examples supply gradients and weight updates in that run.

The `history` dictionary has 20 logged values for each of `accuracy`, `loss`, `val_accuracy`, and `val_loss`. The selection uses `history["val_accuracy"][-1]`, the validation accuracy at **epoch 20**. For each learning rate, the code averages those five end-of-training results.

| Validation fold | Learning rate `0.01` | Learning rate `0.03` |
| --- | ---: | ---: |
| 1 | 65.16% | 62.15% |
| 2 | 64.16% | 64.21% |
| 3 | 62.42% | 63.75% |
| 4 | 65.92% | 63.82% |
| 5 | 64.55% | 64.85% |
| **Mean at epoch 20** | **64.44%** | **63.75%** |

`best_rate = max(scores, key=scores.get)` selects `0.01`. It was ahead by about 0.69 percentage points on the five-fold mean. The fold scores vary, and `0.03` actually did better on folds 2, 3, and 5. The larger advantage for `0.01` on folds 1 and 4 gave it the higher average. The experiment compares these two choices for this architecture and training setup.

After selection, the notebook makes another fresh model, sets seed `12`, and trains it on all **19,972** photos in the training pool with learning rate `0.01`. It uses the full pool now because the learning-rate comparison is over. The test photos stay separate. The final prediction comes from this one refitted model.

The two learning rates and five folds make **10 cross-validation fits**, each of 20 epochs. The final 20-epoch fit makes 11 fits in all. The saved run measured **30.3 seconds** for cross-validation, **3.0 seconds** for the final fit, and **33.3 seconds** from the start of cross-validation to the end of model fitting. The timer starts after data preparation. Downloading and loading the JPEGs happen earlier, and test evaluation happens later. Another machine can give different times.

## 8. What the training curves show

The accuracy and loss plots show the five-fold average for the selected `0.01` learning rate. To draw the value at epoch 5, for example, the notebook takes the epoch-5 value from each of the five histories and averages them. It repeats that for the training and validation metrics at all 20 epochs. The solid lines are training values, and the dashed lines are validation values.

- **Accuracy.** Training accuracy climbs from roughly 56% to 70%. Validation accuracy improves early and spends much of the second half around 63–65%. Its final mean is 64.44%.
- **Loss.** Training loss trends down throughout the run. Validation loss drops early and then moves up and down around a narrower range. It finishes around 0.64.

Around epoch 10 and after, the model keeps improving on the photos used to update its weights, while the validation curve changes much less. That widening gap suggests **some overfitting**. The model is learning patterns that help it classify the fitting photos, and only part of that improvement carries over to the validation photos. The plots record the cross-validation fits, and the test table comes from the separate final refit.

## 9. What the test results mean

### The evaluation code

`model.evaluate(X_test, y_test)` produces the final model's test loss and test accuracy. `model.predict(X_test)` produces two scores per photo. Applying `.argmax(axis=1)` picks the higher score in each row and returns its integer class ID. The notebook checks that the accuracy from `evaluate()` agrees with the fraction of IDs that match `test_labels`.

| Test quantity | Result |
| --- | ---: |
| Number of test photos | 4,993 |
| Correct predictions | 3,338 |
| Incorrect predictions | 1,655 |
| Test accuracy | 66.85% |
| Test categorical cross-entropy | 0.6129 |
| Accuracy from always predicting the majority class | 50.01% |

The test set contains 2,496 cats and 2,497 dogs. A baseline that calls every test photo Dog gets `2,497 / 4,993 = 50.01%`. The model scores about **16.84 percentage points higher**. The baseline is a useful reference because the two classes have nearly the same number of test photos.

### Read the confusion matrix

Rows represent the actual labels and columns represent what the model predicted. The upper-left and lower-right cells are the correct predictions.

| Actual class | Predicted Cat | Predicted Dog | Total actual |
| --- | ---: | ---: | ---: |
| Cat | **1,722** | 774 | 2,496 |
| Dog | 881 | **1,616** | 2,497 |
| **Total predicted** | **2,603** | **2,390** | **4,993** |

You can recover the test accuracy from this table by adding the diagonal and dividing by the total, `(1,722 + 1,616) / 4,993 = 66.85%`. The other cells show how the mistakes break down. The model labelled 881 dogs as cats and 774 cats as dogs. It predicted Cat 2,603 times, more often than the 2,496 cats actually present.

### Precision, recall, and F1-score

The classification report treats each class as the class of interest in turn. For Cat, a true positive is an actual cat predicted as Cat. A false positive is an actual dog predicted as Cat. A false negative is an actual cat predicted as Dog.

- **Cat precision** asks how many Cat predictions were right. It uses the Predicted Cat column as its denominator, so it is `1,722 / 2,603 ≈ 0.662`.
- **Cat recall** asks how many actual cats the model found. It uses the Actual Cat row as its denominator, so it is `1,722 / 2,496 ≈ 0.690`.
- **Cat F1** combines precision and recall using their harmonic mean, `2 × precision × recall / (precision + recall)`. Using the unrounded precision and recall gives about `0.675`.

For Dog, the same calculations use the Predicted Dog column and the Actual Dog row. The model found a smaller share of the actual dogs than of the actual cats, even though its Dog predictions had slightly higher precision.

| Class or average | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| Cat | 0.662 | 0.690 | 0.675 | 2,496 |
| Dog | 0.676 | 0.647 | 0.661 | 2,497 |
| Macro average | 0.669 | 0.669 | 0.668 | 4,993 |
| Weighted average | 0.669 | 0.669 | 0.668 | 4,993 |

**Support** is the number of actual test examples of that class. The **macro average** gives Cat and Dog equal weight. The **weighted average** weights them by their support. The two classes differ by one photo here, so both averages round to the same values. The report displays overall accuracy as `0.669` because it rounds to three decimal places. The notebook's percentage output uses two decimal places and shows `66.85%`.

## 10. What the assignment shows and where it is limited

The Dense network learned enough from the resized RGB pixels to beat the majority-class baseline on the held-out photos. The run also shows the difference between improving on the training examples and improving on unseen photos. By the later epochs, training accuracy continues to rise while validation accuracy stays around 64%. The final held-out evaluation puts the model at 66.85%, with about one in three test photos classified incorrectly.

The test accuracy is higher than the 64.44% mean cross-validation accuracy at the chosen rate. The five validation scores came from five models trained on four folds each. The test score came from a fresh model trained on the whole training pool and scored on a different group of photos. The larger fitting set and the mix of evaluation photos can both affect the difference.

The 32 × 32 input retains coarse color and shape information. Fine details can disappear during the resize. The Dense architecture also learns from fixed pixel positions, so handling changes in pose, framing, or background can be difficult. The test score reflects the combined choices for image size, hidden-layer size, and weight decay. The experiment varies only the learning rate.

## 11. How to run the notebook again

1. Open the `.ipynb` file in Jupyter or Colab and select a Python 3 kernel.
2. If the environment needs the packages, install them with the command below. NumPy is also used by the notebook and comes with the required numerical packages.
3. Run all cells in order. Allow about 787 MiB for the first dataset download. Keras can reuse the cached archive in a later run from the same environment.
4. Check the preprocessing counts first, then the fold results and selected rate, and finally the test loss, accuracy, and confusion matrix. Training durations can change with the hardware. Differences in package versions and numeric execution can also change a rerun's scores.

```bash
python -m pip install tensorflow scikit-learn pillow matplotlib
```

The assignment files save the code and the outputs from the completed run. Running the training cells rebuilds the model weights in the notebook session.
