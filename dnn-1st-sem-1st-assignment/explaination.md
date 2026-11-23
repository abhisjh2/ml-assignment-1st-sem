# How I Did the Cats vs Dogs Assignment

For this assignment I used a Deep Neural Network (DNN) to sort photos into two groups, cats and dogs. I took photos that already had labels and used most of them to train the model. I kept the rest aside to see how the model would do on new photos.

I got **3,338 out of 4,993 test photos right**, which is **66.85% accuracy**. I used Keras for the model and Python for the rest of the work. The code, plots, and outputs are in the notebook. [Jupyter notebook](DL_assignment_Student2026CS05046.ipynb) · [PDF version](DL_assignment_student2026CS05046.pdf)

## Contents

1. [What I wanted the model to do](#1-what-i-wanted-the-model-to-do)
2. [Getting the photos and labels](#2-getting-the-photos-and-labels)
3. [Getting the photos ready for the model](#3-getting-the-photos-ready-for-the-model)
4. [Splitting the data](#4-splitting-the-data)
5. [Building the network](#5-building-the-network)
6. [Training the model](#6-training-the-model)
7. [Trying two learning rates](#7-trying-two-learning-rates)
8. [Reading the training plots](#8-reading-the-training-plots)
9. [Checking the test results](#9-checking-the-test-results)
10. [What I noticed](#10-what-i-noticed)
11. [Running the notebook again](#11-running-the-notebook-again)

## 1. What I wanted the model to do

My input is one photo. The model has to give one answer for the whole photo, Cat or Dog. Since there are two possible answers, this is called **binary classification**. It is also an example of **supervised learning** because the photos used for training come with the correct answers.

During training, the model makes a guess and compares it with the answer. It uses the difference to adjust its weights. Later, when I test it, I give it only the pixels to make a guess. I use the labels after that to check how many guesses were right.

I used Dense layers as required in the assignment. A Dense layer expects a row of numbers, so the first part of the notebook is about turning photos into rows the model can use.

## 2. Getting the photos and labels

I used Microsoft's PetImages dataset. It comes as a ZIP file with the cat and dog photos in separate folders. I used the folder name as the answer for each photo. In my code, `Cat` is `0` and `Dog` is `1`. [Microsoft dataset download](https://www.microsoft.com/en-us/download/details.aspx?id=54765) · [TensorFlow Datasets catalog](https://www.tensorflow.org/datasets/catalog/cats_vs_dogs)

The loading part of the notebook works like this.

1. `keras.utils.get_file()` gets the ZIP file. If the file is already in the cache, it can use that copy. I pass in a checksum so Keras can check that the ZIP matches the expected file.
2. `ZipFile` lets me read the `.jpg` files straight from the ZIP. The code goes through `PetImages/Cat/` first and `PetImages/Dog/` second.
3. Pillow opens each photo and prepares it as a small RGB image. `np.asarray()` converts the image into an array of numbers. I add the image and its label in the same order, so they stay matched.
4. If reading a photo raises one of the errors or warnings handled in the code, I skip that file and add one to `skipped`. After loading, `np.stack(images)` puts the readable images into one array.

| What I counted | Photos |
| --- | ---: |
| JPEGs in the ZIP | 25,000 |
| Files skipped | 3 |
| Readable cats | 12,499 |
| Readable dogs | 12,498 |
| Total readable photos | 24,997 |

I used `np.bincount(labels)` to make the first plot. It counts the readable photos before I remove duplicates. The two bars are almost the same height because there is only a one-photo difference. These are the counts from opening the ZIP with Pillow in this run.

## 3. Getting the photos ready for the model

### Resizing and converting to RGB

The photos in the ZIP come in different sizes. I resized them all to **32 × 32** so they would give the model the same number of inputs. I also converted them to RGB so each pixel would have a red, green, and blue value.

A photo now has `32 × 32 = 1,024` pixels. With three color values per pixel, that gives `1,024 × 3 = 3,072` numbers. Using small images keeps the model and training time down. It also means some details, like fine whiskers, can be harder to see in the resized photos.

### Removing copies

I checked for duplicate photos after resizing and before the train and test split. The check compares the prepared pixels. The code gives each image a key with `sha256(image.tobytes()).digest()`. Photos with the same prepared pixels get the same key.

- `seen` remembers the first image for each key. If I find a copy with the same label, one is enough.
- `conflicting_images` tracks keys that show up with both labels. I drop those groups because the same input would have two answers.
- `keep` stores the indices of the images that stay. I use it on the photos and labels together.

The code counted **30 repeated copies** beyond the first image in each group. It also dropped the first image from each of the **2 groups** with conflicting labels. That left `24,997 − 30 − 2 = 24,965` photos.

I did this before splitting so that an exact copy of a training photo would stay out of the test set. If someone crops a photo or changes its colors, the resized numbers may be different and the copy can get through this check.

### Making one row per photo

The line below does the flattening and scaling.

```python
X = images[keep].reshape(len(keep), -1).astype("float32") / 255
```

`images[keep]` picks the photos I want to use. `reshape` puts each photo into one long row, and `-1` lets NumPy work out how many columns are needed. I convert the whole numbers to decimals and divide by `255`. For example, a pixel with values `[0, 128, 255]` becomes about `[0.0, 0.502, 1.0]`. Every value is now between `0` and `1`. The notebook checks that the finished input has no `NaN` or infinite values.

### Matching the labels to the output

The network will return two scores. I used one-hot encoding so the correct answer also has two places. Keras does this with `to_categorical()`.

| Actual class | Original label | Label used for training |
| --- | ---: | --- |
| Cat | `0` | `[1, 0]` |
| Dog | `1` | `[0, 1]` |

`X` has shape `(24965, 3072)` because it has one row per photo and 3,072 pixel values in each row. `y` has shape `(24965, 2)` because each photo has a two-number answer. I keep the integer labels too, since they are useful when splitting the data and reading the test results.

## 4. Splitting the data

I used `train_test_split()` with `test_size=0.2` to keep 20% aside. I split the photo indices and use the same indices to select the pixels and labels. This makes sure each photo stays with the right answer.

| Part | Number of photos | What I used it for |
| --- | ---: | --- |
| Training pool | 19,972 | Comparing the learning rates and training the final model. |
| Test set | 4,993 | Checking the final model after training. |

I used `stratify=labels` to keep about the same proportion of cats and dogs in both parts. `random_state=12` sets the seed for the split, and the notebook checks that the two index lists have no photo in common.

There is also a **validation** part during training. It comes from the 19,972 photos in the training pool. I use it when deciding which learning rate to pick. After I have made that choice and trained the final model, I use the separate test set to get the final result.

## 5. Building the network

I used a Keras `Sequential` model, which runs the layers one after the other. This is what happens to one row of input.

```text
3,072 pixel values
    → 128 Dense units with ReLU
    → 64 Dense units with ReLU
    → 2 Dense units with softmax
    → Cat or Dog
```

For one Dense unit, I can think of it as multiplying every input by a weight, adding those numbers, and adding one extra value called the **bias**. A small example with just two inputs would be `0.8 × 0.5 + 0.2 × (-0.4) + 0.1 = 0.42`. The first layer does this with 3,072 inputs for each unit. Each unit has its own weights.

I used **ReLU** in the hidden layers. It keeps a positive number as it is and changes a negative number to zero. So `0.42` stays `0.42`, while `-0.42` becomes `0`. This helps the model learn more than a single weighted sum of the pixels.

For the last layer I used **softmax**. It gives me two scores that add up to `1`. If I get `[0.27, 0.73]`, the second score is higher, so I pick Dog.

### Counting the weights and biases

I counted one weight for every input to each unit and one bias for each unit. For the first layer, that is `3,072 × 128 + 128 = 393,344`.

| Layer | Number of weights | Number of biases | Total |
| --- | ---: | ---: | ---: |
| First hidden layer | 393,216 | 128 | 393,344 |
| Second hidden layer | 8,192 | 64 | 8,256 |
| Output layer | 128 | 2 | 130 |
| **All three layers** | **401,536** | **194** | **401,730** |

Keras prints the same total in the notebook. Most of it comes from the first layer because all 3,072 inputs are connected to each of its 128 units.

## 6. Training the model

I used **categorical cross-entropy** for the loss. In this assignment it looks at the score given to the correct class. If the photo is a cat and the model gives Cat a score of `0.8`, the loss is `−log(0.8)`, about `0.223`. If it gives Cat only `0.2`, the loss is about `1.609`. The model tries to lower this loss during training. Accuracy is the fraction of labels it guessed correctly.

I grouped the photos into batches of 256. For each batch, the model predicts the labels, calculates the loss, and works out how changing each weight would affect the loss. That step is called **backpropagation**, and the values it calculates are called **gradients**. I used stochastic gradient descent (SGD) to update the weights using those gradients. Going through all the training photos once is **one epoch**. I used **20 epochs** in each run.

| Setting | Value | How I understand it |
| --- | --- | --- |
| Optimizer | SGD | Makes the weight updates using a batch at a time. |
| Learning rate | `0.01` or `0.03` | Controls the size of the update. |
| Momentum | `0.9` | Uses some of the previous update when making the next one. |
| Weight decay | `0.0005` | Shrinks the weights a little during training to discourage very large values. |
| Batch size | `256` | Number of photos in a full batch. |
| Epochs | `20` | Number of passes through the training photos in each run. |

The settings in this table are values I gave to the model. The weights and biases are values it learns while fitting the data.

## 7. Trying two learning rates

I tried `0.01` and `0.03`. I used **five-fold cross-validation** to compare them. I split the training pool into five smaller parts, each with about the same mix of cats and dogs. For the first run I trained on parts 2 to 5 and used part 1 to check the model. On the next run I used part 2 for the check, and so on. Each part had one turn as the validation set. Each validation part had 3,994 or 3,995 photos.

I started a fresh model each time. For a given fold, both learning rates used the same split and random seed. This let me compare the rates with the other settings kept the same. During each run only the four training parts were used to update the weights. The validation part gave me another accuracy and loss reading after each epoch.

I saved the accuracy and loss values from all 20 epochs. To pick the rate, I used the **validation accuracy at epoch 20** for each of the five folds and took the average.

| Validation fold | Rate `0.01` | Rate `0.03` |
| --- | ---: | ---: |
| 1 | 65.16% | 62.15% |
| 2 | 64.16% | 64.21% |
| 3 | 62.42% | 63.75% |
| 4 | 65.92% | 63.82% |
| 5 | 64.55% | 64.85% |
| **Average** | **64.44%** | **63.75%** |

I picked **`0.01`** because its average was higher. The difference was small, and `0.03` did better on three individual folds. With `0.01` chosen, I started a fresh model again and trained it for 20 epochs on **all 19,972 photos in the training pool**. I used this last model for the test.

In the saved run, the ten cross-validation fits took **30.3 seconds** in total, and the final fit took **3.0 seconds**. The notebook reports **33.3 seconds** from starting cross-validation to finishing the final fit. This timer starts after the photos have been loaded and prepared.

## 8. Reading the training plots

The plots show the training and validation history for the chosen learning rate. For each epoch, I took the average of the five fold results. The solid line shows training and the dashed line shows validation.

- In the accuracy plot, the training line goes from about **56% to 70%**. The validation line improves at first, then stays around **64%** for much of the second half.
- In the loss plot, training loss keeps going down. Validation loss drops early and then goes up and down a little. It ends at about **0.64**.

After about epoch 10, the model keeps getting better at the photos it is learning from, while the validation results move much less. I took that as a sign of some **overfitting**. The validation photos help show how much of the improvement carries over to photos the model has left out of training.

## 9. Checking the test results

I ran `model.evaluate()` on the 4,993 test photos to get accuracy and loss. I also used `model.predict()` to get a pair of scores for every photo. The code takes the position of the larger score using `argmax`. The notebook checks that comparing those predicted labels with the actual labels gives the same accuracy as `evaluate()`.

| Test result | Value |
| --- | ---: |
| Correct predictions | 3,338 |
| Wrong predictions | 1,655 |
| Test accuracy | 66.85% |
| Test loss | 0.6129 |

I also checked what would happen if I guessed Dog for every photo, because Dog was the slightly bigger group in the test set. That would get `2,497 / 4,993 = 50.01%`. This gave me a simple number to compare with the model's **66.85%**.

### Where the model made mistakes

In the confusion matrix, the rows tell me what the photo actually was. The columns tell me what the model guessed.

| Actual photo | Guessed Cat | Guessed Dog | Total |
| --- | ---: | ---: | ---: |
| Cat | **1,722** | 774 | 2,496 |
| Dog | 881 | **1,616** | 2,497 |
| **Total** | **2,603** | **2,390** | **4,993** |

The two bold numbers in the Cat and Dog rows are correct predictions. Adding them gives `1,722 + 1,616 = 3,338`. Dividing by `4,993` gives the test accuracy. I can also see that more dogs were called cats (881) than cats were called dogs (774).

### Precision, recall, and F1-score

I read these scores one class at a time. For Cat, precision asks, “Of the times the model guessed Cat, how often was it right?” Recall asks, “Of the actual cats, how many did the model find?”

- Cat precision is `1,722 / 2,603`, about **0.662**.
- Cat recall is `1,722 / 2,496`, about **0.690**.
- Cat F1 combines the two. The formula is `2 × precision × recall / (precision + recall)`, which gives about **0.675** using the unrounded numbers.

| Class or average | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| Cat | 0.662 | 0.690 | 0.675 | 2,496 |
| Dog | 0.676 | 0.647 | 0.661 | 2,497 |
| Macro average | 0.669 | 0.669 | 0.668 | 4,993 |
| Weighted average | 0.669 | 0.669 | 0.668 | 4,993 |

Support is just the number of actual photos of that class. The macro average counts Cat and Dog equally. The weighted average gives the class with more photos more weight. The two classes are almost the same size here, so the averages round to the same values. From the recall column, I can see that the model found about **69% of the cats** and **65% of the dogs**.

## 10. What I noticed

The model got about two out of three photos right. That was better than guessing the biggest class for every photo. It still made 1,655 mistakes. The training plot also shows why I kept a test set. A model can improve on the photos it is learning from while making very little progress on photos it has left out.

The test accuracy was a little higher than the average validation accuracy. The final model trained on the whole training pool, and I checked it on a different group of photos. Each fold model trained on four of the five parts, so there were fewer photos to learn from in those runs.

The small images made it possible to fit the model with a manageable number of inputs. At that size, some details that help tell a cat from a dog can disappear. The Dense layers also work with fixed pixel positions, so images taken from different angles or with different backgrounds can be difficult. I only compared the learning rates in this assignment, so the results tell me how this particular setup worked.

## 11. Running the notebook again

Open the Jupyter notebook in Jupyter or Colab and run the cells in order. For a local environment, install the packages first.

```bash
python -m pip install tensorflow scikit-learn pillow matplotlib
```

The notebook downloads the dataset the first time if Keras has no copy in its cache. The ZIP is about **787 MiB**. After the training cells finish, the notebook prints the fold results and test scores and shows the plots. The time can be different on another computer.
