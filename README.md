# Gradient Descent on CelebA: Smiling Classifier

Course project for COSC 3P96 (Machine Learning), Brock University.
Linear and logistic regression written from scratch in NumPy, trained with gradient descent to predict the "Smiling" attribute from face images.

## Setup
- Data: CelebA, 32x32 grayscale, flattened to 1024 features
- Balanced split: 7000 train, 1500 validation, 1500 test
- 200 epochs, learning rates tested for logistic regression: 0.001, 0.01, 0.05, 0.1
- The dataset is not included. Download it from the official CelebA page and place `img_align_celeba/`, `list_attr_celeba.txt` and `list_eval_partition.txt` next to the notebook.

## Results (test set)
| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Linear regression | 0.507 | 0.509 | 0.403 | 0.450 |
| Logistic regression (lr = 0.01) | 0.680 | 0.658 | 0.749 | 0.701 |

## What I learned
- Linear regression with MSE diverged to NaN at lr = 0.01 on unnormalized pixels. A smaller rate (0.0001) fixed it, but accuracy stayed near chance.
- Logistic regression at lr 0.05 and 0.1 became unstable and mostly predicted "not smiling" (recall around 0.11 to 0.12).
- The loss only fell from 0.693 to 0.638 over 200 epochs, so the model underfits. Next step: standardize the features and try a small neural network.

## Files
- `gradient_descent_celeba.ipynb`: the full notebook with outputs
