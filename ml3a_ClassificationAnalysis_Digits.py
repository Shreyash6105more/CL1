import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_digits
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report

digits = load_digits(n_class=10)

X = digits.data
y = digits.target

plt.imshow(digits.images[0], cmap='gray')
plt.title("Digit: " + str(y[0]))
plt.show()

X_train = X[:1500]
y_train = y[:1500]

X_test = X[1501:]
y_test = y[1501:]

model = svm.SVC(gamma=0.001, C=100)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Predicted:", predictions)
print("Actual:", y_test)

cm = confusion_matrix(y_test, predictions)

sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print(classification_report(y_test, predictions))

# ==========================================
# CODE EXPLANATION
# ==========================================
# - Imports numpy, matplotlib, seaborn, and scikit-learn's `load_digits`, `svm`, and evaluation metrics.
# - `load_digits(n_class=10)` loads 1,797 grayscale images (8x8 pixels) of handwritten digits (0–9).
# - `digits.data` holds flattened 64-feature arrays (`X`), while `digits.target` holds the true digit labels (`y`).
# - `plt.imshow(digits.images[0], cmap='gray')` renders the first 8x8 image matrix in grayscale.
# - Manually slices the dataset into training (first 1,500 samples) and testing (index 1,501 onward) sets.
# - Initializes a Support Vector Classifier (`svm.SVC`) with `gamma=0.001` and `C=100`, then trains it via `.fit()`.
# - Predicts labels on `X_test` and prints predicted vs. actual values.
# - Builds a confusion matrix (`confusion_matrix`) and plots it as an integer-annotated heatmap (`fmt='d'`).
# - Prints `classification_report` showing Precision, Recall, F1-score, and Support for every digit class (0–9).

# ==========================================
# VIVA QUESTIONS & ANSWERS
# ==========================================
# Q1: What is the shape of each image in `digits.images` vs. `digits.data`?
# A1: `digits.images` stores each sample as a 2D `8x8` pixel matrix (for visualization), whereas `digits.data` flattens it into a 1D vector of `64` features (for model training).
#
# Q2: What is SVM (Support Vector Machine)?
# A2: A supervised machine learning algorithm that finds the optimal hyperplane maximizing the margin between different classes in high-dimensional space.
#
# Q3: What do the hyperparameters `C` and `gamma` control in `svm.SVC`?
# A3: `C` controls the trade-off between a smooth decision boundary and classifying training points correctly (high `C` = stricter, fewer misclassifications). `gamma` defines how far the influence of a single training example reaches in the RBF kernel.
#
# Q4: What is the default kernel used by `svm.SVC`?
# A4: The Radial Basis Function (`rbf`) kernel, which maps data into higher dimensions to handle non-linear decision boundaries.
#
# Q5: What does `fmt='d'` do in `sns.heatmap()`?
# A5: It formats the cell annotations as standard decimal integers instead of scientific notation (e.g., `25` instead of `2.5e+01`).
#
# Q6: How do you read a Confusion Matrix?
# A6: Diagonal cells represent correct predictions (True Positives for each class), while off-diagonal cells represent misclassifications between actual and predicted digits.
#
# Q7: What do Precision, Recall, and F1-Score mean in `classification_report`?
# A7: Precision is how many predicted positives were actually correct; Recall is how many actual positives the model caught; F1-Score is the harmonic mean balancing both.