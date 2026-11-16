import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

cols = ['buying', 'maint', 'doors', 'persons',
        'lug_boot', 'safety', 'class']

data = pd.read_csv("car.csv", names=cols)

le = LabelEncoder()

for col in data.columns:
    data[col] = le.fit_transform(data[col])

X = data.drop('class', axis=1)
y = data['class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

print("Feature Importance:")
print(model.feature_importances_)

# ==========================================
# CODE EXPLANATION
# ==========================================
# - Imports pandas, matplotlib, seaborn, and scikit-learn tools for label encoding, data splitting, Random Forest classification, and evaluation metrics.
# - Loads `car.csv` and assigns custom column names (`buying`, `maint`, `doors`, `persons`, `lug_boot`, `safety`, `class`) since the raw file lacks headers.
# - Loops through all columns and applies `LabelEncoder()` to convert text categories (e.g., 'low', 'med', 'high') into integers (0, 1, 2...).
# - Splits features (`X`) from the target label (`y = class`) into 80% training and 20% testing sets.
# - Initializes `RandomForestClassifier` with 100 decision trees (`n_estimators=100`) and trains it on `X_train`, `y_train`.
# - Predicts car acceptability classes on `X_test` and prints overall accuracy alongside precision, recall, and F1-scores.
# - Computes and plots the confusion matrix as an integer-annotated heatmap (`fmt='d'`).
# - Prints `model.feature_importances_` showing the relative contribution of each feature to the predictions.

# ==========================================
# VIVA QUESTIONS & ANSWERS
# ==========================================
# Q1: What is a Random Forest Classifier?
# A1: An ensemble supervised learning algorithm that builds multiple Decision Trees on random subsets of data and features, combining their predictions via majority voting to improve accuracy and prevent overfitting.
#
# Q2: What does `n_estimators=100` mean?
# A2: It specifies that the Random Forest will construct 100 individual Decision Trees in the ensemble.
#
# Q3: Why is `names=cols` passed inside `pd.read_csv()`?
# A3: The raw `car.csv` file has no header row; `names=cols` prevents the first data record from being treated as column headers and assigns meaningful names.
#
# Q4: What does `LabelEncoder` do, and why is it needed?
# A4: It converts categorical text labels into numeric integers (e.g., `0, 1, 2`), because scikit-learn models require numerical inputs.
#
# Q5: What is Bagging (Bootstrap Aggregating) in Random Forest?
# A5: Each tree in the forest is trained on a random sample of rows drawn with replacement (bootstrapping), reducing model variance.
#
# Q6: What does `model.feature_importances_` tell us?
# A6: It gives a normalized score (summing to 1.0) indicating how strongly each feature (such as `safety` or `persons`) reduced impurity across all trees in the forest.
#
# Q7: Why is Random Forest preferred over a single Decision Tree?
# A7: A single Decision Tree easily overfits training data and is sensitive to small variations, whereas averaging 100 trees yields higher generalization accuracy and stability.