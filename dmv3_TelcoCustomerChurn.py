import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

data = pd.read_csv("Telecom_Customer_Churn.csv")

print(data.head())
print(data.shape)
print(data.info())
print(data.isnull().sum())

data["TotalCharges"] = pd.to_numeric(
    data["TotalCharges"], errors="coerce"
)

data["TotalCharges"] = data["TotalCharges"].fillna(
    data["TotalCharges"].median()
)

data = data.drop_duplicates()

print(data.describe())

plt.boxplot(data["tenure"])
plt.title("Tenure Outliers")
plt.show()

plt.boxplot(data["MonthlyCharges"])
plt.title("Monthly Charges Outliers")
plt.show()

sns.pairplot(data[["tenure", "MonthlyCharges", "TotalCharges"]])
plt.show()

X = data.drop("Churn", axis=1)
y = data["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

data.to_csv("Cleaned_Telecom_Customer_Churn.csv", index=False)

print("Data cleaning completed!")

# ==========================================
# CODE EXPLANATION
# ==========================================
# - Imports pandas for data manipulation, matplotlib & seaborn for plotting, and train_test_split from scikit-learn.
# - Loads the telecom churn dataset and prints sample rows, dimensions (shape), data types (info), and null counts.
# - pd.to_numeric(..., errors="coerce") converts 'TotalCharges' from string/object to float, turning blank/invalid strings into NaN.
# - fillna(median()) imputes those missing 'TotalCharges' values using the column's median.
# - drop_duplicates() removes exact duplicate rows to avoid bias.
# - describe() outputs summary statistics (count, mean, std, min, quartiles, max) for numerical columns.
# - Boxplots check for extreme outliers in 'tenure' and 'MonthlyCharges'.
# - sns.pairplot() creates a grid of scatter plots and histograms showing pairwise relationships among 'tenure', 'MonthlyCharges', and 'TotalCharges'.
# - Separates features (X) from the target variable ('Churn', y) and splits the dataset into 80% training and 20% testing sets.
# - Saves the cleaned DataFrame to "Cleaned_Telecom_Customer_Churn.csv" without row indices.

# ==========================================
# VIVA QUESTIONS & ANSWERS
# ==========================================
# Q1: Why was `TotalCharges` converted using `pd.to_numeric(..., errors="coerce")`?
# A1: In raw telecom datasets, new customers with 0 tenure often have blank spaces `" "` in `TotalCharges`, making Pandas read the whole column as text (`object`). `errors="coerce"` turns blanks into `NaN` so it becomes numeric.
#
# Q2: Why use `median()` instead of `mean()` to fill missing values?
# A2: Median is robust against extreme outliers and skewed distributions, whereas the mean can be heavily distorted by unusually high charges.
#
# Q3: What does `sns.pairplot()` visualize?
# A3: It plots pairwise scatter plots between every pair of selected variables to show correlations, and histograms/KDE plots along the diagonal to show individual distributions.
#
# Q4: What is the purpose of `train_test_split()`?
# A4: It divides the dataset into a training set (to train the machine learning model) and an unseen testing set (to evaluate how well the model generalizes).
#
# Q5: What does `test_size=0.2` mean?
# A5: 20% of the data is reserved for testing, and the remaining 80% is used for training.
#
# Q6: Why do we set `random_state=42`?
# A6: It fixes the random seed so the data splits in the exact same way every time the code runs, ensuring reproducible results.
#
# Q7: In `X = data.drop("Churn", axis=1)`, what are `X` and `y`?
# A7: `X` represents the independent input features (predictors), and `y` represents the dependent target variable (`Churn`) we want to predict.
#
# Q8: Why should duplicate records be removed before splitting the data?
# A8: If duplicates exist, identical rows could end up in both `X_train` and `X_test`, causing data leakage and artificially inflated model accuracy.