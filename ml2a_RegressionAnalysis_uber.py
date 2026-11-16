import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import make_pipeline
import numpy as np

df = pd.read_csv("uber.csv")

df = df.dropna()

cols = ['pickup_longitude', 'pickup_latitude',
        'dropoff_longitude', 'dropoff_latitude',
        'passenger_count', 'fare_amount']

df = df[cols]

df = df[(df['fare_amount'] > 0) & (df['fare_amount'] < 100)]
df = df[(df['passenger_count'] > 0) & (df['passenger_count'] <= 6)]

df = df[
    (df['pickup_longitude'].between(-180, 180)) &
    (df['dropoff_longitude'].between(-180, 180)) &
    (df['pickup_latitude'].between(-90, 90)) &
    (df['dropoff_latitude'].between(-90, 90))
]

sns.heatmap(df.corr(), annot=True)
plt.show()

X = df.drop('fare_amount', axis=1)
y = df['fare_amount']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Linear": LinearRegression(),
    "Ridge": make_pipeline(StandardScaler(), Ridge()),
    "Lasso": make_pipeline(StandardScaler(), Lasso())
}

for name, model in models.items():
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print(name)
    print("R2 Score:", r2)
    print("RMSE:", rmse)
    print()

# ==========================================
# CODE EXPLANATION
# ==========================================
# - Imports pandas, numpy, matplotlib, seaborn, and scikit-learn tools for preprocessing, pipelines, regression models, and evaluation metrics.
# - Loads `uber.csv`, removes rows with missing values (`dropna()`), and keeps only pickup/dropoff coordinates, passenger count, and fare amount.
# - Filters out unrealistic outliers: fares must be between $0 and $100, passengers between 1 and 6, longitudes within [-180, 180], and latitudes within [-90, 90].
# - Plots a correlation heatmap (`sns.heatmap(df.corr(), annot=True)`) to inspect linear relationships between variables.
# - Splits features (`X`) and target (`y = fare_amount`) into 80% training and 20% testing sets.
# - Defines three regression models in a dictionary: standard Linear Regression, and pipelines pairing `StandardScaler()` with Ridge and Lasso regression.
# - Loops through each model to train (`fit`), predict on `X_test`, and print `R2 Score` and `RMSE` (Root Mean Squared Error).

# ==========================================
# VIVA QUESTIONS & ANSWERS
# ==========================================
# Q1: What is the difference between Linear, Ridge, and Lasso Regression?
# A1: Linear Regression minimizes ordinary least squares without penalty; Ridge adds an L2 penalty (squared weights) to shrink coefficients; Lasso adds an L1 penalty (absolute weights) which can shrink unimportant feature coefficients all the way to zero.
#
# Q2: Why are Ridge and Lasso wrapped inside `make_pipeline(StandardScaler(), ...)`?
# A2: Regularization penalties depend on the scale of features—`StandardScaler` normalizes all features to mean 0 and variance 1 so penalties apply fairly, while `make_pipeline` prevents data leakage between train and test splits.
#
# Q3: Why were longitude and latitude filtered to `[-180, 180]` and `[-90, 90]`?
# A3: Valid Earth coordinates strictly range from -180° to +180° for longitude and -90° to +90° for latitude; values outside these bounds are corrupted GPS readings.
#
# Q4: What does `annot=True` do in `sns.heatmap()`?
# A4: It displays the exact numerical correlation coefficient inside each cell of the heatmap.
#
# Q5: What is the R² (R-squared) score?
# A5: Also called the coefficient of determination, it measures the proportion of variance in the target variable (`fare_amount`) explained by the model (1.0 is perfect, 0 means no better than predicting the mean).
#
# Q6: What is RMSE, and why take the square root of MSE?
# A6: RMSE (Root Mean Squared Error) measures average prediction error; taking the square root brings the error back into the original target unit (dollars) instead of squared dollars.
#
# Q7: Why might raw coordinates give a low R² score in linear models?
# A7: Fare depends on travel distance rather than raw latitude/longitude numbers; computing Haversine or Euclidean distance between pickup and dropoff significantly improves accuracy.