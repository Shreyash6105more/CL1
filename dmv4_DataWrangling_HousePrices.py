import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("Bengaluru_House_Data.csv")

print(df.head())
print(df.shape)
print(df.isnull().sum())

df = df.drop(
    ["area_type", "society", "balcony", "availability"],
    axis=1
)

df = df.dropna()

df["bhk"] = df["size"].apply(
    lambda x: int(x.split(" ")[0])
)

def convert_sqft(x):
    try:
        if "-" in x:
            a, b = x.split("-")
            return (float(a) + float(b)) / 2
        return float(x)
    except:
        return None

df["total_sqft"] = df["total_sqft"].apply(convert_sqft)
df = df.dropna()

df["price_per_sqft"] = df["price"] * 100000 / df["total_sqft"]

avg_price = df.groupby("location")["price"].mean()

print("-------- Avg Price --------")
print(avg_price.head())

plt.boxplot(df["price"])
plt.title("House Price Outliers")
plt.show()

df.to_csv("Cleaned_RealEstate.csv", index=False)

print("Data Wrangling Completed!")

# ==========================================
# CODE EXPLANATION
# ==========================================
# - Imports pandas, numpy, and matplotlib for data wrangling and visualization.
# - Loads the Bengaluru housing dataset and inspects its first 5 rows, dimensions (shape), and null counts.
# - Drops non-essential columns ("area_type", "society", "balcony", "availability") along axis=1 (columns).
# - dropna() removes rows containing missing values.
# - Extracts the integer bedroom count from strings like "2 BHK" or "4 Bedroom" using split() and a lambda function, storing it in a new 'bhk' column.
# - convert_sqft() parses 'total_sqft': if it's a range (e.g., "1200 - 1500"), it takes the average; if single number, converts to float; if invalid text (e.g., "34Sq. Meter"), returns None.
# - Drops rows where convert_sqft() returned None.
# - Creates 'price_per_sqft' (feature engineering) by converting 'price' from Lakhs to Rupees (* 100,000) and dividing by 'total_sqft'.
# - Computes average house price per location using groupby().
# - Plots a boxplot of 'price' to visually detect extreme outliers.
# - Exports the cleaned DataFrame to "Cleaned_RealEstate.csv" without row index numbers.

# ==========================================
# VIVA QUESTIONS & ANSWERS
# ==========================================
# Q1: What is Data Wrangling (Data Munging)?
# A1: The process of cleaning, structuring, and enriching raw data into a desired format for better decision-making and analysis.
#
# Q2: What does `axis=1` specify in `df.drop()`?
# A2: It tells Pandas to drop columns rather than rows (`axis=0` refers to rows).
#
# Q3: How does `x.split(" ")[0]` extract the BHK value?
# A3: It splits a string like `"2 BHK"` at the space into `["2", "BHK"]`, selects the first element (`[0]`), and `int()` converts `"2"` into integer `2`.
#
# Q4: Why do we use a `try-except` block inside `convert_sqft()`?
# A4: Some entries contain non-numeric units (like `"34Sq. Meter"` or `"4125Perch"`). `try-except` prevents ValueError crashes by safely returning `None`.
#
# Q5: What is Feature Engineering in this code?
# A5: Creating new, more informative columns (`bhk` and `price_per_sqft`) from existing raw columns (`size`, `price`, `total_sqft`).
#
# Q6: Why is `price` multiplied by `100000`?
# A6: In the dataset, house prices are recorded in Lakhs (1 Lakh = 100,000 Rupees), so multiplying converts them to standard Rupees per square foot.
#
# Q7: How do you identify outliers in a Boxplot?
# A7: Data points plotted as individual dots/circles beyond the upper or lower whiskers (outside 1.5 × Interquartile Range) represent outliers.
#
# Q8: Why is `index=False` used in `to_csv()`?
# A8: It prevents Pandas from writing the DataFrame's row index numbers as an extra unnamed column in the saved CSV file.