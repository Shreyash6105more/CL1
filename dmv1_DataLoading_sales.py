import matplotlib.pyplot as plt
import pandas as pd

pd.DataFrame({'ID': [1, 2], 'Category': ['Electronics', 'Clothing'], 'Qty': [2, 3], 'Price': [500, 100]}).to_csv('sales.csv', index=False)
pd.DataFrame({'ID': [3, 4], 'Category': ['Electronics', 'Furniture'], 'Qty': [1, None], 'Price': [700, 300]}).to_excel('sales.xlsx', index=False)
pd.DataFrame({'ID': [5, 5], 'Category': ['Clothing', 'Clothing'], 'Qty': [4, 4], 'Price': [150, 150]}).to_json('sales.json', orient='records')

df_csv = pd.read_csv('sales.csv')
df_excel = pd.read_excel('sales.xlsx')
df_json = pd.read_json('sales.json')

df = pd.concat([df_csv, df_excel, df_json], ignore_index=True)

print(df.info())
print("Missing values:\n", df.isnull().sum())

df.fillna(df.mean(numeric_only=True), inplace=True)
df.drop_duplicates(inplace=True)

df['Total_Sales'] = df['Qty'] * df['Price']
print("\nUnified & Cleaned Data:\n", df)

print("\nSummary Stats:\n", df.describe())
cat_sales = df.groupby('Category')['Total_Sales'].sum()
print("\nTotal Sales by Category:\n", cat_sales)

cat_sales.plot(kind='bar', title='Total Sales by Category', ylabel='Sales')
plt.show()

cat_sales.plot(kind='pie', autopct='%1.1f%%', title='Sales Share by Category')
plt.show()

# ==========================================
# CODE EXPLANATION
# ==========================================
# - Imports pandas for data handling/aggregation and matplotlib.pyplot for charting.
# - pd.read_csv("sales.csv") loads the sales records into a DataFrame named 'df'.
# - df.head() previews the first 5 rows, and df.info() shows column names, non-null counts, and data types.
# - df.dropna() removes any rows containing missing (NaN) values.
# - df.drop_duplicates() deletes identical duplicate rows to prevent double-counting sales.
# - df["Sales_Amount"].sum() computes the overall total revenue across all valid records.
# - groupby("Product_Category")["Sales_Amount"].sum() calculates total sales separately for each product category.
# - category_sales.plot(kind="bar") plots a bar chart comparing total sales across categories with axis labels and a title.

# ==========================================
# VIVA QUESTIONS & ANSWERS
# ==========================================
# Q1: What is the difference between `df.dropna()` and `df.fillna()`?
# A1: `dropna()` deletes rows/columns with missing values, whereas `fillna()` replaces missing values with a specified value (like 0, mean, or median).
#
# Q2: Why is `drop_duplicates()` important before calculating total sales?
# A2: Duplicate rows would count the same transaction multiple times, inflating the total and category sales figures.
#
# Q3: What does `df.info()` display?
# A3: It provides a concise summary of the DataFrame: index range, column names, count of non-null values per column, data types (`dtypes`), and memory usage.
#
# Q4: What type of object does `df.groupby("Product_Category")["Sales_Amount"].sum()` return?
# A4: It returns a 1D Pandas `Series` where `Product_Category` names are the index and the summed `Sales_Amount` values are the data.
#
# Q5: How can Pandas load other file formats besides CSV?
# A5: Using functions like `pd.read_excel()` for `.xlsx` files, `pd.read_json()` for `.json` files, and `pd.read_sql()` for database queries.
#
# Q6: Why do we use a Bar Chart here instead of a Line Chart?
# A6: `Product_Category` is discrete categorical data with no continuous time order, making a bar chart ideal for comparing magnitudes across categories.