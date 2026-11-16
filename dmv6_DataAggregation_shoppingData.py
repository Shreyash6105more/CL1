import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("customer_shopping_data.csv")

print(df.head())

print(df.groupby("shopping_mall").count())

branch_sales = df.groupby("shopping_mall").sum(numeric_only=True)

category_sales = df.groupby("category").sum(numeric_only=True)

print(branch_sales.sort_values(by="price", ascending=False))

print(category_sales.sort_values(by="price", ascending=False))

combined = df.groupby(["shopping_mall", "category"]).sum(numeric_only=True)
print(combined)

plt.pie(branch_sales["price"], labels=branch_sales.index)
plt.title("Sales by Shopping Mall")
plt.show()

plt.pie(category_sales["price"], labels=category_sales.index)
plt.title("Sales by Product Category")
plt.show()

pivot = df.pivot_table(
    index="shopping_mall",
    columns="category",
    values="price",
    aggfunc="sum"
)

pivot.plot(kind="bar", figsize=(10, 6))
plt.title("Category-wise Sales by Shopping Mall")
plt.xlabel("Shopping Mall")
plt.ylabel("Total Sales")
plt.show()

# ==========================================
# CODE EXPLANATION
# ==========================================
# - Imports pandas for data manipulation and matplotlib.pyplot for visualization.
# - pd.read_csv() loads the dataset into a DataFrame named 'df'.
# - df.head() displays the first 5 rows to inspect the structure of the data.
# - groupby().count() counts non-null entries per column for each shopping mall.
# - groupby().sum(numeric_only=True) calculates total numeric sums (like price/quantity) grouped by mall and category separately.
# - sort_values(by="price", ascending=False) ranks malls and categories from highest to lowest total sales.
# - groupby(["shopping_mall", "category"]) creates a multi-level summary showing sales for each category inside each mall.
# - plt.pie() plots two pie charts showing percentage distribution of total sales by mall and by category.
# - df.pivot_table() reshapes data into a 2D matrix with malls as rows, categories as columns, and summed prices as values.
# - pivot.plot(kind="bar") generates a grouped bar chart comparing category sales across different malls.

# ==========================================
# VIVA QUESTIONS & ANSWERS
# ==========================================
# Q1: What is a DataFrame in Pandas?
# A1: A 2D tabular data structure with labeled rows and columns, similar to an Excel spreadsheet or SQL table.
#
# Q2: Why is `numeric_only=True` used inside `.sum()`?
# A2: It ensures Pandas only adds numerical columns (like price and quantity) and ignores text/string columns, preventing errors.
#
# Q3: How does `groupby()` work internally?
# A3: It follows a "Split-Apply-Combine" strategy—splitting data into groups, applying an aggregation function (sum/count), and combining the results.
#
# Q4: What is the difference between `groupby()` and `pivot_table()`?
# A4: `groupby()` returns a stacked/long Series or DataFrame (MultiIndex), whereas `pivot_table()` reshapes data into a wide 2D grid (rows vs. columns) ideal for plotting.
#
# Q5: What does `branch_sales.index` represent in `plt.pie()`?
# A5: After grouping by "shopping_mall", the mall names become the DataFrame's index, which provides the labels for the pie slices.
#
# Q6: What is the purpose of `ascending=False` in `sort_values()`?
# A6: It sorts values in descending order (highest to lowest) so top performers appear first.
#
# Q7: Why do we call `plt.show()` after each plot?
# A7: It renders and displays the current figure on screen and clears the canvas so subsequent plots don't overlap on the same chart.
#
# Q8: What does `figsize=(10, 6)` do?
# A8: It sets the width to 10 inches and height to 6 inches for better readability of grouped bars.