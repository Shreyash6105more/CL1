import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("aqi.csv", encoding="latin-1")

print(data.head())
print(data.info())
print(data.isnull().sum())

cols = ["so2", "no2", "rspm", "spm"]

for col in cols:
    data[col] = data[col].fillna(data[col].mean())

data["date"] = pd.to_datetime(
    data["sampling_date"], errors="coerce"
)

data = data.dropna(subset=["date"])

data["year"] = data["date"].dt.year

yearly = data.groupby("year")[cols].mean()

yearly.plot(kind="line", figsize=(10, 5))
plt.title("Yearly Air Quality Trends")
plt.xlabel("Year")
plt.ylabel("Pollutant Level")
plt.legend()
plt.show()

data[cols].mean().plot(kind="bar")
plt.title("Average Pollutant Levels")
plt.xlabel("Pollutants")
plt.ylabel("Average Level")
plt.show()

data[cols].plot(kind="box")
plt.title("Pollutant Distribution")
plt.show()

plt.scatter(data["so2"], data["no2"])
plt.title("SO2 vs NO2")
plt.xlabel("SO2")
plt.ylabel("NO2")
plt.show()

print("Data Visualization Completed!")

# ==========================================
# CODE EXPLANATION
# ==========================================
# - Imports pandas for data cleaning/analysis and matplotlib.pyplot for plotting.
# - pd.read_csv(..., encoding="latin-1") loads the CSV file while handling special/non-UTF-8 characters.
# - head(), info(), and isnull().sum() inspect sample rows, data types, and missing value counts per column.
# - A loop replaces missing (NaN) values in key pollutant columns (so2, no2, rspm, spm) with each column's mean.
# - pd.to_datetime(..., errors="coerce") converts string dates into datetime objects, turning invalid formats into NaT.
# - dropna(subset=["date"]) removes rows where the date conversion failed (NaT).
# - data["date"].dt.year extracts the year from the datetime column into a new 'year' column.
# - Line plot: Groups data by 'year', computes mean pollutant levels, and plots multi-line trends over time.
# - Bar plot: Compares overall average concentrations across the four pollutants.
# - Box plot: Displays the statistical distribution (median, quartiles, and outliers) of each pollutant.
# - Scatter plot: Visualizes the correlation/relationship between SO2 and NO2 levels.

# ==========================================
# VIVA QUESTIONS & ANSWERS
# ==========================================
# Q1: Why is `encoding="latin-1"` used in `read_csv()`?
# A1: Default UTF-8 encoding fails on certain special characters or symbols found in older CSV files; `latin-1` reads all byte values without throwing a UnicodeDecodeError.
#
# Q2: What does `errors="coerce"` do in `pd.to_datetime()`?
# A2: Instead of crashing on unparseable or corrupted date strings, it converts invalid dates into `NaT` (Not a Time), which can then be safely dropped.
#
# Q3: Why did we impute missing pollutant values with `.mean()` instead of dropping rows?
# A3: Dropping rows with any missing pollutant would cause significant data loss; mean imputation preserves dataset size while keeping the column average unchanged.
#
# Q4: What is the difference between `NaN` and `NaT` in Pandas?
# A4: `NaN` (Not a Number) represents missing numeric/object data, whereas `NaT` (Not a Time) represents missing datetime/timestamp data.
#
# Q5: What is `.dt` in `data["date"].dt.year`?
# A5: `.dt` is the datetime accessor in Pandas that allows extracting date/time properties (like year, month, day) from a datetime Series.
#
# Q6: What information does a Box Plot show?
# A6: It shows the five-number summary: minimum, first quartile (Q1), median (Q2), third quartile (Q3), and maximum, along with outliers as individual points.
#
# Q7: When do we use a Line Plot vs. a Scatter Plot?
# A7: A line plot shows continuous trends over time (e.g., yearly changes), while a scatter plot shows the relationship or correlation between two numerical variables (SO2 vs. NO2).
#
# Q8: What do RSPM and SPM stand for in air quality data?
# A8: RSPM stands for Respirable Suspended Particulate Matter, and SPM stands for Suspended Particulate Matter.