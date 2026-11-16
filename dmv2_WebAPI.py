import requests
import pandas as pd
import matplotlib.pyplot as plt

api_key = "1f87700e618acc5cf4f567a0f071b329"
lat = 18.184135
lon = 74.610764

lat = 26.850000
lon = 80.950000
api_key = "1f87700e618acc5cf4f567a0f071b329"

url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}"

response = requests.get(url)
data = response.json()

if response.status_code == 200:
    temperatures = [x['main']['temp'] for x in data['list']]
    humidity = [x['main']['humidity'] for x in data['list']]
    wind_speed = [x['wind']['speed'] for x in data['list']]
    timestamps = [pd.to_datetime(x['dt'], unit='s') for x in data['list']]

    df = pd.DataFrame({
        'Timestamp': timestamps,
        'Temperature': temperatures,
        'Humidity': humidity,
        'Wind Speed': wind_speed
    })

    df.set_index('Timestamp', inplace=True)

    df.fillna(0, inplace=True)

    df['Temperature'] = df['Temperature'] - 273.15

    print("Maximum Temperature:", df['Temperature'].max())
    print("Minimum Temperature:", df['Temperature'].min())

    print(df)

    df['Temperature'].plot()
    plt.title("Temperature Over Time")
    plt.xlabel("Time")
    plt.ylabel("Temperature (Celsius)")
    plt.show()
else:
    print(f"Failed to fetch data. Status Code: {response.status_code}")
    print("Error Message:", data.get("message", "Unknown error"))

# ==========================================
# CODE EXPLANATION
# ==========================================
# - Imports `requests` to call web APIs, `pandas` for tabular structuring, and `matplotlib.pyplot` for plotting.
# - Sets latitude, longitude, and API key, then builds the OpenWeatherMap 5-day forecast endpoint URL using an f-string.
# - `requests.get(url)` sends an HTTP GET request, and `.json()` parses the JSON response into a Python dictionary.
# - Checks `response.status_code == 200` to verify the API call succeeded before extracting data.
# - Uses list comprehensions to extract temperature, humidity, wind speed, and Unix timestamps (`dt`) from `data['list']`.
# - `pd.to_datetime(x['dt'], unit='s')` converts Unix epoch seconds into readable date-time objects.
# - Creates a DataFrame, sets `Timestamp` as the index (`inplace=True`), and replaces any missing values with `0`.
# - Converts temperature from Kelvin (OpenWeatherMap's default unit) to Celsius by subtracting `273.15`.
# - Prints the max/min temperatures and full DataFrame, then plots temperature trends over time.
# - If the request fails (e.g., status 401 or 404), prints the HTTP status code and error message.

# ==========================================
# VIVA QUESTIONS & ANSWERS
# ==========================================
# Q1: What is an API, and what library is used here to access it?
# A1: An API (Application Programming Interface) allows two software systems to communicate over the web. Python's `requests` library is used here to send HTTP GET requests.
#
# Q2: What does HTTP status code `200` signify?
# A2: It means the HTTP request succeeded and the server returned the requested data (common error codes include `401` for invalid API key and `404` for not found).
#
# Q3: Why do we subtract `273.15` from `Temperature`?
# A3: By default, OpenWeatherMap returns temperatures in Kelvin; subtracting `273.15` converts them to Celsius (alternatively, adding `&units=metric` to the URL returns Celsius directly).
#
# Q4: What is a Unix timestamp (`x['dt']`), and how does `unit='s'` convert it?
# A4: A Unix timestamp is the number of seconds elapsed since January 1, 1970 (UTC). `pd.to_datetime(..., unit='s')` converts those seconds into a standard human-readable date and time.
#
# Q5: What is JSON, and what does `response.json()` do?
# A5: JSON (JavaScript Object Notation) is a lightweight key-value text format for data exchange. `response.json()` parses the raw JSON text into a native Python dictionary.
#
# Q6: What does `inplace=True` do in `set_index()` and `fillna()`?
# A6: It modifies the existing DataFrame directly in memory without needing to reassign it back to `df`.
#
# Q7: Why did `df['Temperature'].plot()` automatically put timestamps on the X-axis?
# A7: Because `Timestamp` was set as the DataFrame's index, and Pandas `.plot()` uses the index as the X-axis by default.
#
# Q8: What is a list comprehension?
# A8: A concise single-line syntax in Python (e.g., `[x['main']['temp'] for x in data['list']]`) used to create a new list by iterating over an existing iterable.