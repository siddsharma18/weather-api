

import requests
import csv
from datetime import datetime, timedelta
import os

LAT = 40.7128   # New York latitude for Open-Meteo
LON = -74.0060  # New York longitude for Open-Meteo
OUTPUT_FILE = "historical_weather_data.csv"

def fetch_weather_data():
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=5)

    start_str = start_date.isoformat()
    end_str = end_date.isoformat()

    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={LAT}&longitude={LON}"
        f"&start_date={start_str}&end_date={end_str}"
        f"&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
        f"&timezone=auto"
    )

    print(f"Fetching data from {start_str} to {end_str}...")
    response = requests.get(url)
    data = response.json()

    if "hourly" in data:
        hourly = data["hourly"]
        results = []
        for i in range(len(hourly["time"])):
            results.append({
                "datetime": hourly["time"][i],
                "temp": hourly["temperature_2m"][i],
                "humidity": hourly["relative_humidity_2m"][i],
                "wind_speed": hourly["wind_speed_10m"][i],
                "weather_main": ""  # placeholder if needed
            })

        print(f"Saving to {OUTPUT_FILE}...")
        with open(OUTPUT_FILE, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=["datetime", "temp", "humidity", "wind_speed", "weather_main"])
            writer.writeheader()
            writer.writerows(results)

        print("✅ Done!")
    else:
        print("❌ Failed to fetch data.")

if __name__ == "__main__":
    fetch_weather_data()