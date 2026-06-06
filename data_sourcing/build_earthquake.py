# download_earthquakes.py

import requests
import pandas as pd

url = (
    "https://earthquake.usgs.gov/fdsnws/event/1/query"
)

params = {
    "format": "geojson",
    "starttime": "2005-01-01",
    "endtime": "2025-12-31",
    "minmagnitude": 2.5,
    "minlatitude": 6,
    "maxlatitude": 38,
    "minlongitude": 68,
    "maxlongitude": 98,
    "limit": 20000
}

r = requests.get(
    url,
    params=params,
    timeout=120
)

data = r.json()

rows = []

for feature in data["features"]:

    props = feature["properties"]
    geom = feature["geometry"]

    rows.append({
        "time": props["time"],
        "mag": props["mag"],
        "place": props["place"],
        "latitude": geom["coordinates"][1],
        "longitude": geom["coordinates"][0],
        "depth": geom["coordinates"][2]
    })

df = pd.DataFrame(rows)

print(df.head())
print()
print("Rows:", len(df))

df.to_csv(
    "earthquakes.csv",
    index=False
)

print("\nSaved earthquakes.csv")
