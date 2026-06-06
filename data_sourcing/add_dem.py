import pandas as pd
import requests
import time

# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "master_boreholes_stage2.parquet"
OUTPUT_FILE = "master_boreholes_stage3.parquet"

API_URL = "https://api.opentopodata.org/v1/srtm30m"

BATCH_SIZE = 100

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_parquet(INPUT_FILE)

print("Rows:", len(df))

# ============================================================
# FUNCTION
# ============================================================

def query_batch(latitudes, longitudes):

    locations = "|".join(
        f"{lat},{lon}"
        for lat, lon in zip(latitudes, longitudes)
    )

    payload = {
        "locations": locations,
        "interpolation": "bilinear"
    }

    r = requests.post(API_URL, json=payload, timeout=60)

    r.raise_for_status()

    data = r.json()

    if data["status"] != "OK":
        raise RuntimeError(data)

    return [
        item["elevation"]
        for item in data["results"]
    ]

# ============================================================
# DOWNLOAD DEM
# ============================================================

elevations = []

n = len(df)

for start in range(0, n, BATCH_SIZE):

    end = min(start + BATCH_SIZE, n)

    batch = df.iloc[start:end]

    print(
        f"Processing {start} -> {end}"
    )

    vals = query_batch(
        batch["lat"].tolist(),
        batch["lon"].tolist()
    )

    elevations.extend(vals)

    time.sleep(0.25)

# ============================================================
# ADD COLUMN
# ============================================================

df["elevation_dem_m"] = elevations

# ============================================================
# CHECK
# ============================================================

print("\nDEM Statistics:")
print(df["elevation_dem_m"].describe())

print("\nMissing DEM:")
print(df["elevation_dem_m"].isna().sum())

print("\nSample:")
print(
    df[
        [
            "lat",
            "lon",
            "elevation_dem_m"
        ]
    ].head()
)

# ============================================================
# SAVE
# ============================================================

df.to_parquet(
    OUTPUT_FILE,
    index=False
)

print("\nSaved:")
print(OUTPUT_FILE)
