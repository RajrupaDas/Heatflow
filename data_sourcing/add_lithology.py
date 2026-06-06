import pandas as pd
import requests
import time

# =====================================================
# CONFIG
# =====================================================

INPUT_FILE = "master_boreholes_stage3.parquet"
OUTPUT_FILE = "master_boreholes_stage4.parquet"

# =====================================================
# LOAD
# =====================================================

df = pd.read_parquet(INPUT_FILE)

print("Rows:", len(df))

# =====================================================
# QUERY FUNCTION
# =====================================================

def get_lithology(lat, lon):

    url = (
        "https://macrostrat.org/api/v2/geologic_units/map"
        f"?lat={lat}&lng={lon}"
    )

    try:

        r = requests.get(url, timeout=30)

        if r.status_code != 200:
            return None

        data = r.json()

        records = data["success"]["data"]

        if len(records) == 0:
            return None

        rec = records[0]

        return {
            "lithology_text": rec.get("lith"),
            "unit_name": rec.get("name"),
            "age_name": rec.get("best_int_name"),
            "rock_age_top_ma": rec.get("t_age"),
            "rock_age_bottom_ma": rec.get("b_age")
        }

    except Exception:
        return None
# =====================================================
# DOWNLOAD
# =====================================================

lithology_text = []
top_age = []
bottom_age = []

for i, row in df.iterrows():

    if i % 20 == 0:
        print(f"{i}/{len(df)}")

    result = get_lithology(
        row["lat"],
        row["lon"]
    )

    if result is None:

        lithology_text.append(None)
        top_age.append(None)
        bottom_age.append(None)

    else:

        lithology_text.append(
            result["lithology_text"]
        )

        top_age.append(
            result["rock_age_top_ma"]
        )

        bottom_age.append(
            result["rock_age_bottom_ma"]
        )

    time.sleep(0.1)

# =====================================================
# ADD COLUMNS
# =====================================================

df["lithology_text"] = lithology_text
df["rock_age_top_ma"] = top_age
df["rock_age_bottom_ma"] = bottom_age

df["rock_age_mean_ma"] = (
    df["rock_age_top_ma"]
    +
    df["rock_age_bottom_ma"]
) / 2

# =====================================================
# CHECK
# =====================================================

print("\nLithology counts:")
print(
    df["lithology_text"]
    .value_counts(dropna=False)
    .head(20)
)

print("\nMissing lithology:")
print(
    df["lithology_text"]
    .isna()
    .sum()
)

# =====================================================
# SAVE
# =====================================================

df.to_parquet(
    OUTPUT_FILE,
    index=False
)

print("\nSaved:")
print(OUTPUT_FILE)
