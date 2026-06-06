import pandas as pd
import numpy as np

# ---------------------------
# Load boreholes
# ---------------------------

boreholes = pd.read_parquet(
    "master_boreholes_stage0.parquet"
)

# ---------------------------
# Load crust model
# ---------------------------

crust = pd.read_csv(
    "crustal_thickness.xyz",
    sep=r"\s+",
    header=None,
    names=["lon", "lat", "crust_thickness_km"],
    engine="python"
)

sediment = pd.read_csv(
    "sediment_thickness.xyz",
    sep=r"\s+",
    header=None,
    names=["lon", "lat", "sediment_thickness_km"],
    engine="python"
)

# ---------------------------
# Create lookup tables
# ---------------------------

crust_lookup = {
    (row.lat, row.lon): row.crust_thickness_km
    for row in crust.itertuples()
}

sed_lookup = {
    (row.lat, row.lon): row.sediment_thickness_km
    for row in sediment.itertuples()
}

# ---------------------------
# CRUST1.0 grid centers
#
# Example:
# 77.2 -> 77.5
# 77.8 -> 77.5
# ---------------------------

def nearest_grid_center(x):
    return np.floor(x) + 0.5

# ---------------------------
# Lookup features
# ---------------------------

crust_vals = []
sed_vals = []

for row in boreholes.itertuples():

    grid_lat = nearest_grid_center(row.lat)
    grid_lon = nearest_grid_center(row.lon)

    crust_vals.append(
        crust_lookup.get(
            (grid_lat, grid_lon),
            np.nan
        )
    )

    sed_vals.append(
        sed_lookup.get(
            (grid_lat, grid_lon),
            np.nan
        )
    )

boreholes["crust_thickness_km"] = crust_vals
boreholes["sediment_thickness_km"] = sed_vals

# ---------------------------
# Quick sanity check
# ---------------------------

print(
    boreholes[
        [
            "lat",
            "lon",
            "crust_thickness_km",
            "sediment_thickness_km"
        ]
    ].head()
)

print("\nMissing crust:")
print(
    boreholes["crust_thickness_km"].isna().sum()
)

print("\nMissing sediment:")
print(
    boreholes["sediment_thickness_km"].isna().sum()
)

# ---------------------------
# Save
# ---------------------------

boreholes.to_parquet(
    "master_boreholes_stage1.parquet",
    index=False
)

print("\nSaved:")
print("master_boreholes_stage1.parquet")
