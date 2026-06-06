import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# =====================================================
# LOAD BOREHOLES
# =====================================================

df = pd.read_parquet(
    "master_boreholes_stage1.parquet"
)

print("Boreholes:", len(df))

boreholes = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(
        df.lon,
        df.lat
    ),
    crs="EPSG:4326"
)

# =====================================================
# LOAD FAULTS
# =====================================================

faults = gpd.read_file(
    "gem_active_faults.geojson"
)

# keep India region only
faults = faults.cx[68:98, 6:38]

print("India faults:", len(faults))

# =====================================================
# PROJECT
# =====================================================

boreholes = boreholes.to_crs(3857)
faults = faults.to_crs(3857)

# =====================================================
# NEAREST DISTANCE (OPTIMIZED & FIXED)
# =====================================================

# sjoin_nearest finds the closest fault for each borehole. 
# distance_col automatically calculates the projected distance in meters.
boreholes = gpd.sjoin_nearest(
    boreholes, 
    faults, 
    how="left", 
    distance_col="distance_m"
)

# Convert distance to kilometers (and drop the extra index column added by sjoin)
boreholes["fault_distance_km"] = boreholes["distance_m"] / 1000.0
boreholes = boreholes.drop(columns=["distance_m"])
if "index_right" in boreholes.columns:
    boreholes = boreholes.drop(columns=["index_right"])

# =====================================================
# RETURN TO LAT/LON
# =====================================================

boreholes = boreholes.to_crs(
    "EPSG:4326"
)

# Convert back to standard DataFrame by dropping geometry
boreholes = pd.DataFrame(boreholes.drop(columns=["geometry"]))

# =====================================================
# CHECK
# =====================================================

print()

print("Output rows:")
print(len(boreholes))

print()

print(
    boreholes[
        [
            "lat",
            "lon",
            "fault_distance_km"
        ]
    ].head()
)

print()

print(
    boreholes["fault_distance_km"].describe()
)

# =====================================================
# SAVE
# =====================================================

boreholes.to_parquet(
    "master_boreholes_stage2.parquet",
    index=False
)

print()
print("Saved:")
print("master_boreholes_stage2.parquet")
