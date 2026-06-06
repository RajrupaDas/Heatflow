import pandas as pd
import geopandas as gpd
import numpy as np

# =====================================================
# FILES
# =====================================================

BOREHOLES = "master_boreholes_stage4.parquet"
EARTHQUAKES = "earthquakes.csv"

OUTPUT = "master_boreholes_stage5.parquet"

# =====================================================
# LOAD BOREHOLES
# =====================================================

boreholes = pd.read_parquet(BOREHOLES)

bore_gdf = gpd.GeoDataFrame(
    boreholes,
    geometry=gpd.points_from_xy(
        boreholes.lon,
        boreholes.lat
    ),
    crs="EPSG:4326"
)

# =====================================================
# LOAD EARTHQUAKES
# =====================================================

eq = pd.read_csv(EARTHQUAKES)

eq = eq.dropna(
    subset=["latitude", "longitude", "mag"]
)

eq_gdf = gpd.GeoDataFrame(
    eq,
    geometry=gpd.points_from_xy(
        eq.longitude,
        eq.latitude
    ),
    crs="EPSG:4326"
)

# =====================================================
# INDIA REGION FILTER
# =====================================================

eq_gdf = eq_gdf.cx[68:98, 6:38]

print("Earthquakes in India bbox:", len(eq_gdf))

# =====================================================
# PROJECT
# =====================================================

bore_gdf = bore_gdf.to_crs(3857)
eq_gdf = eq_gdf.to_crs(3857)

# =====================================================
# SPATIAL INDEX
# =====================================================

eq_sindex = eq_gdf.sindex

# =====================================================
# FEATURE ARRAYS
# =====================================================

count_50 = []
count_100 = []

mean_mag_50 = []
mean_mag_100 = []

max_mag_100 = []

# =====================================================
# LOOP
# =====================================================

for i, point in enumerate(bore_gdf.geometry):

    if i % 50 == 0:
        print(f"{i}/{len(bore_gdf)}")

    r50 = 50000
    r100 = 100000

    bbox100 = (
        point.x - r100,
        point.y - r100,
        point.x + r100,
        point.y + r100
    )

    candidates = list(
        eq_sindex.intersection(bbox100)
    )

    nearby = eq_gdf.iloc[candidates]

    distances = nearby.geometry.distance(point)

    eq50 = nearby[distances <= r50]
    eq100 = nearby[distances <= r100]

    # -------------------
    # counts
    # -------------------

    count_50.append(len(eq50))
    count_100.append(len(eq100))

    # -------------------
    # mean magnitudes
    # -------------------

    if len(eq50) > 0:
        mean_mag_50.append(eq50["mag"].mean())
    else:
        mean_mag_50.append(np.nan)

    if len(eq100) > 0:
        mean_mag_100.append(eq100["mag"].mean())
        max_mag_100.append(eq100["mag"].max())
    else:
        mean_mag_100.append(np.nan)
        max_mag_100.append(np.nan)

# =====================================================
# SAVE FEATURES
# =====================================================

bore_gdf["eq_count_50km"] = count_50
bore_gdf["eq_count_100km"] = count_100

bore_gdf["mean_mag_50km"] = mean_mag_50
bore_gdf["mean_mag_100km"] = mean_mag_100

bore_gdf["max_mag_100km"] = max_mag_100

# back to normal dataframe

bore_gdf = bore_gdf.drop(columns=["geometry"])

# =====================================================
# CHECK
# =====================================================

print("\nFeature summary:\n")

cols = [
    "eq_count_50km",
    "eq_count_100km",
    "mean_mag_50km",
    "mean_mag_100km",
    "max_mag_100km"
]

print(
    bore_gdf[cols].describe()
)

# =====================================================
# SAVE
# =====================================================

bore_gdf.to_parquet(
    OUTPUT,
    index=False
)

print("\nSaved:")
print(OUTPUT)
