import pandas as pd

df = pd.read_excel("india_heatflow.xlsx")

cols = [
    "q",
    "lat_NS",
    "long_EW",
    "elevation",
    "total_depth_MD",
    "T_grad_mean",
    "tc_mean",
    "geo_lithology",
    "geo_stratigraphy"
]

master = df[cols].copy()

master = master.rename(
    columns={
        "lat_NS": "lat",
        "long_EW": "lon",
        "q": "heat_flow"
    }
)

print(master.shape)
print(master.head())

master.to_parquet(
    "master_boreholes_stage0.parquet",
    index=False
)
