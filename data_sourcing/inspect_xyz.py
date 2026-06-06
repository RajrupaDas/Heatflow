import pandas as pd
import numpy as np

FILES = [
    "crustal_thickness.xyz",
    "sediment_thickness.xyz"
]


def inspect_xyz(fname):

    print("\n" + "=" * 100)
    print(fname)
    print("=" * 100)

    df = pd.read_csv(
        fname,
        sep=r"\s+",
        header=None,
        names=["lon", "lat", "value"],
        engine="python"
    )

    print("\nShape:")
    print(df.shape)

    print("\nFirst 10 rows:")
    print(df.head(10))

    print("\nLast 10 rows:")
    print(df.tail(10))

    print("\nColumn ranges:")
    print(df.describe())

    print("\nLongitude range:")
    print(df.lon.min(), "->", df.lon.max())

    print("\nLatitude range:")
    print(df.lat.min(), "->", df.lat.max())

    print("\nValue range:")
    print(df.value.min(), "->", df.value.max())

    print("\nUnique longitude count:")
    print(df.lon.nunique())

    print("\nUnique latitude count:")
    print(df.lat.nunique())

    lon_step = np.diff(
        np.sort(df.lon.unique())
    )

    lat_step = np.diff(
        np.sort(df.lat.unique())
    )

    print("\nLongitude spacing:")
    print(np.unique(np.round(lon_step, 6))[:20])

    print("\nLatitude spacing:")
    print(np.unique(np.round(lat_step, 6))[:20])

    print("\nExpected grid cells:")
    print(
        df.lon.nunique() *
        df.lat.nunique()
    )

    print("\nActual rows:")
    print(len(df))

    print("\nMissing cells:")
    print(
        df.lon.nunique() *
        df.lat.nunique()
        - len(df)
    )

    print("\nRandom samples:")
    print(df.sample(10, random_state=42))


for f in FILES:
    inspect_xyz(f)
