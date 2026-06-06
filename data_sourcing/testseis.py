import pandas as pd

url = (
    "https://earthquake.usgs.gov/fdsnws/event/1/query.csv"
    "?starttime=2005-01-01"
    "&endtime=2025-12-31"
    "&minlatitude=6"
    "&maxlatitude=38"
    "&minlongitude=68"
    "&maxlongitude=98"
    "&minmagnitude=2.5"
)

df = pd.read_csv(url)

print(df.head())
print(len(df))
