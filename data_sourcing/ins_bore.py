import pandas as pd

FILE = "india_heatflow.xlsx"

df = pd.read_excel(FILE)

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nInfo:")
print(df.info())

print("\nMissing values:")
print(df.isna().sum())
