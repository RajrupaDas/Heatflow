import geopandas as gpd

countries = gpd.read_file("India_Country_Boundary.shp")

print(countries.columns)

print("\nArea values:\n")
print(countries["Area"])
