import geopandas as gpd

faults = gpd.read_file("gem_active_faults.geojson")

print(faults.shape)
print(faults.columns)
print(faults.crs)
print(faults.head(3))
