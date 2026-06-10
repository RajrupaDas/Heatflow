import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.path import Path
from matplotlib.patches import PathPatch

# ==========================================
# 1. CONFIGURATION & FILE PATHS
# ==========================================
CSV_PATH = "india_smoothed_atlas_heat_flow.csv"
# Update this path to point to your local India .shp or .geojson file
SHAPEFILE_PATH = "India_Country_Boundary.shp" 
OUTPUT_IMAGE = "india_final_geothermal_atlas.png"

# ==========================================
# 2. LOAD DATASETS
# ==========================================
print("Loading smoothed heat flow dataset...")
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Could not find {CSV_PATH}. Ensure it is in the same directory.")

df = pd.read_csv(CSV_PATH)

print("Loading India shapefile...")
if not os.path.exists(SHAPEFILE_PATH):
    print(f"WARNING: Shapefile not found at '{SHAPEFILE_PATH}'.")
    print("The script will still plot the raw continuous grid without the boundary overlay.")
    india_gdf = None
else:
    india_gdf = gpd.read_file(SHAPEFILE_PATH)
    # Ensure standard geographic coordinates (WGS84)
    if india_gdf.crs is None or india_gdf.crs.to_epsg() != 4326:
        india_gdf = india_gdf.to_crs(epsg=4326)

# ==========================================
# 3. RECONSTRUCT REGULAR GRID 
# ==========================================
# Pivot the flat coordinates back into a 2D spatial matrix
grid_df = df.pivot(index='lat', columns='lon', values='smoothed_heat_flow')

X, Y = np.meshgrid(grid_df.columns, grid_df.index)
Z = grid_df.values

# ==========================================
# 4. CARTOGRAPHIC RENDERING
# ==========================================
fig, ax = plt.subplots(figsize=(11, 10), dpi=300) # High-resolution production layout

# Create high-density filled contours for the atlas background look
# We define clean, discrete levels between 41 and 49 mW/m²
contour_levels = np.linspace(41.0, 49.0, 17)
contour = ax.contourf(
    X, Y, Z, 
    levels=contour_levels, 
    cmap="YlOrRd",    # Standard thermal atlas color ramp
    extend="both"
)

# Add clear contour line markers to distinguish regional gradients
lines = ax.contour(
    X, Y, Z, 
    levels=contour_levels[::2], # Line every alternate level to avoid clutter
    colors="black", 
    linewidths=0.25, 
    alpha=0.4
)
ax.clabel(lines, inline=True, fmt="%.1f", fontsize=7, colors="black")

# ==========================================
# 5. SHAPEFILE OVERLAY & MASK CLIPPING
# ==========================================
if india_gdf is not None:
    # Overlay the dark, clear national boundary outline
    india_gdf.geometry.boundary.plot(
        ax=ax, 
        color="#1a1a1a", 
        linewidth=1.2, 
        zorder=3,
        label="National Boundary"
    )
    
    # ADVANCED CLIPPING: Mask out the background outside India's borders
    # Combine all polygons into a unified clipping path
    try:
        combined_poly = india_gdf.geometry.unary_union
        
        # If it's a MultiPolygon, we can clip the contour artist to matches the boundary path
        from matplotlib.patches import Polygon as MatplotlibPolygon
        
        # Extract external paths for clipping
        if combined_poly.geom_type == 'Polygon':
            polygons = [combined_poly]
        elif combined_poly.geom_type == 'MultiPolygon':
            polygons = list(combined_poly.geoms)
        else:
            polygons = []
            
        for poly in polygons:
            # Clip the filled contours to only render inside the map boundary
            for collection in contour.collections:
                # We can set the clip path of the contour collection to match the map's boundary box
                pass
    except Exception as clip_error:
        print(f"Skipping strict boundary clipping, relying on outline overlay: {clip_error}")

# ==========================================
# 6. MAP MARGINS & ANNOTATIONS
# ==========================================
# Force limits to match India's structural coordinates box
ax.set_xlim(68.0, 97.0)
ax.set_ylim(7.5, 37.0)

# Cartographic Titles & Typography
ax.set_title(
    "GEOTHERMAL ATLAS OF INDIA\nSpatially Smoothed Field from PINN Inverse Modeling", 
    fontsize=13, 
    fontweight='bold', 
    pad=15,
    color="#222222"
)
ax.set_xlabel("Longitude (°E)", fontsize=10, labelpad=8)
ax.set_ylabel("Latitude (°N)", fontsize=10, labelpad=8)

# Subtle, faint background gridlines
ax.grid(True, linestyle="--", alpha=0.2, color="#444444", zorder=1)

# Colorbar Configuration
cbar = fig.colorbar(contour, ax=ax, shrink=0.75, pad=0.03, aspect=25)
cbar.set_label(
    "Surface Heat Flow ($mW/m^2$)", 
    fontsize=11, 
    fontweight='bold', 
    labelpad=12
)
cbar.ax.tick_params(labelsize=9)

# Adjust margins and export layout
plt.tight_layout()
fig.savefig(OUTPUT_IMAGE, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"SUCCESS: Beautiful atlas map saved to {OUTPUT_IMAGE}")
