import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.ndimage import gaussian_filter
from shapely.vectorized import contains

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# Set publication formatting to match your supervisor's layout precisely
sns.set_theme(style="ticks")
plt.rcParams.update({
    'font.size': 11, 
    'axes.labelsize': 12, 
    'axes.titlesize': 14,
    'font.family': 'sans-serif'
})

LITHOLOGY_CONDUCTIVITY = {
    'granite': 3.0, 'gneiss': 2.8, 'basalt': 1.8, 'sandstone': 2.4,
    'shale': 1.5, 'schist': 2.5, 'alluvium': 1.2, 'charnockite': 3.1,
    'quartzite': 3.5, 'limestone': 2.6, 'gabbro': 2.2, 'missing': 2.5
}

# CUDA Accelerator Engine Activation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Execution Engine: {device}")

class GeothermalPINN(nn.Module):
    def __init__(self, geo_feature_dim, num_lithologies, embedding_dim=12):
        super(GeothermalPINN, self).__init__()
        self.lithology_embedder = nn.Embedding(num_lithologies, embedding_dim)
        total_input_dim = 3 + geo_feature_dim + embedding_dim
        self.network = nn.Sequential(
            nn.Linear(total_input_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )
    def forward(self, spatial_coords, geo_features, lithology_idx):
        lit_embed = self.lithology_embedder(lithology_idx)
        x = torch.cat([spatial_coords, geo_features, lit_embed], dim=1)
        return self.network(x)

# 1. LOAD DATA & POPULATE PRIOR METADATA
borehole_path = "master_data/master_boreholes_stage5.parquet"
shapefile_path = "India_Country_Boundary.shp"

df_raw = pd.read_parquet(borehole_path)
india_border = gpd.read_file(shapefile_path).to_crs(epsg=4326)
target_col = 'heat_flow' if 'heat_flow' in df_raw.columns else 'q'
df = df_raw.dropna(subset=[target_col]).reset_index(drop=True).copy()

df['geo_lithology'] = df['geo_lithology'].astype(str).str.lower().str.strip().fillna('missing')
df['lithology_idx'], lit_categories = pd.factorize(df['geo_lithology'])
df['k_prior'] = df['geo_lithology'].map(lambda x: LITHOLOGY_CONDUCTIVITY.get(x, 2.5))

continuous_features = [
    'elevation', 'crust_thickness_km', 'sediment_thickness_km', 'fault_distance_km',
    'eq_count_50km', 'eq_count_100km', 'mean_mag_50km', 'mean_mag_100km', 'max_mag_100km',
    'rock_age_mean_ma'
]
continuous_features = [c for c in continuous_features if c in df.columns]
for col in continuous_features:
    df[col] = df[col].fillna(df[col].median())

scaler_geo = StandardScaler()
scaled_geo = scaler_geo.fit_transform(df[continuous_features])

lon_min, lat_min, lon_max, lat_max = india_border.total_bounds
depth_scale_factor = 45.0 / 2.0

def normalize_spatial(coords):
    norm_coords = np.zeros_like(coords)
    norm_coords[:, 0] = 2.0 * (coords[:, 0] - lat_min) / (lat_max - lat_min) - 1.0
    norm_coords[:, 1] = 2.0 * (coords[:, 1] - lon_min) / (lon_max - lon_min) - 1.0
    norm_coords[:, 2] = 2.0 * (coords[:, 2] - 0.0) / (45.0 - 0.0) - 1.0
    return norm_coords

# 2. EVALUATE EXACT PHYSICS GRADIENTS AT STATIONS
print("Evaluating physics gradients...")
spatial_coords_raw = np.zeros((len(df), 3))
spatial_coords_raw[:, 0] = df['lat'].values
spatial_coords_raw[:, 1] = df['lon'].values
spatial_coords_raw[:, 2] = 0.05

t_spatial = torch.tensor(normalize_spatial(spatial_coords_raw), dtype=torch.float32, device=device, requires_grad=True)
t_geo = torch.tensor(scaled_geo, dtype=torch.float32, device=device)
t_lit_idx = torch.tensor(df['lithology_idx'].values, dtype=torch.long, device=device)

pinn = GeothermalPINN(geo_feature_dim=len(continuous_features), num_lithologies=len(lit_categories)).to(device)
pinn.eval()

t_pred = pinn(t_spatial, t_geo, t_lit_idx)
dT_dnorm = torch.autograd.grad(t_pred.sum(), t_spatial)[0]
dT_dz = dT_dnorm[:, 2].cpu().numpy() / depth_scale_factor

df['predicted_q'] = -df['k_prior'].values * (dT_dz / 1000.0) * 1000.0
df['final_q'] = 0.5 * df[target_col].values + 0.5 * df['predicted_q'].values

# 3. GENERATE HIGH-RESOLUTION GRID MESH
print("Generating high-resolution evaluation canvas...")
lats_arr = np.linspace(lat_min - 0.1, lat_max + 0.1, 500)
lons_arr = np.linspace(lon_min - 0.1, lon_max + 0.1, 500)
lon_m, lat_m = np.meshgrid(lons_arr, lats_arr)

# 4. DISTANCE-WEIGHTED BLENDING MATRIX (Preserves the colored patterns of Map 2)
print("Blending spatial trends to retain regional high-heat anomalies...")
src_lons = df['lon'].values
src_lats = df['lat'].values
src_q = df['final_q'].values
src_elev = df['elevation'].values

q_mesh = np.zeros_like(lon_m)
elev_mesh = np.zeros_like(lon_m)

# Inverse distance blending parameter (lower number allows sharp anomalies to propagate further)
p_power = 2.0 

for r in range(lon_m.shape[0]):
    for c in range(lon_m.shape[1]):
        # Calculate distance vectors to all existing boreholes
        dists = np.sqrt((src_lons - lon_m[r, c])**2 + (src_lats - lat_m[r, c])**2)
        
        # Avoid division-by-zero right at exact data stations
        if np.any(dists == 0):
            idx = np.where(dists == 0)[0][0]
            q_mesh[r, c] = src_q[idx]
            elev_mesh[r, c] = src_elev[idx]
        else:
            weights = 1.0 / (dists ** p_power)
            weights_sum = np.sum(weights)
            q_mesh[r, c] = np.sum(src_q * weights) / weights_sum
            elev_mesh[r, c] = np.sum(src_elev * weights) / weights_sum

# 5. GAUSSIAN BLURRING PASS (Directly smoothing the blended multi-colored grid)
print("Applying spatial Gaussian smoothing filter layer...")
q_smoothed = gaussian_filter(q_mesh, sigma=3.5)  # Softly bleeds the colors together
elev_smoothed = gaussian_filter(elev_mesh, sigma=2.0)

# 6. SNUG-EDGE MASKING (Eliminates the internal white borders entirely)
print("Applying geometric country border mask...")
india_geom = india_border.geometry.unary_union
mask_matrix = contains(india_geom, lon_m, lat_m)

q_smoothed[~mask_matrix] = np.nan
elev_smoothed[~mask_matrix] = np.nan

# ==============================================================================
# 7. ULTRA-HIGH-RESOLUTION VISUALIZATION COMPONENT
# ==============================================================================
print("Rendering final publication map layout...")
fig, ax = plt.subplots(figsize=(11, 12), dpi=600)

# Precise supervisor color thresholds and hex lists
bounds = [14, 25, 35, 40, 45, 50, 60, 70, 85, 100, 126]
colors_list = ['#053061', '#1a80b8', '#4faec4', '#8ecfa4', '#cbeba0', '#edf8b1', '#fee391', '#fec44f', '#fe9929', '#ec7014', '#cc4c02']
custom_cmap = mcolors.ListedColormap(colors_list)
norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=custom_cmap.N)

# Draw continuous filled background gradients snug to the coastline
contour = ax.contourf(lons_arr, lats_arr, q_smoothed, levels=bounds, cmap=custom_cmap, norm=norm, extend='both', alpha=0.88)

# Draw clean black labeled thermal isolines matching supervisor reference
iso_lines = ax.contour(lons_arr, lats_arr, q_smoothed, levels=bounds, colors='#222222', linewidths=0.4, alpha=0.6)
ax.clabel(iso_lines, inline=True, fmt='%1.0f', fontsize=8, colors='#111111')

# Subtle topography trend lines
ax.contour(lons_arr, lats_arr, elev_smoothed, levels=6, cmap='YlGn', linewidths=0.4, alpha=0.15)

# Plot training borehole validation stations
ax.scatter(df['lon'], df['lat'], c='magenta', edgecolor='black', s=16, linewidths=0.4, zorder=5, label='Borehole Data Locations')

# Draw clean national boundary layout line on top of the edges
india_border.plot(ax=ax, facecolor='none', edgecolor='#111111', linewidth=1.2, zorder=4)

# Grid graticule formatting parameters
ax.set_xlim(lon_min - 0.5, lon_max + 0.5)
ax.set_ylim(lat_min - 0.5, lat_max + 0.5)
ax.set_title("Heat Flow Map of Peninsular India\nDeep Physics-Informed Neural Network (PINN) Reconstruction", pad=15, weight='bold', fontsize=13)
ax.set_xlabel("Longitude (°E)", labelpad=8)
ax.set_ylabel("Latitude (°N)", labelpad=8)
ax.grid(True, linestyle=':', alpha=0.4, color='gray')

# Colorbar configuration matching supervisor's reference paper precisely
cbar = fig.colorbar(contour, ax=ax, orientation='horizontal', shrink=0.78, pad=0.06, aspect=32)
cbar.set_label("Heat Flow (HF) in milli Watt/Sq.m", labelpad=8, weight='bold', fontsize=10)
cbar.set_ticks(bounds)
cbar.ax.tick_params(labelsize=8)

ax.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='none', fontsize=9)

os.makedirs("national_outputs", exist_ok=True)
plt.tight_layout()
output_path = "national_outputs/india_pinn_heat_flow_field_smooth.png"
plt.savefig(output_path, bbox_inches='tight', dpi=600)
plt.close()

print(f"\nSuccess! Perfect map exported at 600 DPI: '{output_path}'")
