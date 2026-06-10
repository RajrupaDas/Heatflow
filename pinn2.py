import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns
from scipy.ndimage import gaussian_filter

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

# Set up publication formatting matching your target reference
sns.set_theme(style="ticks")
plt.rcParams.update({
    'font.size': 11, 
    'axes.labelsize': 12, 
    'axes.titlesize': 14,
    'font.family': 'sans-serif'
})

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Executing Geothermal PINN Pipeline on Core Unit: {device}")

# ==============================================================================
# PARTS 2 & 3: GEOPHYSICAL PRIOR LOOKUP TABLES (FIXED, NOT LEARNED)
# ==============================================================================
LITHOLOGY_CONDUCTIVITY = {
    'granite': 3.0, 'gneiss': 2.8, 'basalt': 1.8, 'sandstone': 2.4,
    'shale': 1.5, 'schist': 2.5, 'alluvium': 1.2, 'charnockite': 3.1,
    'quartzite': 3.5, 'limestone': 2.6, 'gabbro': 2.2, 'missing': 2.5
}

def get_radiogenic_heat(lithology, mean_age_ma):
    lit = str(lithology).lower().strip()
    age = float(mean_age_ma) if pd.notna(mean_age_ma) else 500.0
    
    if 'granite' in lit or 'gneiss' in lit:
        if age > 2500: return 3.5     # Archean Craton signatures
        if age > 541:  return 2.5     # Proterozoic
        return 1.8                    # Phanerozoic
    elif 'basalt' in lit or 'gabbro' in lit:
        return 0.4
    elif 'shale' in lit or 'schist' in lit:
        return 1.5
    return 1.0  # Baseline crustal baseline median (μW/m³)

# ==============================================================================
# PART 1: GEOTHERMAL PINN NET ARCHITECTURE (STRICT TO BLUEPRINT CONSTRAINTS)
# ==============================================================================
class GeothermalPINN(nn.Module):
    def __init__(self, geo_feature_dim, num_lithologies, embedding_dim=12):
        super(GeothermalPINN, self).__init__()
        
        # Categorical Embedding Layer for Lithology labels
        self.lithology_embedder = nn.Embedding(num_lithologies, embedding_dim)
        
        # Total Input Layout = 3 Spatial (x,y,z) + Continuous Context Variables + Learned Embedding
        total_input_dim = 3 + geo_feature_dim + embedding_dim
        
        # Fully connected network with Swish/SiLU activations for robust second-order derivatives
        self.network = nn.Sequential(
            nn.Linear(total_input_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1)  # EXACT BLUEPRINT OUTPUT: Scalar Temperature Field T(x,y,z) only
        )
        
    def forward(self, spatial_coords, geo_features, lithology_idx):
        lit_embed = self.lithology_embedder(lithology_idx)
        x = torch.cat([spatial_coords, geo_features, lit_embed], dim=1)
        return self.network(x)

# ==============================================================================
# PART 5: COLLOCATION POINT SAMPLING STRATEGY
# ==============================================================================
def generate_collocation_points(india_border, num_points=15000):
    lon_min, lat_min, lon_max, lat_max = india_border.total_bounds
    points = []
    
    while len(points) < num_points:
        lons = np.random.uniform(lon_min, lon_max, num_points)
        lats = np.random.uniform(lat_min, lat_max, num_points)
        
        gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lons, lats), crs="EPSG:4326")
        inside = gpd.sjoin(gdf, india_border, how="inner", predicate="within")
        
        if not inside.empty:
            x_coords = inside.geometry.x.values
            y_coords = inside.geometry.y.values
            for i in range(len(inside)):
                depth_km = np.random.uniform(0.0, 45.0)  # Down to deep lithospheric boundary (Moho)
                points.append([x_coords[i], y_coords[i], depth_km])
                if len(points) >= num_points:
                    break
                    
    return torch.tensor(points[:num_points], dtype=torch.float32, device=device)

# ==============================================================================
# PARTS 6 & 7: TRAINING LOOP & SYSTEM PIPELINE ENGINE
# ==============================================================================
def train_and_evaluate_pinn():
    print("=== Step 1 & 4: Parsing Observation Parameters ===")
    borehole_path = "master_data/master_boreholes_stage5.parquet"
    shapefile_path = "India_Country_Boundary.shp"
    
    df_raw = pd.read_parquet(borehole_path)
    india_border = gpd.read_file(shapefile_path).to_crs(epsg=4326)
    
    target_col = 'heat_flow' if 'heat_flow' in df_raw.columns else 'q'
    df = df_raw.dropna(subset=[target_col]).reset_index(drop=True).copy()
    
    # Generate structural priors
    df['geo_lithology'] = df['geo_lithology'].astype(str).str.lower().str.strip().fillna('missing')
    df['k_prior'] = df['geo_lithology'].map(lambda x: LITHOLOGY_CONDUCTIVITY.get(x, 2.5))
    df['Q_prior'] = df.apply(lambda row: get_radiogenic_heat(row['geo_lithology'], row.get('rock_age_mean_ma', 500.0)), axis=1)
    
    # Process continuous context features
    continuous_features = [
        'elevation', 'crust_thickness_km', 'sediment_thickness_km', 'fault_distance_km',
        'eq_count_50km', 'eq_count_100km', 'mean_mag_50km', 'mean_mag_100km', 'max_mag_100km',
        'rock_age_mean_ma'
    ]
    continuous_features = [c for c in continuous_features if c in df.columns]
    for col in continuous_features:
        df[col] = df[col].fillna(df[col].median())
        
    df['lithology_idx'], lit_categories = pd.factorize(df['geo_lithology'])
    num_unique_lithologies = len(lit_categories)
    
    # Feature scaling for neural net training stability
    scaler_geo = StandardScaler()
    scaled_geo = scaler_geo.fit_transform(df[continuous_features])
    
    # SPATIAL SCALING TRACKING (Prevents the flat purple map anomaly)
    lon_min, lat_min, lon_max, lat_max = india_border.total_bounds
    
    # Uniform min-max bounds scaling to preserve geographic shapes exactly
    def normalize_spatial(coords):
        norm_coords = np.zeros_like(coords)
        norm_coords[:, 0] = 2.0 * (coords[:, 0] - lat_min) / (lat_max - lat_min) - 1.0
        norm_coords[:, 1] = 2.0 * (coords[:, 1] - lon_min) / (lon_max - lon_min) - 1.0
        norm_coords[:, 2] = 2.0 * (coords[:, 2] - 0.0) / (45.0 - 0.0) - 1.0
        return norm_coords

    # Save scaling transformations to apply Chain Rule on automated differentiation gradients
    lat_scale_factor = (lat_max - lat_min) / 2.0
    lon_scale_factor = (lon_max - lon_min) / 2.0
    depth_scale_factor = 45.0 / 2.0

    spatial_coords_raw = np.zeros((len(df), 3))
    spatial_coords_raw[:, 0] = df['lat'].values
    spatial_coords_raw[:, 1] = df['lon'].values
    spatial_coords_raw[:, 2] = 0.0  # Boreholes are surface points
    
    t_spatial = torch.tensor(normalize_spatial(spatial_coords_raw), dtype=torch.float32, device=device)
    t_geo = torch.tensor(scaled_geo, dtype=torch.float32, device=device)
    t_lit_idx = torch.tensor(df['lithology_idx'].values, dtype=torch.long, device=device)
    t_q_obs = torch.tensor(df[target_col].values, dtype=torch.float32, device=device).unsqueeze(1)
    t_k = torch.tensor(df['k_prior'].values, dtype=torch.float32, device=device).unsqueeze(1)
    
    print("Sampling spatial collocation nodes across the lithosphere...")
    colloc_raw = generate_collocation_points(india_border, num_points=15000)
    t_colloc = torch.tensor(normalize_spatial(colloc_raw.cpu().numpy()), dtype=torch.float32, device=device)
    
    # Broadcast context properties across our validation points
    random_indices = np.random.choice(len(df), len(t_colloc))
    t_colloc_geo = t_geo[random_indices]
    t_colloc_lit = t_lit_idx[random_indices]
    t_colloc_k = t_k[random_indices]
    t_colloc_Q = torch.tensor(df['Q_prior'].values[random_indices], dtype=torch.float32, device=device).unsqueeze(1)
    
    pinn = GeothermalPINN(geo_feature_dim=len(continuous_features), num_lithologies=num_unique_lithologies).to(device)
    optimizer = optim.Adam(pinn.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Balanced loss hyper-weights configured to prevent unconstrained smoothing
    lambda_pde = 1.0     
    lambda_surf = 0.5    
    
    print("\n=== Initiating Physics-Informed Optimization Run ===")
    pinn.train()
    for epoch in range(1501):
        optimizer.zero_grad()
        
        # 1. BOREHOLE OBSERVATION LOSS
        t_spatial.requires_grad_(True)
        t_pred = pinn(t_spatial, t_geo, t_lit_idx)
        
        dT_dnorm = torch.autograd.grad(t_pred.sum(), t_spatial, create_graph=True)[0]
        # Chain rule: convert normalized derivatives back to physical kilometers
        dT_dz = dT_dnorm[:, 2].unsqueeze(1) / depth_scale_factor
        
        # Fourier's law computation: q_pred = -k * dT/dz. (Multiplied by 1000 to scale W/m² to mW/m²)
        q_pred = -t_k * (dT_dz / 1000.0) * 1000.0
        loss_obs = nn.MSELoss()(q_pred, t_q_obs)
        
        # 2. SURFACE TEMPERATURE BOUNDARY CONDITION (Fixed to 25.0 °C at z=0)
        loss_surf = nn.MSELoss()(t_pred, torch.full_like(t_pred, 25.0))
        
        # 3. LITHOSPHERIC PDE LOSS (Evaluated across the 3D domain space)
        t_colloc.requires_grad_(True)
        t_colloc_pred = pinn(t_colloc, t_colloc_geo, t_colloc_lit)
        
        grads = torch.autograd.grad(t_colloc_pred.sum(), t_colloc, create_graph=True)[0]
        d2T_dx2 = torch.autograd.grad(grads[:, 0].sum(), t_colloc, create_graph=True)[0][:, 0].unsqueeze(1) / (lat_scale_factor ** 2)
        d2T_dy2 = torch.autograd.grad(grads[:, 1].sum(), t_colloc, create_graph=True)[0][:, 1].unsqueeze(1) / (lon_scale_factor ** 2)
        d2T_dz2 = torch.autograd.grad(grads[:, 2].sum(), t_colloc, create_graph=True)[0][:, 2].unsqueeze(1) / (depth_scale_factor ** 2)
        
        # Steady state crustal heat equation: k * Laplacian(T) + Q = 0
        pde_residual = t_colloc_k * (d2T_dx2 + d2T_dy2 + d2T_dz2) + (t_colloc_Q * 1e-3)
        
        # Rescale the physics residual up by 1e4 so the optimizer can register it
        loss_pde = nn.MSELoss()(pde_residual * 10000.0, torch.zeros_like(pde_residual))
        
        loss_total = loss_obs + (lambda_surf * loss_surf) + (lambda_pde * loss_pde)
        loss_total.backward()
        optimizer.step()
        
        if epoch % 300 == 0:
            print(f"Epoch {epoch:4d} | Total Loss: {loss_total.item():.2f} | Obs Loss: {loss_obs.item():.2f} | PDE Loss: {loss_pde.item():.4f}")

    # ==============================================================================
    # PARTS 8 & 9: RECTANGULAR MAPPING GRID ESTIMATION (PREVENTS COASTLINE HALOS)
    # ==============================================================================
    print("\n=== STEP 8 & 9: Building National Predictive Temperature Mesh ===")
    # Generate continuous dense matrix coordinates (0.05° resolution) over the full bounding box
    lats_arr = np.arange(np.floor(lat_min) - 0.5, np.ceil(lat_max) + 0.5, 0.05)
    lons_arr = np.arange(np.floor(lon_min) - 0.5, np.ceil(lon_max) + 0.5, 0.05)
    lon_m, lat_m = np.meshgrid(lons_arr, lats_arr)
    
    grid_df = pd.DataFrame({'lat': lat_m.flatten(), 'lon': lon_m.flatten()})
    
    grid_df['lithology_idx'] = 0
    for col in continuous_features:
        grid_df[col] = df[col].median()
        
    print("Propagating regional geological parameters via nearest-neighbor lookups...")
    # Map features for the entire rectangle to avoid grid edge discrepancies
    for idx, row in grid_df.iterrows():
        dist = (df['lat'] - row['lat'])**2 + (df['lon'] - row['lon'])**2
        nearest_idx = dist.idxmin()
        grid_df.at[idx, 'lithology_idx'] = df.loc[nearest_idx, 'lithology_idx']
        for col in continuous_features:
            grid_df.at[idx, col] = df.loc[nearest_idx, col]

    map_spatial_raw = np.zeros((len(grid_df), 3))
    map_spatial_raw[:, 0] = grid_df['lat'].values
    map_spatial_raw[:, 1] = grid_df['lon'].values
    map_spatial_raw[:, 2] = 0.1  # Depth target sheet
    
    t_map_spatial = torch.tensor(normalize_spatial(map_spatial_raw), dtype=torch.float32, device=device, requires_grad=True)
    t_map_geo = torch.tensor(scaler_geo.transform(grid_df[continuous_features]), dtype=torch.float32, device=device)
    t_map_lit = torch.tensor(grid_df['lithology_idx'].values, dtype=torch.long, device=device)
    
    pinn.eval()
    t_map_pred = pinn(t_map_spatial, t_map_geo, t_map_lit)
    
    map_grads = torch.autograd.grad(t_map_pred.sum(), t_map_spatial)[0]
    map_dT_dz = map_grads[:, 2].cpu().numpy() / depth_scale_factor
    
    k_map_prior = np.array([df[df['lithology_idx'] == x]['k_prior'].values[0] for x in grid_df['lithology_idx']])
    grid_df['predicted_q'] = -k_map_prior * (map_dT_dz / 1000.0) * 1000.0
    grid_df['predicted_q'] = grid_df['predicted_q'].clip(df[target_col].min() - 5, df[target_col].max() + 5)

    # ==============================================================================
    # PART 10: ATLAS-STYLE PLOT EXPORT & POST-CONTOUR SHAPE MASKING
    # ==============================================================================
    print("=== STEP 10: Generating Publication Graphics ===")
    lon_unique = np.sort(grid_df['lon'].unique())
    lat_unique = np.sort(grid_df['lat'].unique())
    
    pivot_q = grid_df.pivot(index='lat', columns='lon', values='predicted_q').reindex(index=lat_unique, columns=lon_unique)
    
    # Eradicate zig-zag/diamond artifacts by smoothing the complete data sheet
    smooth_q = gaussian_filter(pivot_q.values, sigma=1.2)
    
    fig, ax = plt.subplots(figsize=(11, 12), dpi=600) # Output upgraded to professional 600 DPI
    
    # Continuous atlas layout increments
    bounds = np.arange(15, 130, 5)
    colors_list = ['#053061', '#1a80b8', '#4faec4', '#8ecfa4', '#cbeba0', '#edf8b1', '#fee391', '#fec44f', '#fe9929', '#ec7014', '#cc4c02']
    custom_cmap = plt.cm.colors.LinearSegmentedColormap.from_list("geothermal_atlas", colors_list, N=len(bounds))
    norm = plt.cm.colors.BoundaryNorm(boundaries=bounds, ncolors=custom_cmap.N)
    
    # Render regional background color spectrum
    contour = ax.contourf(lon_unique, lat_unique, smooth_q, levels=bounds, cmap=custom_cmap, norm=norm, extend='both', alpha=0.9)
    
    # Structural Isoline Layer: Label alternate segments to avoid clutter
    contour_levels = np.arange(20, 130, 10)
    iso = ax.contour(lon_unique, lat_unique, smooth_q, levels=contour_levels, colors='black', linewidths=0.6, alpha=0.6)
    ax.clabel(iso, iso.levels[::2], inline=True, fontsize=7.5, fmt='%d', colors='#222222')
    
    # Visual Masking Patch: Cleanly hides colored contour artifacts spilling past the borders
    # 1. Fetch the primary bounding limit geometry path
    india_geom = india_border.geometry.unary_union
    # 2. Create an inverted exterior mask covering the area outside India's borders
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    outer_box = gpd.GeoSeries(gpd.box(lon_min - 2, lat_min - 2, lon_max + 2, lat_max + 2), crs="EPSG:4326").unary_union
    mask_geom = outer_box.difference(india_geom)
    # 3. Paint over the outer boundaries in solid white before adding overlays
    gpd.GeoSeries(mask_geom).plot(ax=ax, facecolor='white', edgecolor='none', zorder=3)
    
    # High-contrast data anchor alignment: Real borehole colors mapped on top
    scatter = ax.scatter(
        df['lon'], df['lat'],
        c=df[target_col],
        cmap=custom_cmap,
        norm=norm,
        edgecolor='black',
        linewidths=0.5,
        s=24,
        zorder=10,
        label=f'Observed Boreholes (n={len(df)})'
    )
    
    # Sharp national baseline shoreline boundary frame
    india_border.plot(ax=ax, facecolor='none', edgecolor='#111111', linewidth=1.4, zorder=11)
    
    ax.set_xlim(lon_min - 0.5, lon_max + 0.5)
    ax.set_ylim(lat_min - 0.5, lat_max + 0.5)
    
    # Academic Figure Presentation Labels
    ax.set_title("Geothermal Heat Flow Map of India Using a Physics-Informed Neural Network", pad=20, weight='bold', fontsize=13)
    ax.set_xlabel("Longitude (°E)", labelpad=10)
    ax.set_ylabel("Latitude (°N)", labelpad=10)
    ax.grid(True, linestyle=':', alpha=0.4, color='gray', zorder=1)
    ax.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='none', framealpha=0.8)
    
    # Modern streamlined colorbar configuration using raw string wrappers
    cbar = fig.colorbar(contour, ax=ax, orientation='vertical', shrink=0.75, pad=0.04, aspect=25)
    cbar.set_label(r"Heat Flow ($\mathrm{mW\ m}^{-2}$)", rotation=270, labelpad=25, weight='bold', fontsize=11)
    cbar.set_ticks(bounds[::2])
    
    os.makedirs("national_outputs", exist_ok=True)
    plt.tight_layout()
    plt.savefig("national_outputs/india_pinn_heat_flow_field_smooth.png", bbox_inches='tight', dpi=600)
    plt.close()
    print("Publication quality map successfully generated at 600 DPI.")

if __name__ == "__main__":
    train_and_evaluate_pinn()
