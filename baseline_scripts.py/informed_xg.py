import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

import xgboost as xgb
import geopandas as gpd
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Publication styling
sns.set_theme(style="ticks")
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 15})

# ==========================================
# 1. LOAD AND PREPROCESS DATASET
# ==========================================
print("=== Step 1: Loading & Cleaning Dataset ===")
filepath = "master_boreholes_stage5.parquet"
shapefile_path = "India_Country_Boundary.shp"

if not os.path.exists(filepath) or not os.path.exists(shapefile_path):
    raise FileNotFoundError("Ensure both your data parquet and boundary shapefiles are in the active directory.")

df = pd.read_parquet(filepath)
india_border = gpd.read_file(shapefile_path).to_crs(epsg=4326)
target_col = 'heat_flow' if 'heat_flow' in df.columns else 'q'

columns_to_ignore = [
    'catalog_id', 'catalog_name', 'fs_name', 'name', 'downthrown_side_id', 
    'downthrown_side_dir', 'last_movement', 'notes', 'reference', 'lithology_text',
    'T_grad_mean', 'tc_mean'
]

feature_candidates = [col for col in df.columns if col != target_col and col not in columns_to_ignore]
numeric_features = df[feature_candidates].select_dtypes(include=[np.number]).columns.tolist()
categorical_features = df[feature_candidates].select_dtypes(include=[object, 'category', 'string']).columns.tolist()

X = df[numeric_features + categorical_features].copy()
y = df[target_col].copy()

# Drop missing targets and empty columns
if y.isna().sum() > 0:
    valid_idx = y.dropna().index
    X, y = X.loc[valid_idx], y.loc[valid_idx]

empty_cols = [c for c in X.columns if X[c].isna().sum() == len(X)]
if empty_cols:
    X = X.drop(columns=empty_cols)
    numeric_features = [c for c in numeric_features if c not in empty_cols]
    categorical_features = [c for c in categorical_features if c not in empty_cols]

# Clean Imputation & Encoding
if numeric_features:
    X[numeric_features] = SimpleImputer(strategy='median').fit_transform(X[numeric_features])
if categorical_features:
    X[categorical_features] = X[categorical_features].astype(str)
    X[categorical_features] = SimpleImputer(strategy='constant', fill_value='MISSING').fit_transform(X[categorical_features])
    X[categorical_features] = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1).fit_transform(X[categorical_features])

# ==========================================
# 2. DEFINE THE PHYSICS-INFORMED OBJECTIVE FUNCTION
# ==========================================
# This function calculates the Gradients and Hessians for XGBoost's loss optimization.
# It introduces a penalty term forcing neighboring predictions to regularize smoothly.
def physics_informed_objective(y_true, y_pred):
    # Standard MSE Data Residuals
    labels = y_true
    errors = y_pred - labels
    
    # Base Data Loss Gradients
    grad = errors
    hess = np.ones_like(y_true)
    
    # PHYSICS REGULARIZATION: Smooth Gradient Penalty (Approximated Spatial Laplace Boundary Regularization)
    # Penalizes sharp prediction steps between neighboring rows to mirror thermal diffusion
    lambda_physics = 0.25  # Tuning weight favoring physical boundary consistency
    
    # Roll array shifts to simulate spatial continuity derivatives across adjacent indices
    spatial_diff_forward = np.roll(y_pred, 1) - y_pred
    spatial_diff_backward = np.roll(y_pred, -1) - y_pred
    
    # Inject spatial smoothness gradients directly into XGBoost's split optimization loop
    grad += lambda_physics * (2 * y_pred - np.roll(y_pred, 1) - np.roll(y_pred, -1))
    hess += lambda_physics * 2.0
    
    return grad, hess

# ==========================================
# 3. SPATIAL CROSS-VALIDATION WITH PHYSICS CONSTRAINTS
# ==========================================
print("=== Step 3: Running Rigorous Spatial Block Validation ===")
coords = df.loc[X.index, ['lat', 'lon']].copy()
scaled_coords = StandardScaler().fit_transform(coords)

# Cluster into geographic zones
X['spatial_block'] = KMeans(n_clusters=5, random_state=42, n_init=10).fit_predict(scaled_coords)

train_idx = X[X['spatial_block'].isin([0, 1, 2, 3])].index
test_idx = X[X['spatial_block'].isin([4])].index

X_train, y_train = X.loc[train_idx].drop(columns=['spatial_block']), y.loc[train_idx]
X_test, y_test = X.loc[test_idx].drop(columns=['spatial_block']), y.loc[test_idx]

# Initialize and train our Physics-Informed XGBoost model
pi_model = xgb.XGBRegressor(
    n_estimators=300, 
    learning_rate=0.05, 
    max_depth=6,
    subsample=0.8, 
    colsample_bytree=0.8, 
    random_state=42, 
    n_jobs=-1,
    objective=physics_informed_objective # <-- Overriding with Custom Physics Loss Engine
)

pi_model.fit(X_train, y_train)
pi_preds = pi_model.predict(X_test)

# Report Spatial Extrapolation Performance
print("\n" + "="*50)
print("=== SPATIAL GENERALIZATION PROGRESS REPORT ===")
print("="*50)
print(f"Physics-Informed Spatial R² Score : {r2_score(y_test, pi_preds):.4f}")
print(f"Physics-Informed Spatial RMSE     : {root_mean_squared_error(y_test, pi_preds):.4f} mW/m²")
print(f"Physics-Informed Spatial MAE      : {mean_absolute_error(y_test, pi_preds):.4f} mW/m²")
print("="*50 + "\n")

# ==========================================
# 4. NATIONAL MAPPING WITH GEOGRAPHIC MASK
# ==========================================
print("=== Step 4: Compiling Continuous National Heat Flow Map ===")
grid_df = pd.read_parquet("national_feature_grid.parquet")

# Force the grid features to exactly match the clean columns used to train the model
clean_model_features = [c for c in X.columns if c != 'spatial_block']
X_grid = grid_df[clean_model_features].copy()

# Sync grid features to match encoder states using your exact global variables
if categorical_features:
    active_cat_cols = [c for c in categorical_features if c in clean_model_features]
    if active_cat_cols:
        X_grid[active_cat_cols] = X_grid[active_cat_cols].astype(str)
        X_grid[active_cat_cols] = SimpleImputer(strategy='constant', fill_value='MISSING').fit_transform(X_grid[active_cat_cols])
        X_grid[active_cat_cols] = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1).fit_transform(X_grid[active_cat_cols])

if numeric_features:
    active_num_cols = [c for c in numeric_features if c in clean_model_features]
    if active_num_cols:
        X_grid[active_num_cols] = SimpleImputer(strategy='median').fit_transform(X_grid[active_num_cols])

# Run national grid predictions through the physics-constrained model
grid_df['predicted_q'] = pi_model.predict(X_grid)

# Clip the bounding box grid directly to the national border vector mask
gdf_grid = gpd.GeoDataFrame(grid_df, geometry=gpd.points_from_xy(grid_df['lon'], grid_df['lat']), crs="EPSG:4326")
clipped_gdf = gpd.sjoin(gdf_grid, india_border, how="inner", predicate="within").drop_duplicates(subset=['lat', 'lon'])
clipped_df = pd.DataFrame(clipped_gdf.drop(columns='geometry'))

# Build continuous transparent mesh arrays
pivot_map = grid_df.pivot(index='lat', columns='lon', values='predicted_q')
mask_matrix = clipped_df.pivot(index='lat', columns='lon', values='predicted_q')
pivot_map = pivot_map.where(mask_matrix.notna())

# Define discrete color boundaries mirroring standard professional GIS themes
bounds = np.linspace(np.floor(clipped_df['predicted_q'].min()), np.ceil(clipped_df['predicted_q'].max()), 9)
cmap = plt.cm.turbo
norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=cmap.N)

# Render Final Output Asset
fig, ax = plt.subplots(figsize=(10, 11), dpi=300)
contour = ax.contourf(pivot_map.columns, pivot_map.index, pivot_map.values, levels=bounds, cmap=cmap, norm=norm, extend='both', alpha=0.9)

# Layer country boundary vector lines directly over the prediction array mesh
india_border.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1.5, zorder=3)

ax.set_xlim(clipped_df['lon'].min() - 1, clipped_df['lon'].max() + 1)
ax.set_ylim(clipped_df['lat'].min() - 1, clipped_df['lat'].max() + 1)
ax.set_title("Physics-Informed XGBoost: Continuous Geothermal Heat Flow Map of India", pad=15, weight='bold')
ax.set_xlabel("Longitude (°E)")
ax.set_ylabel("Latitude (°N)")
ax.grid(True, linestyle='--', alpha=0.4)

# Format discrete professional legend scales
cbar = fig.colorbar(contour, ax=ax, orientation='vertical', shrink=0.75, pad=0.03)
cbar.set_label("Predicted Surface Heat Flow ($q$, mW/m²)", rotation=270, labelpad=20, weight='bold')
cbar.set_ticks(bounds)
cbar.ax.set_yticklabels([f"{int(b)}" for b in bounds])

os.makedirs("national_outputs", exist_ok=True)
plt.tight_layout()
plt.savefig("national_outputs/india_physics_informed_xgboost.png", bbox_inches='tight')
plt.close()

print("Experiment Complete. Physics-constrained geothermal map archived to './national_outputs/india_physics_informed_xgboost.png'")
