import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

# Set styling for publication-quality plots
sns.set_theme(style="white")
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16})

# ==========================================
# 1. LOAD AND PREPROCESS DATASET
# ==========================================
print("=== Step 1: Loading & Cleaning Dataset ===")
filepath = "master_boreholes_stage5.parquet"
if not os.path.exists(filepath):
    raise FileNotFoundError(f"Missing file: {filepath}")

df = pd.read_parquet(filepath)
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

# Drop missing targets
if y.isna().sum() > 0:
    valid_idx = y.dropna().index
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

# Drop completely empty columns
empty_cols = [c for c in X.columns if X[c].isna().sum() == len(X)]
if empty_cols:
    X = X.drop(columns=empty_cols)
    numeric_features = [c for c in numeric_features if c not in empty_cols]
    categorical_features = [c for c in categorical_features if c not in empty_cols]

# Impute and Encode
if numeric_features:
    X[numeric_features] = SimpleImputer(strategy='median').fit_transform(X[numeric_features])
if categorical_features:
    X[categorical_features] = X[categorical_features].astype(str)
    X[categorical_features] = SimpleImputer(strategy='constant', fill_value='MISSING').fit_transform(X[categorical_features])
    X[categorical_features] = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1).fit_transform(X[categorical_features])

# ==========================================
# 2. SPATIAL CLUSTERING (K-MEANS)
# ==========================================
print("=== Step 2: Spatial Clustering into Geographic Blocks ===")
# Scaler ensures lat and lon are treated with equal weighting by KMeans distance checks
coords = df.loc[X.index, ['lat', 'lon']].copy()
scaled_coords = StandardScaler().fit_transform(coords)

# Segment into 5 geographic spatial zones
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
X['spatial_block'] = kmeans.fit_predict(scaled_coords)

# Define our Spatial Split: Train on Blocks 0, 1, 2, 3; Completely withhold Block 4
train_blocks = [0, 1, 2, 3]
test_block = [4]

train_idx = X[X['spatial_block'].isin(train_blocks)].index
test_idx = X[X['spatial_block'].isin(test_block)].index

X_train, y_train = X.loc[train_idx].drop(columns=['spatial_block']), y.loc[train_idx]
X_test, y_test = X.loc[test_idx].drop(columns=['spatial_block']), y.loc[test_idx]

print(f"Spatial Validation Configured. Training Samples: {len(X_train)} | Spatial Test Samples: {len(X_test)}")

# ==========================================
# 3 & 4. TRAIN AND PREDICT (SPATIAL VS RANDOM)
# ==========================================
print("=== Step 3 & 4: Training Models and Evaluating Extrapolation ===")

def train_eval_xgb(X_tr, y_tr, X_te, y_te):
    model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    return preds, r2_score(y_te, preds), root_mean_squared_error(y_te, preds), mean_absolute_error(y_te, preds)

# Spatial Block Split Evaluation
spatial_preds, r2_sp, rmse_sp, mae_sp = train_eval_xgb(X_train, y_train, X_test, y_test)

# Comparative Benchmark: Baseline Random Split (using exact same test set fraction size)
test_fraction = len(X_test) / len(X)
from sklearn.model_selection import train_test_split
X_train_rnd, X_test_rnd, y_train_rnd, y_test_rnd = train_test_split(
    X.drop(columns=['spatial_block']), y, test_size=test_fraction, random_state=42
)
_, r2_rnd, rmse_rnd, mae_rnd = train_eval_xgb(X_train_rnd, y_train_rnd, X_test_rnd, y_test_rnd)

# ==========================================
# 5 & 6. REPORT COMPARATIVE METRICS
# ==========================================
print("\n" + "="*50)
print("=== Task 5 & 6: SPATIAL SPLIT VS. RANDOM SPLIT COMPARISON ===")
print("="*50)
print(f"METRIC       |  RANDOM SPLIT (Naïve)  |  SPATIAL BLOCK SPLIT (Rigorous)")
print(f"R² Score     |  {r2_rnd:21.4f}  |  {r2_sp:31.4f}")
print(f"RMSE (mW/m²) |  {rmse_rnd:21.4f}  |  {rmse_sp:31.4f}")
print(f"MAE (mW/m²)  |  {mae_rnd:21.4f}  |  {mae_sp:31.4f}")
print("="*50 + "\n")

# ==========================================
# 8. GENERATE GEOSPATIAL VALIDATION MAPS
# ==========================================
print("=== Task 8: Generating Geospatial Maps ===")
os.makedirs("spatial_outputs", exist_ok=True)

# Build a plotting DataFrame matching original indices
map_df = coords.copy()
map_df['Split_Assignment'] = 'Training Region'
map_df.loc[test_idx, 'Split_Assignment'] = 'Withheld Test Region'
map_df['Error'] = 0.0
map_df.loc[test_idx, 'Error'] = np.abs(y_test - spatial_preds)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Map 1: Train vs Test Geographic Clustering Realities
sns.scatterplot(
    data=map_df, x='lon', y='lat', hue='Split_Assignment', 
    palette={'Training Region': '#4C72B0', 'Withheld Test Region': '#C44E52'}, 
    alpha=0.8, ax=axes[0], edgecolor='k', s=60
)
axes[0].set_title("Spatial Block Split Configuration", fontsize=14, fontweight='bold')
axes[0].set_xlabel("Longitude")
axes[0].set_ylabel("Latitude")
axes[0].legend(frameon=True, loc='upper left')

# Map 2: Spatial Prediction Absolute Errors across the withheld region
scatter_err = axes[1].scatter(
    map_df.loc[test_idx, 'lon'], map_df.loc[test_idx, 'lat'], 
    c=map_df.loc[test_idx, 'Error'], cmap='YlOrRd', 
    edgecolor='k', alpha=0.9, s=70
)
cbar = fig.colorbar(scatter_err, ax=axes[1])
cbar.set_label("Absolute Error |Actual - Predicted| (mW/m²)", rotation=270, labelpad=20)
axes[1].set_title("Extrapolation Errors in Withheld Region", fontsize=14, fontweight='bold')
axes[1].set_xlabel("Longitude")
axes[1].set_ylabel("Latitude")

# Clean layouts and archive figures
plt.tight_layout()
plt.savefig("spatial_outputs/geospatial_cross_validation_profile.png", dpi=300)
plt.close()

print("Geospatial validation map saved safely to './spatial_outputs/geospatial_cross_validation_profile.png'\n")
