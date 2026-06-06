import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

# Set styling for publication-quality plots
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16})

# ==========================================
# 1. LOAD THE PARQUET FILE & STRUCTURAL CHECK
# ==========================================
print("=== Step 1: Loading and Preprocessing Dataset ===")
filepath = "master_boreholes_stage5.parquet"

if not os.path.exists(filepath):
    raise FileNotFoundError(f"Ensure '{filepath}' is in your active directory before running.")

df = pd.read_parquet(filepath)

# Target identification
target_col = 'heat_flow' if 'heat_flow' in df.columns else 'q'
if target_col not in df.columns:
    raise ValueError("Target variable not found in the parquet schema.")

# Exclude leaky and non-numeric metadata
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

# Drop samples with NaN targets
if y.isna().sum() > 0:
    valid_idx = y.dropna().index
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

# Clean entirely empty features dynamically
completely_empty_cols = [col for col in X.columns if X[col].isna().sum() == len(X)]
if completely_empty_cols:
    X = X.drop(columns=completely_empty_cols)
    numeric_features = [c for c in numeric_features if c not in completely_empty_cols]
    categorical_features = [c for c in categorical_features if c not in completely_empty_cols]

# Imputation and Target Encoding
if numeric_features:
    X[numeric_features] = SimpleImputer(strategy='median').fit_transform(X[numeric_features])

if categorical_features:
    X[categorical_features] = X[categorical_features].astype(str)
    X[categorical_features] = SimpleImputer(strategy='constant', fill_value='MISSING').fit_transform(X[categorical_features])
    X[categorical_features] = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1).fit_transform(X[categorical_features])

print(f"Data cleaned. Ready for Cross-Validation on {X.shape[0]} samples with {X.shape[1]} features.\n")

# ==========================================
# 2 & 3. 5-FOLD SHUFFLED CROSS VALIDATION
# ==========================================
print("=== Step 2 & 3: Executing 5-Fold Cross-Validation ===")

kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold_r2_scores = []
fold_rmse_scores = []

# Arrays to store out-of-fold predictions for visualization
oof_predictions = np.zeros(len(X))
oof_actuals = np.zeros(len(X))

# Convert to arrays for clean positional indexing during splits
X_arr = X.values
y_arr = y.values

for fold, (train_idx, test_idx) in enumerate(kf.split(X_arr, y_arr), 1):
    X_train, X_val = X_arr[train_idx], X_arr[test_idx]
    y_train, y_val = y_arr[train_idx], y_arr[test_idx]
    
    # Initialize the base geospatial XGBoost regressor
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    
    # Calculate fold metrics
    fold_r2 = r2_score(y_val, preds)
    fold_rmse = root_mean_squared_error(y_val, preds)
    
    fold_r2_scores.append(fold_r2)
    fold_rmse_scores.append(fold_rmse)
    
    # Save out-of-fold evaluations
    oof_predictions[test_idx] = preds
    oof_actuals[test_idx] = y_val
    
    print(f"Fold {fold} | R² Score: {fold_r2:.4f} | RMSE: {fold_rmse:.4f} mW/m²")

print("-" * 40)
print(f"Mean R² Score : {np.mean(fold_r2_scores):.4f}  (± {np.std(fold_r2_scores):.4f})")
print(f"Mean RMSE     : {np.mean(fold_rmse_scores):.4f} mW/m² (± {np.std(fold_rmse_scores):.4f})")
print("-" * 40 + "\n")
# ==========================================
# 4 & 7. SAVE OUTPUT TABLES AND DIAGNOSTIC PLOTS
# ==========================================
print("=== Step 4 & 7: Generating and Saving Visuals ===")
os.makedirs("cv_output_plots", exist_ok=True)

# Save metrics report as csv text
metrics_df = pd.DataFrame({
    'Fold': [f"Fold {i}" for i in range(1, 6)],
    'R2_Score': fold_r2_scores,
    'RMSE_mW_m2': fold_rmse_scores
})
metrics_df.to_csv("cv_output_plots/cv_fold_metrics.csv", index=False)

# Out-of-Fold cross-validation visualization
plt.figure(figsize=(7, 6))
sns.scatterplot(x=oof_actuals, y=oof_predictions, alpha=0.6, edgecolor='k', color='teal')
perfect_line = [oof_actuals.min(), oof_actuals.max()]
plt.plot(perfect_line, perfect_line, color='red', linestyle='--', linewidth=2, label='Perfect Fit')
plt.title("5-Fold Cross-Validation: Out-of-Fold Predictions")
plt.xlabel("Actual Heat Flow ($q$, mW/m²)")
plt.ylabel("Predicted Heat Flow ($q$, mW/m²)")
plt.legend()
plt.tight_layout()
plt.savefig("cv_output_plots/cv_oof_predicted_vs_actual.png", dpi=300)
plt.close()

# Fold Comparison Plot
plt.figure(figsize=(8, 5))
x_axis = np.arange(1, 6)
plt.bar(x_axis - 0.2, fold_r2_scores, width=0.4, label='R² Score', color='navy', alpha=0.8)
plt.ylabel('R² Scale', color='navy')
plt.tick_params(axis='y', labelcolor='navy')

ax2 = plt.twinx()
# FIX: Changed 'darkdarkorange' to a crisp, valid 'darkorange'
ax2.bar(x_axis + 0.2, fold_rmse_scores, width=0.4, label='RMSE', color='darkorange', alpha=0.8, hatch='//')
ax2.set_ylabel('RMSE (mW/m²)', color='darkorange')
ax2.tick_params(axis='y', labelcolor='darkorange')

plt.title("Stability Across Cross-Validation Folds")
plt.xticks(x_axis, [f"Fold {i}" for i in range(1, 6)])
plt.tight_layout()
plt.savefig("cv_output_plots/cv_fold_stability.png", dpi=300)
plt.close()

print("Cross-Validation pipeline complete. Deliverables exported to './cv_output_plots/'\n")

