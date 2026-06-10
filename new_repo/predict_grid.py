#!/usr/bin/env python3
"""
Geothermal PINN Inference Script for Spatial Grid Prediction across India
"""

import os
import sys
import torch
import pandas as pd
import numpy as np

from data.preprocessing import GeothermalPreprocessor
from models.pinn import GeothermalPINN


def run_inference_pipeline():
    # --- Configuration Paths & Settings ---
    INFERENCE_GRID_PATH = "national_feature_grid.parquet"
    MODEL_DIR = "saved_models"
    OUTPUT_CSV_PATH = "india_predicted_surface_heat_flow.csv"

    PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "geothermal_preprocessor.joblib")
    BEST_MODEL_PATH = os.path.join(MODEL_DIR, "pinn_best_model.pt")

    BATCH_SIZE = 4096

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print(f"STAGE 1: Initializing Inference Pipeline on Device: {device}")
    print("=" * 60)

    # --- Step 1: Load Preprocessor ---
    if not os.path.exists(PREPROCESSOR_PATH):
        print(f"Error: Fitted preprocessor state not found at '{PREPROCESSOR_PATH}'.")
        print("Please run train.py first.")
        sys.exit(1)

    loaded_prep = GeothermalPreprocessor.load(PREPROCESSOR_PATH)
    num_dim = len(loaded_prep.numerical_features)

    # --- Step 2: Load Model Weights ---
    if not os.path.exists(BEST_MODEL_PATH):
        print(f"Error: Trained model weights not found at '{BEST_MODEL_PATH}'.")
        sys.exit(1)

    print("Loading optimized model configurations and weights state...")
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=False)

    # Derive total_input_dim from the fitted preprocessor's feature list
    # (num_numerical + all one-hot columns — must match exactly what train.py saw)
    total_input_dim = len(loaded_prep.feature_names_)

    cat_transformer = loaded_prep.preprocessor.named_transformers_['cat']
    cat_cardinalities = [len(cat_list) for cat_list in cat_transformer.categories_]
    embedding_dims = [max(4, int(torch.ceil(torch.tensor(c / 2.0)).item())) for c in cat_cardinalities]

    model = GeothermalPINN(
        num_numerical_features=num_dim,
        cat_cardinalities=cat_cardinalities,
        embedding_dims=embedding_dims,
        hidden_size=256,
        depth_blocks=4,
        total_input_dim=total_input_dim,   # <-- fix 1
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print("Model architecture loaded and validated successfully.")
    print(f"Input dimension : {total_input_dim}  ({num_dim} numerical + {total_input_dim - num_dim} one-hot)")

    # --- Step 3: Load Inference Grid ---
    print("\n" + "=" * 60)
    print("STAGE 2: Extracting Spatial National Feature Grid Matrix")
    print("=" * 60)

    if not os.path.exists(INFERENCE_GRID_PATH):
        print(f"Error: Inference grid file '{INFERENCE_GRID_PATH}' not found.")
        sys.exit(1)

    df_grid = pd.read_parquet(INFERENCE_GRID_PATH)
    print(f"Loaded {df_grid.shape[0]} spatial nodes across {df_grid.shape[1]} attributes.")

    raw_lat = df_grid['lat'].to_numpy()
    raw_lon = df_grid['lon'].to_numpy()

    # <-- fix 2: was `preprocessor`, must be `loaded_prep`
    X_processed, _ = loaded_prep.transform(df_grid, is_inference=True)

    # Sanity check — no NaNs should survive the preprocessor after our earlier fix
    if np.isnan(X_processed).any():
        print("Error: NaNs detected in processed inference features. Check your input grid.")
        sys.exit(1)

    print(f"Feature matrix shape after preprocessing: {X_processed.shape}")

    # --- Step 4: Batched Inference ---
    print("\n" + "=" * 60)
    print("STAGE 3: Executing Vectorized Feed-Forward Prediction Pass")
    print("=" * 60)

    total_samples = X_processed.shape[0]
    predictions_accumulator = []

    with torch.no_grad():
        for start_idx in range(0, total_samples, BATCH_SIZE):
            end_idx = min(start_idx + BATCH_SIZE, total_samples)

            X_batch = torch.tensor(
                X_processed[start_idx:end_idx], dtype=torch.float32, device=device
            )

            x_num = X_batch[:, :num_dim]
            x_cat = X_batch[:, num_dim:].float()

            q_pred_batch = model(x_num, x_cat)
            predictions_accumulator.append(q_pred_batch.cpu().numpy())

            if (start_idx // BATCH_SIZE) % 10 == 0:
                pct = 100.0 * end_idx / total_samples
                print(f"  Processed {end_idx:>7,} / {total_samples:,} nodes  ({pct:.1f}%)")

    predicted_heat_flow = np.vstack(predictions_accumulator).squeeze()

    # --- Step 5: Output ---
    print("\n" + "=" * 60)
    print("STAGE 4: Structuring Geospatial Mapping Array Exports")
    print("=" * 60)

    df_output = pd.DataFrame({
        'lat': raw_lat,
        'lon': raw_lon,
        'predicted_heat_flow_mW_m2': predicted_heat_flow,
    })

    # Basic sanity checks on predictions
    n_nan = df_output['predicted_heat_flow_mW_m2'].isna().sum()
    n_neg = (df_output['predicted_heat_flow_mW_m2'] < 0).sum()
    if n_nan > 0:
        print(f"Warning: {n_nan} NaN predictions in output.")
    if n_neg > 0:
        print(f"Warning: {n_neg} negative heat-flow predictions — check model convergence.")

    print("\n>>> NATIONAL PREDICTED HEAT FLOW FIELD STATISTICS (mW/m²):")
    print(df_output["predicted_heat_flow_mW_m2"].describe().round(3).to_string())
    print("-" * 60)

    df_output.to_csv(OUTPUT_CSV_PATH, index=False)

    print(f"\nMinimum : {df_output['predicted_heat_flow_mW_m2'].min():.2f} mW/m²")
    print(f"Mean    : {df_output['predicted_heat_flow_mW_m2'].mean():.2f} mW/m²")
    print(f"Maximum : {df_output['predicted_heat_flow_mW_m2'].max():.2f} mW/m²")
    print(f"\nOutput saved to: '{OUTPUT_CSV_PATH}'")
    print("=" * 60)


if __name__ == "__main__":
    run_inference_pipeline()
