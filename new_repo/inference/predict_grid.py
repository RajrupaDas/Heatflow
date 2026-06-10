#!/usr/bin/env python3
"""
Geothermal PINN Inference Script for Spatial Grid Prediction across India
"""

import os
import sys
import torch
import pandas as pd
import numpy as np

# Import custom modules 
from data.preprocessing import GeothermalPreprocessor
from models.pinn import GeothermalPINN

def run_inference_pipeline():
    # --- Configuration Paths & Settings ---
    INFERENCE_GRID_PATH = "national_feature_grid.parquet"
    MODEL_DIR = "saved_models"
    OUTPUT_CSV_PATH = "india_predicted_surface_heat_flow.csv"
    
    PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "geothermal_preprocessor.joblib")
    BEST_MODEL_PATH = os.path.join(MODEL_DIR, "pinn_best_model.pt")
    
    BATCH_SIZE = 4096  # Optimized for fast spatial inference
    
    # Automatically allocate execution device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print(f"STAGE 1: Initializing Inference Pipeline on Device: {device}")
    print("=" * 60)

    # --- Step 1: Validate & Load Preprocessor State ---
    if not os.path.exists(PREPROCESSOR_PATH):
        print(f"Error: Fitted preprocessor state not found at '{PREPROCESSOR_PATH}'.")
        print("Please run training ('train.py') first to serialize preprocessing parameters.")
        sys.exit(1)
        
    preprocessor = GeothermalPreprocessor.load(PREPROCESSOR_PATH)
    num_dim = preprocessor.num_numerical_features

    # --- Step 2: Validate & Load Trained Weights Configuration ---
    if not os.path.exists(BEST_MODEL_PATH):
        print(f"Error: Trained model weights file not found at '{BEST_MODEL_PATH}'.")
        print("Please ensure your training run completed successfully and saved the checkpoint.")
        sys.exit(1)
        
    print("Loading optimized model configurations and weights state...")
    checkpoint = torch.load(BEST_MODEL_PATH, map_with_location=device, weights_only=False)
    
    # Deduce embedding dimensions dynamically from preprocessor states to ensure zero mismatch
    cat_transformer = preprocessor.preprocessor.named_transformers_['cat']
    cat_cardinalities = [len(cat_list) for cat_list in cat_transformer.categories_]
    embedding_dims = [max(4, int(torch.ceil(torch.tensor(c / 2.0)).item())) for c in cat_cardinalities]

    # Instantiate model backbone
    model = GeothermalPINN(
        num_numerical_features=num_dim,
        cat_cardinalities=cat_cardinalities,
        embedding_dims=embedding_dims,
        hidden_size=256,  # Matches training configuration parameters
        depth_blocks=4
    )
    
    # Load exact structural parameters
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()  # Disable Dropout layers and Batch Normalization states
    print("Model architecture loaded and validated successfully.")

    # --- Step 3: Load National Inference Grid ---
    print("\n" + "=" * 60)
    print(f"STAGE 2: Extracting Spatial National Feature Grid Matrix")
    print("=" * 60)
    if not os.path.exists(INFERENCE_GRID_PATH):
        print(f"Error: Target inference dataset file '{INFERENCE_GRID_PATH}' not found.")
        sys.exit(1)
        
    df_grid = pd.read_parquet(INFERENCE_GRID_PATH)
    print(f"Loaded {df_grid.shape[0]} target spatial nodes awaiting field transformation.")

    # Keep references to the original non-normalized coordinates for final mapping exports
    raw_lat = df_grid['lat'].to_numpy()
    raw_lon = df_grid['lon'].to_numpy()

    # Apply exact training data transformation matrices to inference grid
    X_processed, _ = preprocessor.transform(df_grid, is_inference=True)

    # --- Step 4: Batch Vectorized Spatial Heat Flow Inference ---
    print("\n" + "=" * 60)
    print("STAGE 3: Executing Vectorized Feed-Forward Prediction Pass")
    print("=" * 60)
    
    total_samples = X_processed.shape[0]
    predictions_accumulator = []

    # Disable autograd gradients completely to save memory footprint overhead during execution
    with torch.no_grad():
        for start_idx in range(0, total_samples, BATCH_SIZE):
            end_idx = min(start_idx + BATCH_SIZE, total_samples)
            
            # Slice and convert batch components to runtime tensors
            X_batch = torch.tensor(X_processed[start_idx:end_idx], dtype=torch.float32, device=device)
            
            # Unpack structural feature spaces matching training input protocol
            x_num = X_batch[:, :num_dim]
            x_cat = X_batch[:, num_dim:]
            
            # Forward prediction pass evaluating Surface Heat Flow (q) directly
            q_pred_batch = model(x_num, x_cat)
            
            # Collect outputs back onto standard CPU RAM storage structures
            predictions_accumulator.append(q_pred_batch.cpu().numpy())

    # Flatten stacked batches array matrices 
    predicted_heat_flow = np.vstack(predictions_accumulator).squeeze()

    # --- Step 5: Construct Mapping DataFrame and Save to Disk ---
    print("\n" + "=" * 60)
    print("STAGE 4: Structuring Geospatial Mapping Array Exports")
    print("=" * 60)
    
    df_output = pd.DataFrame({
        'lat': raw_lat,
        'lon': raw_lon,
        'predicted_heat_flow': predicted_heat_flow
    })

    # Export to production-ready comma-separated value matrix
    df_output.to_csv(OUTPUT_CSV_PATH, index=False)
    
    print(f"Successfully processed spatial mapping traces across India grid framework.")
    print(f"Final inference mapping file outputted directly to: '{OUTPUT_CSV_PATH}'")
    print(f"Prediction statistics summary:\n"
          f" - Minimum: {df_output['predicted_heat_flow'].min():.2f} mW/m²\n"
          f" - Mean   : {df_output['predicted_heat_flow'].mean():.2f} mW/m²\n"
          f" - Maximum: {df_output['predicted_heat_flow'].max():.2f} mW/m²")
    print("=" * 60)

if __name__ == "__main__":
    run_inference_pipeline()
