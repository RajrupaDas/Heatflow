import os
import torch
import pandas as pd
import numpy as np

from data.preprocessing import GeothermalPreprocessor
from data.loader import GeothermalDatasetLoader
from models.pinn import GeothermalPINN
from physics.heat_equation import CrustalHeatEngine

def run_grid_inference(grid_path, model_checkpoint_path, preprocessor_path, output_csv_path, device='cpu'):
    """
    Loads the trained PINN, restores the processing pipeline state, 
    and evaluates the continuous downward heat flow across the entire Indian landmass.
    """
    device = torch.device(device)
    print(f"Initializing Grid Inference Matrix on device: {device}")
    
    # 1. Restore the Serialized Data Ingestion and Pipeline Objects
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"Missing required pipeline artifact: '{preprocessor_path}'")
        
    preprocessor = GeothermalPreprocessor.load(preprocessor_path)
    loader = GeothermalDatasetLoader(device=device.type)
    
    # Load the unlabelled national target grid data
    print(" Ingesting target evaluation grid dataset...")
    raw_grid_df = pd.read_parquet(grid_path)
    
    # 2. Reconstruct Model Architecture and Load Trained Weights
    if not os.path.exists(model_checkpoint_path):
        raise FileNotFoundError(f"Missing required checkpoint model: '{model_checkpoint_path}'")
        
    checkpoint = torch.load(model_checkpoint_path, map_location=device, weights_only=False)
    
    # Extract structural categorical embedding dimensions from the saved configuration
    cat_pipeline = preprocessor.feature_pipeline.named_transformers_['cat']
    encoder = cat_pipeline.named_steps['encoder']
    categorical_cardinalities = [len(cat_list) for cat_list in encoder.categories_]
    
    # Dynamically match layer dimensions used during the training run
    model = GeothermalPINN(
        continuous_dim=len(preprocessor.continuous_features),
        categorical_cardinalities=categorical_cardinalities,
        hidden_dim=256,         # Aligned with training config
        num_residual_blocks=4   # Aligned with training config
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval() # Switch layers to evaluation mode
    
    # 3. Formulate Aligned Evaluation Sub-surface Tensors
    # We evaluate physics parameters slightly below the boundary surface (e.g., z = 0.1 km)
    # to maintain stable derivative evaluations free of clipping constraints.
    print(" Formatting and parsing grid input arrays into tensor matrices...")
    inference_batch = loader.prepare_inference_tensors(raw_grid_df, preprocessor, depth_threshold_km=0.1)
    
    spatial_coords = inference_batch['spatial_coords'].to(device)
    # Assuming spatial_coords currently has columns: 0=lat, 1=lon, 2=depth
    # We want to reorder them to: [depth, lat, lon], which is columns: [2, 0, 1]

    #  ADD THIS FIX LINE:
    spatial_coords = spatial_coords[:, [2, 0, 1]] 

    # Verify the fix with your debug statement:
    print(f"DEBUG (POST-FIX): First 3 rows of spatial_coords:\n{spatial_coords[:3].cpu().numpy()}")
    geo_features = inference_batch['geo_features'].to(device)
    
    # Extract and parse categorical indexes
    cat_cols = preprocessor.categorical_features
    df_cat = raw_grid_df[cat_cols].copy()
    for col in cat_cols:
        df_cat[col] = df_cat[col].astype(str).str.lower().str.strip()
    encoded_cat_array = encoder.transform(df_cat).astype(np.int64)
    categorical_indices = torch.tensor(encoded_cat_array, dtype=torch.long, device=device)
    
    # 4. Initialize the Chain-Rule Derivative Conversion Physics Engine
    lat_scale = (28.88 - 8.22) / 2.0
    lon_scale = (95.55 - 72.18) / 2.0
    depth_scale = 45.0 / 2.0
    heat_engine = CrustalHeatEngine(lat_scale=lat_scale, lon_scale=lon_scale, depth_scale=depth_scale)

    # Add this temporary check right before Step 5:
    print(f"DEBUG: spatial_coords tensor shape: {spatial_coords.shape}")
    print(f"DEBUG: First 3 rows of spatial_coords:\n{spatial_coords[:3].cpu().numpy()}") 

    # 5. Execute Forward Pass and Automatic Differentiation
    print(" Calculating spatial derivative fields and evaluating conductive heat flux...")
    
    # CRITICAL PINN INFERENCE STEP: We enable gradient tracking locally during evaluation
    # because computing Fourier's Heat Flow (q = -k * dT/dz) requires autograd tracking.
    spatial_coords.requires_grad_(True)
    
    with torch.set_grad_enabled(True):
        # Predict the baseline continuous 3D temperature scalar field
        predicted_temperature = model(spatial_coords, geo_features, categorical_indices)
        
        # Pass the temperature field and structural conductivity values to calculate heat flow
        predicted_heat_flow = heat_engine.compute_fourier_heat_flow(
            temperature=predicted_temperature,
            spatial_coords=spatial_coords,
            k_prior=inference_batch['k_prior'].to(device)
        )
        
    # Detach tensors from the processing graph and move arrays back to system memory
    q_out = predicted_heat_flow.detach().cpu().numpy().flatten()

   # 🔍 DIAGNOSTIC BREAKPOINT (Insert right before Step 6)
    print("\n=== 🛠️ PINN INFERENCE GRAPH DIAGNOSTICS ===")
    raw_p_hf = predicted_heat_flow.detach().cpu().numpy().flatten()
    print(f"Raw predicted_heat_flow (from engine) -> Min: {raw_p_hf.min():.6f}, Max: {raw_p_hf.max():.6f}, Mean: {raw_p_hf.mean():.6f}")
    print(f"Calculated q_out (before clipping)   -> Min: {q_out.min():.2f}, Max: {q_out.max():.2f}, Mean: {q_out.mean():.2f}")
    print(f"k_prior (Thermal Conductivity) stats -> Min: {inference_batch['k_prior'].min().item():.2f}, Max: {inference_batch['k_prior'].max().item():.2f}")
    print(f"Predicted Temperature Field stats    -> Min: {predicted_temperature.min().item():.2f}, Max: {predicted_temperature.max().item():.2f}")
    print("===========================================\n")

    # 6. Construct Clean Structural Output Dataframe
    print(" Formatting predictions into final structured output matrix...")
    output_df = pd.DataFrame({
        'lat': raw_grid_df['lat'].values,
        'lon': raw_grid_df['lon'].values,
        'predicted_heat_flow': q_out
    })
    
    # Post-processing quality check: Ensure no extreme non-physical values were predicted
    output_df['predicted_heat_flow'] = output_df['predicted_heat_flow'].clip(lower=10.0, upper=150.0)
    
    # Export clean spatial spreadsheet arrays
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    output_df.to_csv(output_csv_path, index=False)
    print(f"✔️ National continuous heat flow predictions saved to: '{output_csv_path}'")
    print(f"   Processed data footprint: {len(output_df):,} target grid locations.")

if __name__ == "__main__":
    # Resolve project execution pathways
    grid_file = "data/raw/national_feature_grid.parquet"
    best_checkpoint = "models/checkpoints/best_pinn_model.pt"
    pipeline_file = "models/checkpoints/fitted_preprocessor.joblib"
    export_target = "data/processed/national_heat_flow_predictions.csv"
    
    execution_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    run_grid_inference(
        grid_path=grid_file,
        model_checkpoint_path=best_checkpoint,
        preprocessor_path=pipeline_file,
        output_csv_path=export_target,
        device=execution_device
    )
