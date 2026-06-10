import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data.preprocessing import GeothermalPreprocessor
from data.loader import GeothermalDatasetLoader
from models.pinn import GeothermalPINN
from physics.heat_equation import CrustalHeatEngine
from collocation import NationalCollocationSampler
from training.trainer import GeothermalPINNTrainer

# Define execution parameters and loss balancing weights
config = {
    'epochs': 2500,
    'patience': 150,
    'collocation_nodes': 25000,
    'checkpoint_dir': 'models/checkpoints',
    'log_dir': 'runs/india_geothermal_experiment',
    'lambda_obs': 1.0,         # High weight to accurately match borehole data
    'lambda_bc': 0.5,          # Mid weight to enforce surface boundary consistency
    'lambda_pde': 0.1,        # Balanced weight for deep crustal PDE constraints
    'surface_temp_bc': 25.0,   # Fixed surface intercept (°C)
    'hidden_dim': 256,
    'residual_blocks': 4
}

def run_pipeline():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loader = GeothermalDatasetLoader(device=device)
    
    # 1. Ingest Master Borehole Observations File
    print(" Reading raw dataset arrays...")
    raw_boreholes = loader.load_parquet("data/raw/master_boreholes_stage5.parquet")
    
    # Define exact continuous and categorical feature matrices
    continuous_features = [
        'elevation', 'crust_thickness_km', 'sediment_thickness_km', 'fault_distance_km',
        'eq_count_50km', 'eq_count_100km', 'mean_mag_50km', 'mean_mag_100km', 'max_mag_100km',
        'rock_age_mean_ma'
    ]
    categorical_features = ['geo_lithology', 'geo_stratigraphy', 'slip_type']
    
    # 2. Construct Clean Train/Validation Splits (80/20 Spatial Sample Separation)
    train_df, val_df = train_test_split(raw_boreholes, test_size=0.2, random_state=42)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    
    # 3. Initialize and Fit the Processing Pipelines
    preprocessor = GeothermalPreprocessor(continuous_features, categorical_features)
    preprocessor.fit(train_df)
    preprocessor.save("models/checkpoints/fitted_preprocessor.joblib")
    
    # Build operational tensors
    train_batch = loader.prepare_training_tensors(train_df, preprocessor)
    val_batch = loader.prepare_training_tensors(val_df, preprocessor)
    
    # 4. Initialize the National Spatial Collocation Sampler Engine
    collocation_sampler = NationalCollocationSampler(
        national_grid_path="data/raw/national_feature_grid.parquet",
        shapefile_path="data/shapefiles/India_Country_Boundary/India_Country_Boundary.shp",
        device=device
    )
    
    # Define physical background heat production rates (Q_prior) across regions (e.g., 1.5 μW/m³)
    q_prior_colloc_val = np.random.uniform(1.0, 2.0, config['collocation_nodes']).astype(np.float32)
   
    raw_continuous_dim = train_batch['geo_features'].shape[1]
    
    # 5. Determine Embeddings Vocab Cardinality
    cat_pipeline = preprocessor.feature_pipeline.named_transformers_['cat']
    encoder = cat_pipeline.named_steps['encoder']
    categorical_cardinalities = [len(cat_list) for cat_list in encoder.categories_]
    
    # 6. Initialize Model Components
    model = GeothermalPINN(
        continuous_dim=len(continuous_features),
        categorical_cardinalities=categorical_cardinalities,
        hidden_dim=config['hidden_dim'],
        num_residual_blocks=config['residual_blocks']
    )
    
    # Calculate scale parameters from domain ranges to adjust derivatives properly via the Chain Rule
    # Maps spatial constraints between [-1, 1] ranges across lat/lon limits
    lat_scale = (28.88 - 8.22) / 2.0
    lon_scale = (95.55 - 72.18) / 2.0
    depth_scale = 45.0 / 2.0 # Total target thickness tracking block depth scale parameter (km)
    
    heat_engine = CrustalHeatEngine(lat_scale=lat_scale, lon_scale=lon_scale, depth_scale=depth_scale)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    
    # 7. Initialize Trainer and Begin Optimization Loop
    trainer = GeothermalPINNTrainer(
        model=model,
        heat_engine=heat_engine,
        optimizer=optimizer,
        config=config,
        device=device
    )
    
    trainer.fit(
        train_batch=train_batch,
        val_batch=val_batch,
        collocation_sampler=collocation_sampler,
        preprocessor=preprocessor,
        q_prior_colloc_val=q_prior_colloc_val,
        config=config
    )
    print("🏆 PINN Training cycle successfully completed.")

if __name__ == "__main__":
    run_pipeline()
