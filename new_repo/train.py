#!/usr/bin/env python3
"""
Geothermal PINN Training Orchestrator for Surface Heat-Flow Prediction across India.
Author: Gemini AI Collaborator (2026)
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import random_split, DataLoader

# Import components from local modules
from data.preprocessing import GeothermalPreprocessor
from data.loader import GeothermalDataset
from models.pinn import GeothermalPINN
from training.trainer import GeothermalPINNTrainer

def run_training_pipeline():
    # --- Configuration Paths & Hyperparameters ---
    DATA_PATH = "master_boreholes_stage5.parquet"
    MODEL_DIR = "saved_models"
    LOG_DIR = "runs/geothermal_pinn"
    
    PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "geothermal_preprocessor.joblib")
    FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "pinn_final_model.pt")
    HISTORY_PATH = os.path.join(MODEL_DIR, "training_history.json")
    
    # PINN structural parameters
    VAL_SPLIT = 0.15
    BATCH_SIZE = 64
    EPOCHS = 200
    HIDDEN_SIZE = 256
    DEPTH_BLOCKS = 4
    LAMBDA_PHYSICS = 0.05  # Smoothness regularization weight
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    print("=" * 60)
    print("STAGE 1: Loading Raw Volumetric & Spatial Parquet Data")
    print("=" * 60)
    if not os.path.exists(DATA_PATH):
        print(f"Error: Target training dataset '{DATA_PATH}' not found.")
        print("Please ensure your parquet file is present in the working execution path.")
        sys.exit(1)
        
    df_raw = pd.read_parquet(DATA_PATH)
    print(f"Loaded {df_raw.shape[0]} borehole data points across {df_raw.shape[1]} raw attributes.")

    print("\n" + "=" * 60)
    print("STAGE 2: Fitting Preprocessing State & Column Transformations")
    print("=" * 60)
    preprocessor = GeothermalPreprocessor()
    X_processed, y_processed = preprocessor.fit_transform(df_raw)
    
    # Serialize the preprocessor state immediately to prevent downstream leakage risk
    preprocessor.save(PREPROCESSOR_PATH)
    
    # Extract structural dimensions dynamically from fitted transformers
    num_numerical_features = len(preprocessor.numerical_features)
    cat_transformer = preprocessor.preprocessor.named_transformers_['cat']
    
    # Deduce embedding dimensional boundaries based on categorical values found on disk
    cat_cardinalities = [len(cat_list) for cat_list in cat_transformer.categories_]
    # Standard rule of thumb mapping for internal categorical representation: min(50, ceil(cardinality/2))
    embedding_dims = [max(4, int(torch.ceil(torch.tensor(c / 2.0)).item())) for c in cat_cardinalities]

    print(f"Numerical Features Processed : {num_numerical_features}")
    print(f"Categorical Features Encoded : {preprocessor.categorical_features}")
    print(f"Deduced Latent Cardinalities : {cat_cardinalities}")
    print(f"Allocated Embedding Spaces   : {embedding_dims}")

    print("\n" + "=" * 60)
    print("STAGE 3: Partitioning Dataset into Train/Validation Splits")
    print("=" * 60)
    full_dataset = GeothermalDataset(X_processed, y_processed)
    
    val_size = int(VAL_SPLIT * len(full_dataset))
    train_size = len(full_dataset) - val_size
    
    # Fixed seed generator ensures clean replication properties across validation batches
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    
    # Pin memory for fast data loading if CUDA is active
    cuda_available = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=cuda_available)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=cuda_available)
    
    print(f"Training Allocation   : {len(train_dataset)} rows ({len(train_loader)} batches)")
    print(f"Validation Allocation : {len(val_dataset)} rows ({len(val_loader)} batches)")

    print("\n" + "=" * 60)
    print("STAGE 4 & 5: Instantiating Residual PINN & Physics Engine Block")
    print("=" * 60)
    total_input_dim = X_processed.shape[1]  # num_numerical + all one-hot columns

    model = GeothermalPINN(
        num_numerical_features=num_numerical_features,
        cat_cardinalities=cat_cardinalities,
        embedding_dims=embedding_dims,
        hidden_size=HIDDEN_SIZE,
        depth_blocks=DEPTH_BLOCKS,
        total_input_dim=total_input_dim
    )    
    print(f"Total Architecture Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("Physics Term Hook Active: Smoothness regularized mean((∇²q)^2) calculated via PyTorch Autograd.")

    print("\n" + "=" * 60)
    print("STAGE 6: Initializing Execution Loop and Regularization Controls")
    print("=" * 60)
    trainer = GeothermalPINNTrainer(
        model=model,
        lambda_physics=LAMBDA_PHYSICS,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        log_dir=LOG_DIR,
        checkpoint_dir=MODEL_DIR
    )
    assert not np.isnan(X_processed).any(), "NaNs in X_processed!"
    assert not np.isnan(y_processed).any(), "NaNs in y_processed!"
    print(f"Data sanity check passed. X shape: {X_processed.shape}")
    # Execute training (Early Stopping monitor active; best model automatically stored)
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        preprocessor=preprocessor,
        epochs=EPOCHS
    )

    print("\n" + "=" * 60)
    print("STAGE 7: Consolidating Final Artifacts and Loss Traces")
    print("=" * 60)
    
    # Export final current epoch model configuration state
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_numerical_features': num_numerical_features,
        'cat_cardinalities': cat_cardinalities,
        'embedding_dims': embedding_dims,
        'hidden_size': HIDDEN_SIZE,
        'depth_blocks': DEPTH_BLOCKS
    }, FINAL_MODEL_PATH)
    print(f"Saved final network weights state: {FINAL_MODEL_PATH}")
    
    # Export metadata runtime tracking summary history logs
    history_metadata = {
        "dataset_source": DATA_PATH,
        "total_samples": len(full_dataset),
        "train_samples": train_size,
        "val_samples": val_size,
        "lambda_physics_regularization": LAMBDA_PHYSICS,
        "network_hyperparameters": {
            "hidden_size": HIDDEN_SIZE,
            "depth_blocks": DEPTH_BLOCKS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE
        },
        "best_val_loss_achieved": trainer.early_stopping.best_loss if trainer.early_stopping.best_loss != float('inf') else None
    }
    
    with open(HISTORY_PATH, "w") as f:
        json.dump(history_metadata, f, indent=4)
    print(f"Saved runtime parameter configurations history logs to: {HISTORY_PATH}")
    print("=" * 60)
    print("Geothermal PINN training pipeline execution completed successfully.")
    print("=" * 60)

if __name__ == "__main__":
    run_training_pipeline()
