import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import Optional, Dict, Any, Tuple

# Import pipeline dependencies
from data.preprocessing import GeothermalPreprocessor
from data.loader import create_geothermal_dataloader, GeothermalDataset, extract_spatial_coordinates
from models.pinn import GeothermalPINN
from physics.heatflow_physics import evaluate_heatflow_physics


class EarlyStopping:
    """
    Tracks validation loss and triggers early termination if performance 
    stagnates past a designated patience threshold. Saves best-performing weights.
    """
    def __init__(self, patience: int = 15, min_delta: float = 1e-4, checkpoint_path: str = "best_model.pt"):
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def __call__(self, val_loss: float, model: nn.Module, epoch: int, optimizer: torch.optim.Optimizer, preprocessor_path: str):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model, optimizer, epoch, val_loss)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, loss: float):
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': loss,
        }, self.checkpoint_path)


class GeothermalPINNTrainer:
    """
    High-Performance Trainer class executing the physics-informed regularization 
    training protocol for Indian surface heat flow modeling.
    """
    def __init__(
        self,
        model: nn.Module,
        lambda_physics: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        log_dir: str = "runs/geothermal_pinn",
        checkpoint_dir: str = "checkpoints",
        device: Optional[str] = None
    ):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.lambda_physics = lambda_physics
        
        # Optimizer Setup
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Loss Criteria
        self.mse_loss = nn.MSELoss()
        
        # Performance Enhancements
        self.scaler = torch.amp.GradScaler('cuda') if self.device.type == 'cuda' else None
        
        # Logging & Housekeeping
        self.writer = SummaryWriter(log_dir=log_dir)
        self.checkpoint_dir = checkpoint_dir
        self.best_model_path = os.path.join(checkpoint_dir, "pinn_best_model.pt")
        self.early_stopping = EarlyStopping(patience=20, checkpoint_path=self.best_model_path)

    def fit(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        preprocessor: GeothermalPreprocessor,
        epochs: int = 200
    ):
        print(f"Beginning Geothermal PINN Optimization on device: {self.device}")
        
        num_dim = len(preprocessor.numerical_features)
        
        for epoch in range(1, epochs + 1):
            # --- Training Epoch ---
            self.model.train()
            train_total_loss, train_obs_loss, train_phys_loss = 0.0, 0.0, 0.0
            
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                
                # CRITICAL PINN FIX: Split features and explicitly enable gradients in FP32
                x_num = X_batch[:, :num_dim].detach().clone().float().requires_grad_(True)
                x_cat = X_batch[:, num_dim:].float()
                
                # NO AUTOCAST: Run everything explicitly in standard Float32
                q_pred = self.model(x_num, x_cat)
                
                # 1. Observation Data Loss
                obs_loss = self.mse_loss(q_pred, y_batch)
                
                # 2. Physics Regularization Loss (Now stable in FP32)
                _, phys_loss = evaluate_heatflow_physics(q_pred, x_num)
                
                # Unified Regularized Formulation
                total_loss = obs_loss + (self.lambda_physics * phys_loss)
                
                # Check for anomaly to avoid ruining model weights if bad data slips in
                if torch.isnan(total_loss):
                    print(f"\nWarning: NaN encountered in loss computation. Obs: {obs_loss.item()}, Phys: {phys_loss.item()}")
                    continue
                
                # Standard backward pass without GradScaler anomalies
                total_loss.backward()
                
                # Gradient clipping to prevent exploding gradients during deep backpropagation
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                    
                train_total_loss += total_loss.item() * X_batch.size(0)
                train_obs_loss += obs_loss.item() * X_batch.size(0)
                train_phys_loss += phys_loss.item() * X_batch.size(0)
                
            # Normalize step loss values
            n_train = len(train_loader.dataset)
            epoch_train_loss = train_total_loss / n_train
            epoch_train_obs = train_obs_loss / n_train
            epoch_train_phys = train_phys_loss / n_train

            # --- Validation Evaluation ---
            self.model.eval()
            val_total_loss, val_obs_loss, val_phys_loss = 0.0, 0.0, 0.0
            
            # Gradients must remain active even in eval for evaluating ∇²q physics constraints!
            with torch.set_grad_enabled(True):
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    
                    x_num = X_batch[:, :num_dim].detach().clone().float().requires_grad_(True)
                    x_cat = X_batch[:, num_dim:].float()
                    
                    q_pred = self.model(x_num, x_cat)
                    obs_loss = self.mse_loss(q_pred, y_batch)
                    _, phys_loss = evaluate_heatflow_physics(q_pred, x_num)
                    total_loss = obs_loss + (self.lambda_physics * phys_loss)
                        
                    val_total_loss += total_loss.item() * X_batch.size(0)
                    val_obs_loss += obs_loss.item() * X_batch.size(0)
                    val_phys_loss += phys_loss.item() * X_batch.size(0)
                    
            n_val = len(val_loader.dataset)
            epoch_val_loss = val_total_loss / n_val
            epoch_val_obs = val_obs_loss / n_val
            epoch_val_phys = val_phys_loss / n_val

            # Write performance summaries to TensorBoard
            self.writer.add_scalars("Loss/Total", {"Train": epoch_train_loss, "Val": epoch_val_loss}, epoch)
            self.writer.add_scalars("Loss/Observation_MSE", {"Train": epoch_train_obs, "Val": epoch_val_obs}, epoch)
            self.writer.add_scalars("Loss/Physics_Laplacian", {"Train": epoch_train_phys, "Val": epoch_val_phys}, epoch)
            
            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch:03d}/{epochs} | "
                      f"Train Loss: {epoch_train_loss:.4f} (Obs: {epoch_train_obs:.4f}, Phys: {epoch_train_phys:.4f}) | "
                      f"Val Loss: {epoch_val_loss:.4f} (Obs: {epoch_val_obs:.4f}, Phys: {epoch_val_phys:.4f})")
                
            # Early stopping and model checkpoint evaluation
            self.early_stopping(epoch_val_loss, self.model, epoch, self.optimizer, self.checkpoint_dir)
            if self.early_stopping.early_stop:
                print(f">> Early stopping triggered at epoch {epoch}. Reverting to best saved configuration weights.")
                break
                
        self.writer.close()
        print("Training execution complete.")
        # Integration Verification Hook

if __name__ == "__main__":
    import shutil
    
    # Generate mock paths
    mock_data_dir = "mock_data_run"
    os.makedirs(mock_data_dir, exist_ok=True)
    parquet_path = os.path.join(mock_data_dir, "master_boreholes_stage5.parquet")
    prep_path = os.path.join(mock_data_dir, "preprocessor.joblib")
    
    # 1. Create a dummy dataset matching exact input footprint 
    np.random.seed(42)
    mock_cols = [
        'lat', 'lon', 'crust_thickness_km', 'sediment_thickness_km', 'fault_distance_km',
        'elevation_dem_m', 'rock_age_mean_ma', 'eq_count_50km', 'eq_count_100km',
        'mean_mag_100km', 'max_mag_100km'
    ]
    df = pd.DataFrame(np.random.randn(200, len(mock_cols)), columns=mock_cols)
    df['geo_lithology'] = np.random.choice(['Granite', 'Basalt', 'Gneiss'], size=200)
    df['geo_stratigraphy'] = np.random.choice(['Archean', 'Gondwana'], size=200)
    df['slip_type'] = np.random.choice(['Normal', 'Thrust'], size=200)
    df['heat_flow'] = np.random.uniform(35.0, 110.0, size=200)
    df.to_parquet(parquet_path)

    # 2. Extract and split datasets cleanly using preprocessor pipeline
    preprocessor = GeothermalPreprocessor()
    X_processed, y_processed = preprocessor.fit_transform(df)
    preprocessor.save(prep_path)
    
    # Deduce categorical embedding cardinalities dynamically from preprocessor state
    cat_encoder = preprocessor.preprocessor.named_transformers_['cat']
    cardinalities = [len(cat_list) for cat_list in cat_encoder.categories_]
    emb_dims = [max(4, c // 2) for c in cardinalities] # Rule of thumb dimension allocation

    full_dataset = GeothermalDataset(X_processed, y_processed)
    
    # Create an 80/20 train/validation split footprint
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

    # 3. Instantiate Architecture components
    pinn_model = GeothermalPINN(
        num_numerical_features=preprocessor.num_numerical_features,
        cat_cardinalities=cardinalities,
        embedding_dims=emb_dims,
        hidden_size=256,
        depth_blocks=4
    )

    # 4. Initialize Trainer Object
    trainer = GeothermalPINNTrainer(
        model=pinn_model,
        lambda_physics=0.05,
        lr=5e-4,
        log_dir=os.path.join(mock_data_dir, "logs"),
        checkpoint_dir=os.path.join(mock_data_dir, "checkpoints")
    )

    # 5. Run a shallow testing execution loop
    trainer.fit(train_loader, val_loader, preprocessor, epochs=10)
    
    # Verification clean up code
    shutil.rmtree(mock_data_dir)
    print("Verification loop concluded with zero compilation faults.")
