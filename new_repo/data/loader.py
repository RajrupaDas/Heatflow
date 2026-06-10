import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Union

# Assuming preprocessing.py is in the same directory/package level.
# If structured as a package, use: from .preprocessing import GeothermalPreprocessor
from data.preprocessing import GeothermalPreprocessor


class GeothermalDataset(Dataset):
    """
    Custom PyTorch Dataset for Geothermal Heat-Flow Prediction.
    Maintains features and target as PyTorch tensors ready for PINN training/inference.
    """
    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


def create_geothermal_dataloader(
    file_path: str,
    preprocessor_path: str,
    is_inference: bool = False,
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 0
) -> Tuple[DataLoader, GeothermalPreprocessor]:
    """
    Loads a parquet file, preprocesses the data using a saved or newly fitted 
    GeothermalPreprocessor, and returns a PyTorch DataLoader along with the preprocessor instance.
    
    Parameters:
    -----------
    file_path : str
        Path to the input parquet file (master_boreholes_stage5 or national_feature_grid).
    preprocessor_path : str
        Path to save/load the preprocessor state (.joblib).
    is_inference : bool
        If True, runs in inference mode (no target variable, no shuffling by default).
    batch_size : int
        Size of batches produced by the DataLoader.
    shuffle : bool
        Whether to shuffle the data (automatically overridden to False if is_inference=True).
    num_workers : int
        Number of subprocesses for data loading.
        
    Returns:
    --------
    dataloader : DataLoader
        PyTorch DataLoader yielding ready-to-use tensors.
    preprocessor : GeothermalPreprocessor
        The fitted preprocessor instance used for the dataset.
    """
    # 1. Read the parquet file
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at: {file_path}")
    
    df = pd.read_parquet(file_path)
    
    # 2. Handle Preprocessor State
    if is_inference:
        # Inference MUST use the exact preprocessor state fitted during training
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(
                f"Preprocessor state not found at {preprocessor_path}. "
                f"You must fit and save the preprocessor during training before running inference."
            )
        preprocessor = GeothermalPreprocessor.load(preprocessor_path)
        X_array, y_array = preprocessor.transform(df, is_inference=True)
        shuffle = False  # Keep spatial sequential continuity intact during inference mapping
    else:
        # Training/Validation mode: Fit or re-load/fit depending on operational pipeline stage
        preprocessor = GeothermalPreprocessor()
        X_array, y_array = preprocessor.fit_transform(df)
        # Save the preprocessor state immediately after fitting
        preprocessor.save(preprocessor_path)

    # 3. Instantiate Dataset and DataLoader
    dataset = GeothermalDataset(X=X_array, y=y_array)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return dataloader, preprocessor


def extract_spatial_coordinates(X_tensor: torch.Tensor, preprocessor: GeothermalPreprocessor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Helper utility for the PINN physics engine.
    Extracts 'lat' and 'lon' tensors from the preprocessed feature tensor and enables gradients.
    This is essential for evaluating PyTorch autograd over ∇²q.
    
    Parameters:
    -----------
    X_tensor : torch.Tensor
        Batch feature tensor from the DataLoader, shape (B, feature_dim)
    preprocessor : GeothermalPreprocessor
        The fitted preprocessor instance holding feature positions.
        
    Returns:
    --------
    lat : torch.Tensor
        Latitude tensor with requires_grad=True, shape (B, 1)
    lon : torch.Tensor
        Longitude tensor with requires_grad=True, shape (B, 1)
    """
    lat_idx = preprocessor.numerical_features.index('lat')
    lon_idx = preprocessor.numerical_features.index('lon')
    
    # Slice columns while ensuring structural shape integrity (B, 1)
    lat = X_tensor[:, lat_idx:lat_idx+1].clone().detach().requires_grad_(True)
    lon = X_tensor[:, lon_idx:lon_idx+1].clone().detach().requires_grad_(True)
    
    return lat, lon


# Example Verification & Integration Test Pipeline
if __name__ == "__main__":
    # Create temporary files to simulate files on disk
    os.makedirs("data_mock", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)
    
    train_file = "master_boreholes_stage5.parquet"
    infer_file = "national_feature_grid.parquet"
    prep_file = "saved_models/geothermal_preprocessor.joblib"
    
    # 1. Generate Mock Parquet Data
    np.random.seed(42)
    mock_cols = [
        'lat', 'lon', 'crust_thickness_km', 'sediment_thickness_km', 'fault_distance_km',
        'elevation_dem_m', 'rock_age_mean_ma', 'eq_count_50km', 'eq_count_100km',
        'mean_mag_100km', 'max_mag_100km'
    ]
    
    # Create Training Parquet
    df_mock_train = pd.DataFrame(np.random.randn(100, len(mock_cols)), columns=mock_cols)
    df_mock_train['geo_lithology'] = np.random.choice(['Granite', 'Basalt'], size=100)
    df_mock_train['geo_stratigraphy'] = np.random.choice(['Archean', 'Gondwana'], size=100)
    df_mock_train['slip_type'] = np.random.choice(['Normal', 'Thrust'], size=100)
    df_mock_train['heat_flow'] = np.random.uniform(40.0, 90.0, size=100)
    df_mock_train.to_parquet(train_file)
    
    # Create Inference Parquet
    df_mock_infer = pd.DataFrame(np.random.randn(50, len(mock_cols)), columns=mock_cols)
    df_mock_infer['geo_lithology'] = np.random.choice(['Granite', 'Basalt'], size=50)
    df_mock_infer['geo_stratigraphy'] = np.random.choice(['Archean', 'Gondwana'], size=50)
    df_mock_infer['slip_type'] = np.random.choice(['Normal', 'Thrust'], size=50)
    df_mock_infer.to_parquet(infer_file)

    print("--- Testing Training DataLoader Creation ---")
    train_loader, train_prep = create_geothermal_dataloader(
        file_path=train_file,
        preprocessor_path=prep_file,
        is_inference=False,
        batch_size=16,
        shuffle=True
    )
    
    # Check a single batch
    for X_batch, y_batch in train_loader:
        print(f"Train Batch X Shape: {X_batch.shape}, Type: {X_batch.dtype}")
        print(f"Train Batch y Shape: {y_batch.shape}, Type: {y_batch.dtype}")
        
        # Test physics coordinate separation
        lat, lon = extract_spatial_coordinates(X_batch, train_prep)
        print(f"Extracted Lat Shape: {lat.shape}, Requires Grad: {lat.requires_grad}")
        print(f"Extracted Lon Shape: {lon.shape}, Requires Grad: {lon.requires_grad}")
        break

    print("\n--- Testing Inference DataLoader Creation ---")
    infer_loader, infer_prep = create_geothermal_dataloader(
        file_path=infer_file,
        preprocessor_path=prep_file,
        is_inference=True,
        batch_size=16
    )
    
    for batch in infer_loader:
        # In inference, only features are returned
        print(f"Inference Batch X Shape: {batch.shape}, Type: {batch.dtype}")
        break

    # Clean up mock directories
    import shutil
    shutil.rmtree("data_mock")
    shutil.rmtree("saved_models")
    print("\nData loading pipeline executed cleanly and validated successfully.")
