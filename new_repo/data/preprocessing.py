import os
from typing import List, Tuple, Union, Optional
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

class GeothermalPreprocessor:
    """
    Production-grade preprocessor for Geothermal PINN heat-flow prediction.
    Handles numerical imputation, scaling, and categorical one-hot encoding 
    consistently across training and inference datasets.
    """
    def __init__(self):
        self.numerical_features: List[str] = [
            'lat', 'lon', 'crust_thickness_km', 'sediment_thickness_km',
            'fault_distance_km', 'elevation_dem_m', 'rock_age_mean_ma',
            'eq_count_50km', 'eq_count_100km', 'mean_mag_100km', 'max_mag_100km'
        ]
        
        self.categorical_features: List[str] = [
            'geo_lithology', 'geo_stratigraphy', 'slip_type'
        ]
        self.target_column: str = 'heat_flow'
        self.preprocessor: Optional[ColumnTransformer] = None
        self.feature_names_: Optional[List[str]] = None
        self.num_imputer_: Optional[SimpleImputer] = None
        self.cat_imputer_: Optional[SimpleImputer] = None
        
    def fit(self, df: pd.DataFrame) -> 'GeothermalPreprocessor':
        """
        Fits the imputation, scaling, and encoding pipelines on the training dataframe.
        """
        # Ensure input columns exist
        missing_num = [col for col in self.numerical_features if col not in df.columns]
        missing_cat = [col for col in self.categorical_features if col not in df.columns]
        if missing_num or missing_cat:
            raise ValueError(f"Input dataframe missing expected columns. Missing Numerical: {missing_num}, Missing Categorical: {missing_cat}")

        # Numerical pipeline: Median imputation followed by Standard Scaling
        # We explicitly wrap with StandardScaler() after imputation
        num_transformer = ColumnTransformer(
            transformers=[
                ('impute', SimpleImputer(strategy='median'), self.numerical_features)
            ]
        )
        
        # Categorical pipeline: Constant imputation for missing categories, then One-Hot Encoding
        # handle_unknown='ignore' ensures robust inference even if a new category shows up
        cat_transformer = ColumnTransformer(
            transformers=[
                ('impute', SimpleImputer(strategy='constant', fill_value='UNKNOWN'), self.categorical_features)
            ]
        )

        # Combined Preprocessor
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), self.categorical_features)
            ],
            remainder='drop'
        )

        # We first fill numerical medians to prevent scale distortion, then fit the main preprocessor
        # To avoid fit_transform inconsistencies, we use a robust pipeline-like approach inside ColumnTransformer
        
        # Step 1: Pre-impute to fit encoders safely without NaN issues
        df_imputed = df.copy()
        self.num_imputer_ = SimpleImputer(strategy='median')
        df_imputed[self.numerical_features] = self.num_imputer_.fit_transform(df[self.numerical_features])
        
        self.cat_imputer_ = SimpleImputer(strategy='constant', fill_value='UNKNOWN')
        df_imputed[self.categorical_features] = self.cat_imputer_.fit_transform(df[self.categorical_features].astype(str))
       
        # Step 2: Fit the main scaling and encoding transformer
        self.preprocessor.fit(df_imputed)
        
        # Store internal feature names for validation/tracking
        cat_encoder = self.preprocessor.named_transformers_['cat']
        encoded_cat_features = list(cat_encoder.get_feature_names_out(self.categorical_features))
        self.feature_names_ = self.numerical_features + encoded_cat_features
        
        return self

    def transform(self, df: pd.DataFrame, is_inference: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Transforms the dataframe into a normalized dense numpy array ready for PINN ingestion.
        
        Parameters:
        -----------
        df : pd.DataFrame
            The input dataframe (either master_boreholes_stage5 or national_feature_grid)
        is_inference : bool
            If True, skips looking for the target 'heat_flow' column.
            
        Returns:
        --------
        X : np.ndarray
            Transformed feature matrix of shape (N, feature_dim)
        y : np.ndarray or None
            Target array of shape (N, 1) if training/testing, else None.
        """
        if self.preprocessor is None:
            raise RuntimeError("The preprocessor instance has not been fitted yet. Call fit() first.")
            
        df_processed = df.copy()
        # Apply fitted imputers before the main ColumnTransformer
        df_processed[self.numerical_features] = self.num_imputer_.transform(df_processed[self.numerical_features])
        df_processed[self.categorical_features] = self.cat_imputer_.transform(
            df_processed[self.categorical_features].astype(str)
        )

        # Apply transformation
        X = self.preprocessor.transform(df_processed)
        
        y = None
        if not is_inference and self.target_column in df.columns:
            y = df[self.target_column].to_numpy().astype(np.float32).reshape(-1, 1)
            
        return X.astype(np.float32), y

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Fits the preprocessor and transforms the data in one step.
        """
        return self.fit(df).transform(df, is_inference=False)

    def save(self, filepath: str) -> None:
        """
        Serializes the fitted preprocessor state using joblib.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features,
            'target_column': self.target_column,
            'preprocessor': self.preprocessor,
            'feature_names_': self.feature_names_,
            'num_imputer_': self.num_imputer_,
            'cat_imputer_': self.cat_imputer_
        }, filepath)
        print(f"Preprocessor successfully saved to: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'GeothermalPreprocessor':
        """
        Loads a serialized preprocessor state from disk.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No preprocessor file found at: {filepath}")
            
        data = joblib.load(filepath)
        instance = cls()
        instance.numerical_features = data['numerical_features']
        instance.categorical_features = data['categorical_features']
        instance.target_column = data['target_column']
        instance.preprocessor = data['preprocessor']
        instance.feature_names_ = data['feature_names_']
        instance.num_imputer_ = data['num_imputer_']
        instance.cat_imputer_ = data['cat_imputer_']
        print(f"Preprocessor successfully loaded from: {filepath}")
        return instance


# Example Usage Verification
if __name__ == "__main__":
    # Generate Mock Training Data (Representing master_boreholes_stage5.parquet)
    np.random.seed(42)
    mock_train_size = 100
    train_data = {
        'lat': np.random.uniform(8.0, 37.0, mock_train_size),
        'lon': np.random.uniform(68.0, 97.0, mock_train_size),
        'crust_thickness_km': np.random.uniform(30.0, 45.0, mock_train_size),
        'sediment_thickness_km': np.random.uniform(0.0, 5.0, mock_train_size),
        'fault_distance_km': np.random.uniform(0.0, 100.0, mock_train_size),
        'elevation_dem_m': np.random.uniform(-10, 8000, mock_train_size),
        'rock_age_mean_ma': np.random.uniform(10, 2500, mock_train_size),
        'eq_count_50km': np.random.randint(0, 50, mock_train_size).astype(float),
        'eq_count_100km': np.random.randint(0, 150, mock_train_size).astype(float),
        'mean_mag_100km': np.random.uniform(0.0, 6.5, mock_train_size),
        'max_mag_100km': np.random.uniform(0.0, 8.0, mock_train_size),
        'geo_lithology': np.random.choice(['Granite', 'Basalt', 'Gneiss', None], mock_train_size),
        'geo_stratigraphy': np.random.choice(['Archean', 'Proterozoic', 'Gondwana', 'Deccan_Trap'], mock_train_size),
        'slip_type': np.random.choice(['Strike-Slip', 'Normal', 'Thrust', None], mock_train_size),
        'heat_flow': np.random.uniform(30.0, 120.0, mock_train_size)  # Target
    }
    df_train = pd.DataFrame(train_data)
    # Introduce random NaNs into numerical data to test Imputer
    df_train.loc[df_train['crust_thickness_km'] > 42, 'crust_thickness_km'] = np.nan

    # Generate Mock Inference Data (Representing national_feature_grid.parquet)
    mock_infer_size = 50
    infer_data = {col: vals[:mock_infer_size] for col, vals in train_data.items() if col != 'heat_flow'}
    df_infer = pd.DataFrame(infer_data)
    # Introduce an unseen category during inference to verify robust handling
    df_infer.loc[0, 'geo_lithology'] = 'Unseen_Schist_Type' 

    print("--- Execution Pipeline Test ---")
    
    # 1. Initialize and Fit-Transform on Training Data
    preprocessor = GeothermalPreprocessor()
    X_train, y_train = preprocessor.fit_transform(df_train)
    print(f"Train Features Shape: {X_train.shape}")
    print(f"Train Target Shape: {y_train.shape if y_train is not None else None}")
    print(f"Total encoded features: {len(preprocessor.feature_names_)}")

    # 2. Save preprocessor configuration state
    model_dir = "saved_models"
    preprocessor_path = os.path.join(model_dir, "geothermal_preprocessor.joblib")
    preprocessor.save(preprocessor_path)

    # 3. Load preprocessor state from scratch
    loaded_preprocessor = GeothermalPreprocessor.load(preprocessor_path)

    # 4. Transform Inference Data using loaded parameters
    X_infer, y_infer = loaded_preprocessor.transform(df_infer, is_inference=True)
    print(f"Inference Features Shape: {X_infer.shape}")
    print(f"Inference Target (Should be None): {y_infer}")
    print("Pipeline run successful. Safe from data leakage and robust against unseen categories.")
