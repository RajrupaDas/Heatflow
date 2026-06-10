import os
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

class SpatialDomainScaler(BaseEstimator, TransformerMixin):
    """
    Custom spatial domain transformer to uniformly scale lat/lon/depth coordinates 
    to [-1, 1] based on static country-level bounding box limits. This prevents 
    the spatial distortion caused by independent component scaling.
    """
    def __init__(self, lat_min=8.0, lat_max=37.0, lon_min=68.0, lon_max=97.0, depth_max=45.0):
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.depth_max = depth_max

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Expects DataFrame or Array with columns/indices corresponding to [lat, lon, depth]
        X_arr = np.array(X, dtype=np.float32)
        X_norm = np.zeros_like(X_arr)
        
        # Latitude normalization
        X_norm[:, 0] = 2.0 * (X_arr[:, 0] - self.lat_min) / (self.lat_max - self.lat_min) - 1.0
        # Longitude normalization
        X_norm[:, 1] = 2.0 * (X_arr[:, 1] - self.lon_min) / (self.lon_max - self.lon_min) - 1.0
        # Depth normalization (0 to Moho boundary depth threshold)
        X_norm[:, 2] = 2.0 * (X_arr[:, 2] - 0.0) / (self.depth_max - 0.0) - 1.0
        
        return X_norm


class GeothermalPreprocessor:
    """
    Unified preprocessing pipeline handling robust imputation, encoding, 
    and feature transformation across both point observations and dense national grids.
    """
    def __init__(self, continuous_features, categorical_features, spatial_config=None):
        self.continuous_features = continuous_features
        self.categorical_features = categorical_features
        self.spatial_config = spatial_config or {}
        self.feature_pipeline = None
        self.spatial_scaler = None
        self.is_fitted = False

    def build_pipelines(self):
        # Continuous feature engine: Median imputation handles missing regional values safely
        continuous_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Categorical engine: Explicit string uniform cleaning and unknown value handling
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        # Combined parallel transformer layout
        self.feature_pipeline = ColumnTransformer(
            transformers=[
                ('num', continuous_transformer, self.continuous_features),
                ('cat', categorical_transformer, self.categorical_features)
            ],
            remainder='drop'
        )
        
        self.spatial_scaler = SpatialDomainScaler(**self.spatial_config)

    def fit(self, df):
        """Fits the data preprocessing pipeline using the primary training dataset."""
        if self.feature_pipeline is None:
            self.build_pipelines()
            
        # Standardize structural string columns before processing
        df_clean = df.copy()
        for col in self.categorical_features:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).str.lower().str.strip()

        self.feature_pipeline.fit(df_clean)
        self.is_fitted = True
        return self

    def transform_features(self, df):
        """Applies fitted continuous and categorical transforms systematically."""
        if not self.is_fitted:
            raise ValueError("The preprocessor state has not been fitted yet.")
            
        df_clean = df.copy()
        for col in self.categorical_features:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).str.lower().str.strip()
                
        return self.feature_pipeline.transform(df_clean).astype(np.float32)

    def transform_spatial(self, spatial_df):
        """Normalizes spatial coordinates [lat, lon, depth] into standard numerical ranges."""
        if not self.is_fitted:
            raise ValueError("The preprocessor state has not been fitted yet.")
        return self.spatial_scaler.transform(spatial_df)

    def save(self, output_path):
        """Serializes the fully fitted preprocessor pipeline to storage."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        joblib.dump(self, output_path)
        print(f"✔️ Preprocessor successfully serialized to: '{output_path}'")

    @classmethod
    def load(cls, input_path):
        """Restores a serialized preprocessor object configuration."""
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"No preprocessor artifact found at '{input_path}'")
        preprocessor = joblib.load(input_path)
        print(f" Preprocessor successfully restored from: '{input_path}'")
        return preprocessor
