import os
import torch
import pandas as pd
import numpy as np

# Baseline crustal properties based on explicit geological lookup profiles
LITHOLOGY_CONDUCTIVITY_MAP = {
    'granite': 3.0, 'gneiss': 2.8, 'basalt': 1.8, 'sandstone': 2.4,
    'shale': 1.5, 'schist': 2.5, 'alluvium': 1.2, 'charnockite': 3.1,
    'quartzite': 3.5, 'limestone': 2.6, 'gabbro': 2.2, 'missing': 2.5
}

class GeothermalDatasetLoader:
    
    #Orchestrates file I/O operations and transforms processed dataframes 
   # into memory-aligned PyTorch computational tensor structures.
    
    def __init__(self, device='cpu'):
        self.device = torch.device(device)

    def load_parquet(self, path):
        """Loads and reads raw tabular data configurations safely."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing targeted file dependency path: '{path}'")
        return pd.read_parquet(path)

    def extract_geophysical_priors(self, df, lithology_col='geo_lithology', conductivity_col='tc_mean'):
        """
        Extracts structural conductivity priors. Uses true measured 'tc_mean' values 
        where available, falling back to lithological properties for missing indices.
        """
        # 1. Generate the fallback array mapped purely from lithology strings
        lit_series = df[lithology_col].astype(str).str.lower().str.strip().fillna('missing')
        fallback_k = lit_series.map(lambda x: LITHOLOGY_CONDUCTIVITY_MAP.get(x, 2.5))
        
        # 2. Extract measured conductivity if present, filling null gaps with the fallback array
        if conductivity_col in df.columns:
            combined_k = df[conductivity_col].fillna(fallback_k).values
        else:
            combined_k = fallback_k.values
            
        return torch.tensor(combined_k, dtype=torch.float32, device=self.device).unsqueeze(1)
   
    def prepare_training_tensors(self, df, preprocessor, target_col='heat_flow'):
        #Transforms raw borehole observations into optimized tensor inputs 
        #to execute the model optimization loop.
    
        # Ensure targeted training labels are valid and fully populated
        df_clean = df.dropna(subset=[target_col]).reset_index(drop=True)
        
        # Isolate baseline spatial components
        spatial_data = np.zeros((len(df_clean), 3))
        spatial_data[:, 0] = df_clean['lat'].values
        spatial_data[:, 1] = df_clean['lon'].values
        spatial_data[:, 2] = 0.0  # Boreholes sit at surface layer
        
        # Transform continuous and spatial elements through the preprocessor
        scaled_spatial = preprocessor.transform_spatial(spatial_data)
        scaled_features = preprocessor.transform_features(df_clean)
        
        # --- FIX: Dynamically extract and index categorical properties ---
        cat_cols = preprocessor.categorical_features
        df_cat = df_clean[cat_cols].copy()
        for col in cat_cols:
            df_cat[col] = df_cat[col].astype(str).str.lower().str.strip()
            
        # Extract transformers from preprocessor
        cat_pipeline = preprocessor.feature_pipeline.named_transformers_['cat']
        encoder = cat_pipeline.named_steps['encoder']
        
        # Vectorized label extraction
        one_hot_encoded = encoder.transform(df_cat)
        encoded_categories = []
        start_idx = 0
        for categories_list in encoder.categories_:
            end_idx = start_idx + len(categories_list)
            feature_slice = one_hot_encoded[:, start_idx:end_idx]
            cat_idx = np.argmax(feature_slice, axis=1)
            encoded_categories.append(cat_idx)
            start_idx = end_idx
            
        encoded_cat_array = np.stack(encoded_categories, axis=1).astype(np.int64)
        # -----------------------------------------------------------------
        
        # Extract geophysical constraints and labels
        k_priors = self.extract_geophysical_priors(df_clean)
        y_true = torch.tensor(df_clean[target_col].values, dtype=torch.float32, device=self.device).unsqueeze(1)
        
        return {
            'spatial_coords': torch.tensor(scaled_spatial, dtype=torch.float32, device=self.device),
            'geo_features': torch.tensor(scaled_features, dtype=torch.float32, device=self.device),
            'categorical_indices': torch.tensor(encoded_cat_array, dtype=torch.long, device=self.device),
            'k_prior': k_priors,
            'targets': y_true
        }
    def prepare_inference_tensors(self, df, preprocessor, depth_threshold_km=0.1):
        #Transforms the nationwide background grid into optimized matrix evaluation structures,
        #evaluating conditions slightly below the surface to extract temperature gradients.
        # Build dense grid matrix positions 
        spatial_data = np.zeros((len(df), 3))
        spatial_data[:, 0] = df['lat'].values
        spatial_data[:, 1] = df['lon'].values
        spatial_data[:, 2] = depth_threshold_km
        
        # Transform through the exact same preprocessor pipeline
        scaled_spatial = preprocessor.transform_spatial(spatial_data)
        scaled_features = preprocessor.transform_features(df)
        k_priors = self.extract_geophysical_priors(df)
        
        return {
            'spatial_coords': torch.tensor(scaled_spatial, dtype=torch.float32, device=self.device),
            'geo_features': torch.tensor(scaled_features, dtype=torch.float32, device=self.device),
            'k_prior': k_priors
        }
