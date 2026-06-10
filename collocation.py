import numpy as np
import pandas as pd
import geopandas as gpd
import torch
from scipy.spatial import cKDTree
from shapely.geometry import Point

class NationalCollocationSampler:
    """
    Vectorized high-performance spatial sampler for geothermal PINN collocation nodes.
    
    Ensures that randomly sampled 3D coordinates (lat, lon, depth) within India's
    national borders copy real, localized geophysical and categorical features from 
    the nearest cells in the national feature grid.
    """
    def __init__(self, national_grid_path, shapefile_path, device='cpu'):
        """
        Args:
            national_grid_path (str): Path to 'national_feature_grid.parquet'.
            shapefile_path (str): Path to the country boundary shapefile (India_Country_Boundary.shp).
            device (str): Targeted hardware accelerator device context ('cpu' or 'cuda').
        """
        self.device = torch.device(device)
        
        # Load the real national grid features
        print(" Loading national feature grid matrix...")
        self.grid_df = pd.read_parquet(national_grid_path)
        
        # Build a 2D spatial search index (KD-Tree) over grid coordinates for fast spatial lookups
        self.grid_coords = self.grid_df[['lat', 'lon']].values
        self.spatial_tree = cKDTree(self.grid_coords)
        
        # Load and validate the national geographic boundary vector layer
        print(" Ingesting spatial country boundary vector...")
        self.boundary_gdf = gpd.read_file(shapefile_path).to_crs(epsg=4326)
        self.boundary_polygon = self.boundary_gdf.geometry.unary_union
        
        # Pre-calculate geographic bounding limits to optimize the sampling loop
        self.lon_min, self.lat_min, self.lon_max, self.lat_max = self.boundary_gdf.total_bounds

    def sample_nodes(self, num_points=20000, max_depth_km=45.0):
        """
        Samples thousands of valid 3D collocation nodes within the country boundary 
        and maps them to real geological parameters without iterative loops.
        
        Args:
            num_points (int): Total target size for the collocation batch size array.
            max_depth_km (float): Deep lithospheric depth threshold boundary (e.g., Moho boundary at 45km).
            
        Returns:
            dict: Dictionary bundling the aligned structural tensors ready for the physics engine:
                - 'spatial_coords': torch.Tensor [num_points, 3] as [lat, lon, depth]
                - 'geo_features': torch.Tensor [num_points, continuous_dim]
                - 'categorical_indices': torch.Tensor [num_points, categorical_dim]
        """
        print(f" Generating {num_points:,} spatially bound collocation nodes...")
        valid_lats = []
        valid_lons = []
        
        # Vectorized rejection sampling loop for geographic boundary verification
        while len(valid_lats) < num_points:
            points_needed = num_points - len(valid_lats)
            # Oversample to account for points falling outside irregular borders
            oversample_factor = max(int(points_needed * 2.5), 1000)
            
            rand_lons = np.random.uniform(self.lon_min, self.lon_max, oversample_factor)
            rand_lats = np.random.uniform(self.lat_min, self.lat_max, oversample_factor)
            
            # Efficient vectorized spatial intersection check using GeoPandas
            sampled_gdf = gpd.GeoDataFrame(
                geometry=gpd.points_from_xy(rand_lons, rand_lats), 
                crs="EPSG:4326"
            )
            inside_mask = sampled_gdf.geometry.within(self.boundary_polygon)
            
            intersected_lats = rand_lats[inside_mask]
            intersected_lons = rand_lons[inside_mask]
            
            valid_lats.extend(intersected_lats)
            valid_lons.extend(intersected_lons)
            
        # Isolate exactly the requested number of validated points
        colloc_lats = np.array(valid_lats[:num_points], dtype=np.float32)
        colloc_lons = np.array(valid_lons[:num_points], dtype=np.float32)
        
        # Sample deep crustal vertical values down to the lithosphere boundary (Moho)
        colloc_depths = np.random.uniform(0.0, max_depth_km, num_points).astype(np.float32)
        
        # Combine into an unnormalized physical coordinate tracking block
        raw_spatial = np.stack([colloc_lats, colloc_lons, colloc_depths], axis=1)
        
        # Step 2: Query the KD-Tree in parallel to find the nearest national grid cell for all points
        print(" Mapping collocation nodes to national feature grid parameters via vectorized KD-Tree...")
        query_coords = np.stack([colloc_lats, colloc_lons], axis=1)
        _, nearest_grid_indices = self.spatial_tree.query(query_coords, workers=-1)
        
        # Pull matching continuous and categorical rows using the index map
        aligned_grid_data = self.grid_df.iloc[nearest_grid_indices].reset_index(drop=True)
        
        return {
            'raw_spatial_coords': raw_spatial,
            'matched_grid_df': aligned_grid_data
        }
    def prepare_collocation_tensors(self, num_points=20000, max_depth_km=45.0, preprocessor=None):
        """
        Samples valid nodes, processes their matched attributes through your 
        preprocessing pipeline, and transfers the arrays to the GPU/CPU.
        
        Args:
            num_points (int): Total target collocation batch array count.
            max_depth_km (float): Moho boundary depth calculation threshold context.
            preprocessor (GeothermalPreprocessor): Fitted preprocessor state object.
            
        Returns:
            dict: Tensor dictionary ready for forward-pass execution.
        """
        # Generate the points and retrieve their matching real-world geology rows
        samples = self.sample_nodes(num_points=num_points, max_depth_km=max_depth_km)
        raw_spatial = samples['raw_spatial_coords']
        matched_df = samples['matched_grid_df']
        
        if preprocessor is None:
            raise ValueError("A fitted GeothermalPreprocessor instance must be provided.")
            
        # Transform the continuous attributes using your shared preprocessing pipeline
        print(" Transforming dataset components into scaled tensor formats...")
        scaled_spatial = preprocessor.transform_spatial(raw_spatial)
        scaled_features = preprocessor.transform_features(matched_df)
        
        # Extract encoded categorical attributes generated by the preprocessor
        cat_cols = preprocessor.categorical_features
        df_cat = matched_df[cat_cols].copy()
        for col in cat_cols:
            df_cat[col] = df_cat[col].astype(str).str.lower().str.strip()
            
        # Extract individual transformer definitions from the scikit-learn pipeline steps
        cat_pipeline = preprocessor.feature_pipeline.named_transformers_['cat']
        encoder = cat_pipeline.named_steps['encoder']
        
        # 1. Transform all columns at once into a wide One-Hot Matrix
        one_hot_encoded = encoder.transform(df_cat)
        
        # 2. Slice the wide matrix back into features using argmax to extract token IDs
        encoded_categories = []
        start_idx = 0
        for categories_list in encoder.categories_:
            end_idx = start_idx + len(categories_list)
            feature_slice = one_hot_encoded[:, start_idx:end_idx]
            cat_idx = np.argmax(feature_slice, axis=1)
            encoded_categories.append(cat_idx)
            start_idx = end_idx
            
        # 3. Stack features into a 2D matrix of integer indices [num_points, categorical_dim]
        encoded_cat_array = np.stack(encoded_categories, axis=1).astype(np.int64)
        # --- FIX: Implement Blueprint Prior Mapping via Geo-Lithology Parsing ---
        k_values = []
        q_values = []
        for lith in matched_df['geo_lithology'].astype(str).str.lower().str.strip():
            # Check for dominant lithological keys and map to physical properties (k: W/m·K, q: μW/m³)
            if 'granite' in lith or 'gneiss' in lith or 'gnesis' in lith:
                k_values.append(3.1)  # High conductive basement rock
                q_values.append(2.50) # High radiogenic heat production
            elif 'basalt' in lith or 'deccan' in lith:
                k_values.append(2.1)  # Intermediate volcanic trap rock
                q_values.append(0.45) # Low volcanic radiogenic signature
            elif 'sandstone' in lith or 'sediment' in lith:
                k_values.append(2.4)  # Sedimentary blanket baseline
                q_values.append(1.25) # Moderate sedimentary tracking
            elif 'schist' in lith or 'quartzite' in lith:
                k_values.append(3.5)  # Metamorphic highly conductive zones
                q_values.append(1.50) # Moderate metamorphic tracking
            else:
                k_values.append(2.5)  # Global upper crustal rock baseline fallback
                q_values.append(1.00) # Standard crustal crust signature baseline fallback
                
        k_values = np.array(k_values, dtype=np.float32)
        q_values = np.array(q_values, dtype=np.float32)
        # ------------------------------------------------------------------------- 
        return {
            'spatial_coords': torch.tensor(scaled_spatial, dtype=torch.float32, device=self.device),
            'geo_features': torch.tensor(scaled_features, dtype=torch.float32, device=self.device),
            'categorical_indices': torch.tensor(encoded_cat_array, dtype=torch.long, device=self.device),
            'k_prior': torch.tensor(k_values, dtype=torch.float32, device=self.device).unsqueeze(1),
            'q_prior': torch.tensor(q_values, dtype=torch.float32, device=self.device).unsqueeze(1)
        }
