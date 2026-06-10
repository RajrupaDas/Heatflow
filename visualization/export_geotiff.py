import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from scipy.interpolate import griddata

def export_predictions_to_geotiff(input_csv_path, output_tiff_path, target_resolution=0.25):
    """
    Transforms tabular geographic point data into a standardized, georeferenced 
    GeoTIFF raster file (EPSG:4326) compatible with professional GIS suites.
    
    Args:
        input_csv_path (str): Path to the generated predictions spreadsheet.
        output_tiff_path (str): Target export path for the final GeoTIFF.
        target_resolution (float): Pixel size step in decimal degrees (e.g., 0.25° grid spacing).
    """
    print(f"🌍 Starting GIS Rasterization Pipeline: {os.path.basename(input_csv_path)}")
    
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Missing required input predictions: '{input_csv_path}'")
        
    # 1. Ingest predicted continuous spatial coordinates
    df = pd.read_csv(input_csv_path)
    
    lons = df['lon'].values
    lats = df['lat'].values
    values = df['predicted_heat_flow'].values
    
    # 2. Establish Structured Coordinate Bounding Matrices
    # We round out slightly to capture border nodes without truncation clipping
    lon_min, lon_max = np.floor(lons.min()), np.ceil(lons.max())
    lat_min, lat_max = np.floor(lats.min()), np.ceil(lats.max())
    
    # Construct a matching coordinate canvas grid array
    grid_lon = np.arange(lon_min, lon_max + target_resolution, target_resolution)
    grid_lat = np.arange(lat_min, lat_max + target_resolution, target_resolution)
    
    # Flip the latitude sequence because GIS rasters read top-to-bottom (North down to South)
    grid_lat = grid_lat[::-1]
    
    mesh_lon, mesh_lat = np.meshgrid(grid_lon, grid_lat)
    
    # 3. Interpolate Points onto the Matrix Grid Space
    # Linear interpolation structures your rows into a clean, smooth map surface array
    print(" 📐 Mapping vector rows onto a continuous geographical matrix cell grid...")
    points = np.stack([lons, lats], axis=1)
    raster_matrix = griddata(points, values, (mesh_lon, mesh_lat), method='linear')
    
    # 4. Handle Missing Regions & Structural Background Boundaries
    # Define a specific standard background value for cells outside the national boundary matrix
    nodata_value = -9999.0
    raster_matrix = np.nan_to_num(raster_matrix, nan=nodata_value)
    
    # Extract structural dimensions
    height, width = raster_matrix.shape
    
    # 5. Build the Affine Spatial Transform System
    # Specifies the absolute grid origin position and exact coordinate pixel size:
    # from_origin(West boundary longitude, North boundary latitude, X pixel step width, Y pixel step height)
    transform = from_origin(lon_min, lat_max, target_resolution, target_resolution)
    # 6. Configure GIS Metadata and Open Write Stream Context
    raster_profile = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'nodata': nodata_value,  # Ensure this stays exactly -9999.0
        'width': width,
        'height': height,
        'count': 1,
        'crs': 'EPSG:4326',
        'transform': transform,
        'compress': 'lzw',
        'tiled': True,
        'blockxsize': 256,
        'blockysize': 256
    }
    
    print(f" 💾 Serializing grid arrays to disk: Width={width}px, Height={height}px")
    os.makedirs(os.path.dirname(output_tiff_path), exist_ok=True)
    
    with rasterio.open(output_tiff_path, 'w', **raster_profile) as dst:
        # Cast the array safely
        output_matrix = raster_matrix.astype(np.float32)
        
        # Write the band array data
        dst.write(output_matrix, 1)
        
        # 🛠️ THE CRITICAL ADDITION: Write an explicit NoData mask layer
        # This forces GIS suites to clip out and transparentize the background ocean space
        dst.write_mask(output_matrix != nodata_value)
           
    print(f"✔️ Production GeoTIFF successfully generated at: '{output_tiff_path}'")

if __name__ == "__main__":
    prediction_source = "data/processed/national_heat_flow_predictions.csv"
    geotiff_target = "reports/geospatial/india_heat_flow_reconstruction.tif"
    
    # Match the resolution of your original national feature grid dataset (~0.25° degree cells)
    export_predictions_to_geotiff(
        input_csv_path=prediction_source,
        output_tiff_path=geotiff_target,
        target_resolution=0.25
    )
