import os
import pandas as pd

def profile_parquet_file(file_path):
    """
    Performs a thorough structural inspection of a parquet dataset 
    for academic and architectural verification.
    """
    print("=" * 80)
    print(f"📁 CORE INSPECTION: {os.path.basename(file_path)}")
    print("=" * 80)
    
    if not os.path.exists(file_path):
        print(f"❌ ERROR: File not found at '{file_path}'. Check your workspace path.")
        return
    
    # Load the Parquet file
    df = pd.read_parquet(file_path)
    
    # 1. Dataset Dimensions
    print(f"🔹 Structural Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print("-" * 50)
    
    # 2. Comprehensive Data Types and Missing Values Matrix
    print(f"{'Column Name':<30} | {'Data Type':<12} | {'Non-Null Count':<15} | {'Missing %':<10}")
    print("-" * 75)
    
    for col in df.columns:
        non_null = df[col].count()
        missing_pct = (1 - (non_null / len(df))) * 100
        print(f"{col:<30} | {str(df[col].dtype):<12} | {non_null:<15,} | {missing_pct:.2f}%")
    
    print("-" * 50)
    
    # 3. Geo-Spatial Context Summaries (if coordinates exist)
    coord_cols = [c for c in df.columns if c.lower() in ['lat', 'latitude', 'lon', 'longitude']]
    if coord_cols:
        print("🔹 Geographic Coverage Limits:")
        for col in coord_cols:
            print(f"   • {col:<10} -> Min: {df[col].min():.4f} | Max: {df[col].max():.4f}")
        print("-" * 50)
        
    # 4. Statistical Profile of Key Numerical Fields
    print("🔹 Statistical Overview of Data Columns:")
    numeric_df = df.select_dtypes(include=['number'])
    if not numeric_df.empty:
        print(numeric_df.describe().T[['mean', 'std', 'min', 'max']])
    else:
        print("   No numeric features identified.")
    print("-" * 50)
    
    # 5. Visual Preview of Head Records
    print("🔹 Snapshot Data Preview (First 3 Rows):")
    display_cols = df.columns[:8]  # Limit horizontal printing space
    print(df[display_cols].head(3).to_string())
    print("\n")

if __name__ == "__main__":
    # Target workspace files
    borehole_file = "master_boreholes_stage5.parquet"
    grid_file = "national_feature_grid.parquet"
    
    # Execute profiles
    profile_parquet_file(borehole_file)
    profile_parquet_file(grid_file)
