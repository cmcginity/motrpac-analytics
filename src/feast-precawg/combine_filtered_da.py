import os
import pandas as pd
import glob

# Project root
project_root = '/Users/curtismcginity/stanford/research/proj/MoTrPAC/dev'

# Input directory
input_dir = os.path.join(project_root, 'data/_feast-human/da_filtered')

# Find all filtered Parquet files
filtered_files = glob.glob(os.path.join(input_dir, '*_cancer_filtered.parquet'))

# Read and combine
combined_dfs = []
for file in filtered_files:
    omics_type = os.path.basename(file).split('_')[0]  # e.g., 'proteomics' from 'proteomics_cancer_filtered.parquet'
    df = pd.read_parquet(file)
    df['omics_type'] = omics_type  # Ensure column if not already
    combined_dfs.append(df)

if not combined_dfs:
    print("No filtered files found.")
else:
    combined_df = pd.concat(combined_dfs, ignore_index=True)
    
    # Add any necessary columns (e.g., extract timepoint from contrast if needed)
    # Example: if 'contrast' exists, parse timepoint
    if 'contrast' in combined_df.columns:
        combined_df['timepoint'] = combined_df['contrast'].str.extract(r'group_timepointADU(?:Endur|Resist)\.(.*) - group_timepointADU(?:Endur|Resist)\.pre_exercise')
    
    # Export combined
    output_path = os.path.join(input_dir, 'combined_cancer_da.parquet')
    combined_df.to_parquet(output_path, index=False)
    print(f"Exported combined DA to {output_path}")
    
    # Optional: Pancreatic subset if column exists
    if 'is_pancreatic' in combined_df.columns and combined_df['is_pancreatic'].notna().any():
        pancreatic_df = combined_df[combined_df['is_pancreatic'].notna()]
        pancreatic_path = os.path.join(input_dir, 'pancreatic_combined_cancer_da.parquet')
        pancreatic_df.to_parquet(pancreatic_path, index=False)
        print(f"Exported pancreatic combined DA to {pancreatic_path}")
