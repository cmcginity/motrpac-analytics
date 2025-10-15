import os
import pandas as pd
import yaml
import polars as pl  # Added for efficient GCS reading
from google_utils import GoogleCloudHelper

# Load config
project_root = '/Users/curtismcginity/stanford/research/proj/MoTrPAC/dev'
config_path = os.path.join(project_root, 'config/feast/prelim.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Output directory
output_dir = os.path.join(project_root, 'data/_feast-human/da_filtered')
os.makedirs(output_dir, exist_ok=True)

# Initialize GoogleCloudHelper (provide quota project ID if needed; assume from config or env)
helper = GoogleCloudHelper(gcs_quota_project_id=None)  # Set your quota project ID here if required

for omics in config['gcs_paths'].keys():
    if omics not in config['blood_files']:
        print(f"Skipping {omics}: Config missing.")
        continue
    
    # Get DA file path
    da_dir = config['gcs_paths'][omics]['da_data']
    da_file = config['blood_files'][omics]['da_file']
    da_path = f'{da_dir}/{da_file}'  # No "gs://"
    bucket, blob = da_path.split('/', 1)
    da_buffer = helper.download_gcs_file_as_stringio(bucket, blob)
    
    # Read DA data (assume tab-separated txt)
    print(f"Reading DA for {omics} from {da_path}")
    da_buffer.seek(0)
    da_df = pl.read_csv(da_buffer, separator='\t')
    
    # Filter to significant
    threshold = config['analysis_params']['significance_threshold']
    p_adj_col = config['columns']['p_value_adj']
    da_df = da_df.filter(pl.col(p_adj_col) < threshold)
    print(f"Filtered DA for {omics} shape: {da_df.shape}")
    
    # Export as Parquet
    output_path = os.path.join(output_dir, f'{omics}_significant.parquet')
    da_df.write_parquet(output_path)
    print(f"Exported filtered DA for {omics} to {output_path}")
