import os
import polars as pl
import yaml
from google_utils import GoogleCloudHelper

# Load config
project_root = '/Users/curtismcginity/stanford/research/proj/MoTrPAC/dev'
config_path = os.path.join(project_root, 'config/feast/prelim.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Initialize GoogleCloudHelper (provide quota project ID if needed; assume from config or env)
helper = GoogleCloudHelper(gcs_quota_project_id=None)  # Set your quota project ID here if required

# Read general feature-to-gene mapping (GCS, tab-separated)
general_map_path = config['human_feature_to_gene_file']  # No "gs://" addition
bucket, blob = general_map_path.split('/', 1)  # Directly parse as bucket/path
general_buffer = helper.download_gcs_file_as_stringio(bucket, blob)
general_df = pl.read_csv(general_buffer, separator='\t', null_values="NA")
general_df = general_df.with_columns(pl.lit('general').alias('omics_type'))
print(f"General mapping shape: {general_df.shape}")
print(f"Unique feature_ids: {general_df['feature_id'].n_unique()}")
print(f"Unique gene_symbols: {general_df['gene_symbol'].n_unique()}")

# Handle proteomics separately if mapping exists
# merged_dfs = [general_df]
# 
# prot_config = config['gcs_paths'].get('proteomics', {})
# if 'feature_map_data' in prot_config and 'proteomics' in config['blood_files']:
#     prot_dir = prot_config['feature_map_data']
#     prot_file = config['blood_files']['proteomics']['feature_map_file']
#     prot_map_path = f'{prot_dir}/{prot_file}'  # No "gs://"
#     bucket, blob = prot_map_path.split('/', 1)
#     prot_buffer = helper.download_gcs_file_as_stringio(bucket, blob)
#     prot_df = pl.read_csv(prot_buffer.getvalue(), separator='\t')
#     prot_df = prot_df.with_columns(pl.lit('proteomics').alias('omics_type'))
#     merged_dfs.append(prot_df)

# Combine
# combined_df = pl.concat(merged_dfs)
combined_df = general_df

print(f"Combined mapping shape: {combined_df.shape}")
print(f"Unique feature_ids: {combined_df['feature_id'].n_unique()}")
print(f"Unique gene_symbols: {combined_df['gene_symbol'].n_unique()}")


# Export as Parquet
output_dir = os.path.join(project_root, 'data/_feast-human/mapping')
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'feature_to_gene.parquet')
combined_df.write_parquet(output_path)
print(f"Exported combined mapping to {output_path}")
