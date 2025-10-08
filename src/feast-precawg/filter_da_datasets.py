import os
import pandas as pd
import yaml

# Load config
project_root = '/Users/curtismcginity/stanford/research/proj/MoTrPAC/dev'
config_path = os.path.join(project_root, 'config/feast/prelim.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Load merged mapping
mapping_path = os.path.join(project_root, 'data/_feast-human/mapping/cancer_feature_gene_mapping.csv')
mapping_df = pd.read_csv(mapping_path)

# Define omics types and their mapping types
omics_types = {
    'proteomics': 'proteomics',
    'atac': 'general',
    'rna': 'general'
}

# Output directory
output_dir = os.path.join(project_root, 'data/_feast-human/da_filtered')
os.makedirs(output_dir, exist_ok=True)

for omics, map_type in omics_types.items():
    if omics not in config['gcs_paths'] or omics not in config['blood_files']:
        print(f"Skipping {omics}: Config missing.")
        continue
    
    # Get DA file path
    da_dir = config['gcs_paths'][omics]['da_data']
    da_file = config['blood_files'][omics]['da_file']
    da_path = f'gs://{da_dir}/{da_file}'
    
    # Read DA data (assume tab-separated txt)
    print(f"Reading DA for {omics} from {da_path}")
    da_df = pd.read_csv(da_path, sep='\t')
    
    # Filter mapping for this omics
    omics_mapping = mapping_df[mapping_df['omics_type'] == map_type]
    
    # Inner merge to filter DA to cancer-related features (adds mapping info)
    filtered_df = pd.merge(da_df, omics_mapping, on='feature_id', how='inner')
    
    # Handle any omics-specific logic (e.g., for proteomics if needed)
    if omics == 'proteomics':
        # Example: if merge was on different column, adjust here
        pass
    
    # Export as Parquet
    output_path = os.path.join(output_dir, f'{omics}_cancer_filtered.parquet')
    filtered_df.to_parquet(output_path, index=False)
    print(f"Exported filtered DA for {omics} to {output_path}")
