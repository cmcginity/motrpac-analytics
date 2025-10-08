import os
import pandas as pd
import yaml

# Load config
project_root = '/Users/curtismcginity/stanford/research/proj/MoTrPAC/dev'
config_path = os.path.join(project_root, 'config/feast/prelim.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Read cancer mapping (local CSV)
cancer_file = config['human_cancer_mapping_file']
cancer_df = pd.read_csv(cancer_file)

# Read general feature-to-gene mapping (GCS, tab-separated)
general_map_path = 'gs://' + config['human_feature_to_gene_file']
general_df = pd.read_csv(general_map_path, sep='\t')

# Left merge: cancer left, features right
general_merged = pd.merge(cancer_df, general_df, on='gene_symbol', how='left')
general_merged['omics_type'] = 'general'

# Handle proteomics separately if mapping exists
merged_dfs = [general_merged]

prot_config = config['gcs_paths'].get('proteomics', {})
if 'feature_map_data' in prot_config and 'proteomics' in config['blood_files']:
    prot_dir = prot_config['feature_map_data']
    prot_file = config['blood_files']['proteomics']['feature_map_file']
    prot_map_path = f'gs://{prot_dir}/{prot_file}'
    prot_df = pd.read_csv(prot_map_path, sep='\t')
    # Assume columns: feature_id, uniprot_entry, gene_symbol, ...
    # If needed, rename: prot_df = prot_df.rename(columns={'assay': 'gene_symbol'})  # Adjust based on actual
    prot_merged = pd.merge(cancer_df, prot_df, left_on='gene_symbol', right_on='assay', how='left')  # Adjust 'assay' if it's the gene column
    prot_merged['omics_type'] = 'proteomics'
    merged_dfs.append(prot_merged)

# Combine all merged dataframes
combined_merged = pd.concat(merged_dfs, ignore_index=True)

# Optional: Filter for pancreatic if column exists and has values
if 'is_pancreatic' in combined_merged.columns and combined_merged['is_pancreatic'].notna().any():
    pancreatic_df = combined_merged[combined_merged['is_pancreatic'].notna()]  # Or specific condition
    pancreatic_path = os.path.join(project_root, 'data/_feast-human/mapping/pancreatic_cancer_feature_gene_mapping.csv')
    pancreatic_df.to_csv(pancreatic_path, index=False)
    print(f"Exported pancreatic-specific mapping to {pancreatic_path}")

# Export combined mapping
output_dir = os.path.join(project_root, 'data/_feast-human/mapping')
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'cancer_feature_gene_mapping.csv')
combined_merged.to_csv(output_path, index=False)
print(f"Exported combined mapping to {output_path}")
