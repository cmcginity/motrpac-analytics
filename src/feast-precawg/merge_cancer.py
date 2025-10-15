import os
import polars as pl
import json
import yaml

project_root = '/Users/curtismcginity/stanford/research/proj/MoTrPAC/dev'
config_path = os.path.join(project_root, 'config/feast/prelim.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# cancer_df = pl.read_csv(cancer_file)
# cancer_df = pl.read_csv(cancer_file, separator='\t') # temp hack for tsv file
# Load cancer mapping (CSV/TSV; flexible schema)
cancer_file = config['human_cancer_mapping_file']
sep = '\t' if cancer_file.endswith(('.tsv', '.txt')) else ','
try:
    cancer_df = pl.read_csv(
        cancer_file,
        separator=sep,
        null_values=['NA', 'na', 'NaN', ''],
        truncate_ragged_lines=True
    )
except Exception:
    # Retry with alternate separator if initial guess fails
    alt_sep = ',' if sep == '\t' else '\t'
    cancer_df = pl.read_csv(
        cancer_file,
        separator=alt_sep,
        null_values=['NA', 'na', 'NaN', ''],
        truncate_ragged_lines=True
    )

# Normalize/ensure 'gene_symbol' column
if 'gene_symbol' not in cancer_df.columns:
    alt_cols = [c for c in cancer_df.columns if c.lower().replace(' ', '') in {'genesymbol', 'gene', 'genename', 'symbol'}]
    if alt_cols:
        cancer_df = cancer_df.rename({alt_cols[0]: 'gene_symbol'})
    else:
        raise ValueError("Required column 'gene_symbol' not found in cancer mapping file.")

# Keep only necessary column to avoid downstream ambiguity
cancer_df = cancer_df.select(['gene_symbol']).unique()
cancer_unique_genes = cancer_df['gene_symbol'].n_unique()
print(f"Cancer unique genes: {cancer_unique_genes}")

# Load da_genes_df
da_path = os.path.join(project_root, 'data/_feast-human/da_filtered/combined_to_gene.parquet')
da_genes_df = pl.read_parquet(da_path)

# Inner join, keeping relevant cancer columns if needed
overlap_df = da_genes_df.join(cancer_df, on='gene_symbol', how='inner')
print(f"Overlap shape: {overlap_df.shape}")
overlap_unique_genes = overlap_df['gene_symbol'].n_unique()
print(f"Overlap unique genes: {overlap_unique_genes}")

# Update JSON
json_path = os.path.join(project_root, 'data/_feast-human/stats/venn_counts.json')
with open(json_path, 'r') as f:
    venn_counts = json.load(f)
venn_counts['cancer_genes'] = cancer_unique_genes
venn_counts['overlap_genes'] = overlap_unique_genes
with open(json_path, 'w') as f:
    json.dump(venn_counts, f)

# Export overlap
output_dir = os.path.join(project_root, 'data/_feast-human/da_filtered')
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'pancreatic_overlap.parquet')
overlap_df.write_parquet(output_path)
print(f"Exported overlap to {output_path}")
