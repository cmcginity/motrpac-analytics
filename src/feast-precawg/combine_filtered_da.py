import os
import glob
import polars as pl
import json

# Project root
project_root = '/Users/curtismcginity/stanford/research/proj/MoTrPAC/dev'

# Input directory
input_dir = os.path.join(project_root, 'data/_feast-human/da_filtered')

# Find all filtered Parquet files
filtered_files = glob.glob(os.path.join(input_dir, '*_significant.parquet'))

# Read and combine
if not filtered_files:
    print("No filtered files found.")
else:
    dfs = [pl.read_parquet(f) for f in filtered_files]
    combined_df = pl.concat(dfs)
    
    # Select specified columns
    combined_df = combined_df.select(['assay', 'feature_id', 'logFC', 'p_value', 'adj_p_value'])
    
    # Load and select mapping
    map_path = os.path.join(project_root, 'data/_feast-human/mapping/feature_to_gene.parquet')
    map_df = pl.read_parquet(map_path)
    map_df = map_df.select(['feature_id', 'gene_symbol', 'platform'])
    
    print(f"Combined DA shape before merge: {combined_df.shape}")
    da_genes_df = combined_df.join(map_df, on='feature_id', how='left')
    print(f"Combined DA shape after merge: {da_genes_df.shape}")
    
    # Compute unique genes
    total_unique_genes = da_genes_df.select(pl.col('gene_symbol').drop_nulls().n_unique()).item(0, 0)
    print(f"Total unique genes: {total_unique_genes}")
    
    # Save to JSON
    stats_dir = os.path.join(project_root, 'data/_feast-human/stats')
    os.makedirs(stats_dir, exist_ok=True)
    json_path = os.path.join(stats_dir, 'venn_counts.json')
    with open(json_path, 'w') as f:
        json.dump({'total_significant_genes': total_unique_genes}, f)
    
    # Extract timepoint if 'contrast' exists
    if 'contrast' in da_genes_df.columns:
        da_genes_df = da_genes_df.with_columns(
            pl.col('contrast').str.extract(r'group_timepointADU(?:Endur|Resist)\.(.*) - group_timepointADU(?:Endur|Resist)\.pre_exercise').alias('timepoint')
        )
    
    # Export combined
    output_path = os.path.join(input_dir, 'combined_to_gene.parquet')
    da_genes_df.write_parquet(output_path)
    print(f"Exported combined DA to {output_path}")
