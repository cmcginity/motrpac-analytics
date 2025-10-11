import os
import pandas as pd
import yaml
import gseapy as gp

# Load config (using same as prelim)
project_root = '/Users/curtismcginity/stanford/research/proj/MoTrPAC/dev'
config_path = os.path.join(project_root, 'config/feast/prelim.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Load combined DA (use combined_to_gene as per plan)
combined_path = os.path.join(project_root, 'data/_feast-human/da_filtered/combined_to_gene.parquet')
combined_df = pd.read_parquet(combined_path)

# Assume already filtered significant
da_genes_df = combined_df

# Compute average logFC per gene
gene_logfc = da_genes_df.groupby('gene_symbol')['logFC'].mean().reset_index()

# Extract unique gene list
gene_list = da_genes_df['gene_symbol'].unique().tolist()

# Perform enrichment
enr = gp.enrichr(gene_list=gene_list, gene_sets="MSigDB_Hallmark_2020", organism='Human', outdir=None)
results = enr.results

# Filter significant
significant_results = results[results['Adjusted P-value'] < 0.05]

# Add gene_count and avg_logFC
significant_results['gene_count'] = significant_results['Genes'].apply(lambda x: len(x.split(';')))
significant_results['avg_logFC'] = significant_results['Genes'].apply(
    lambda x: gene_logfc[gene_logfc['gene_symbol'].isin(x.split(';'))]['logFC'].mean() if x else 0
)

# Export
enrich_dir = os.path.join(project_root, 'data/_feast-human/enrich')
os.makedirs(enrich_dir, exist_ok=True)
output_path = os.path.join(enrich_dir, 'gse_hallmark_enrichment_results.csv')
significant_results.to_csv(output_path, index=False)
print(f"Exported Hallmark enrichment results to {output_path}")
