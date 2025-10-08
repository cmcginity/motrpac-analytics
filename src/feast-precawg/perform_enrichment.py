import os
import pandas as pd
import yaml
from gprofiler import GProfiler

# Load config
project_root = '/Users/curtismcginity/stanford/research/proj/MoTrPAC/dev'
config_path = os.path.join(project_root, 'config/feast/prelim.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Load combined DA
combined_path = os.path.join(project_root, 'data/_feast-human/da_filtered/combined_cancer_da.parquet')
combined_df = pd.read_parquet(combined_path)

# Extract timepoint if not already (from Step 4 example)
if 'timepoint' not in combined_df.columns and 'contrast' in combined_df.columns:
    combined_df['timepoint'] = combined_df['contrast'].str.extract(r'group_timepointADU(?:Endur|Resist)\.(.*) - group_timepointADU(?:Endur|Resist)\.pre_exercise')[0]

# Significance threshold
threshold = config['analysis_params']['significance_threshold']

# Filter significant features (optional thresholding)
significant_df = combined_df[(combined_df[config['columns']['p_value_adj']] < threshold)]  # Add |logFC| > x if needed

# Get unique groups (e.g., timepoints)
groups = significant_df['timepoint'].unique() if 'timepoint' in significant_df.columns else ['overall']
if 'overall' in groups:
    groups = ['overall']  # Or include per timepoint

# Initialize GProfiler
gp = GProfiler(return_dataframe=True, user_agent='feast_precawg_analysis')

enrichment_results = {}

for group in groups:
    if group == 'overall':
        group_df = significant_df
    else:
        group_df = significant_df[significant_df['timepoint'] == group]
    
    genes = group_df['gene_symbol'].dropna().unique().tolist()  # Use gene_symbol; adjust to 'entrez_gene' if available
    
    if not genes:
        continue
    
    enrichment_df = gp.profile(
        organism='hsapiens',
        query=genes,
        sources=['GO:BP', 'KEGG', 'REAC', 'WP'],  # Cancer-focused
        user_threshold=0.05,
        significance_threshold_method='gSCS'
    )
    
    if not enrichment_df.empty:
        enrichment_df['group'] = group
        enrichment_results[group] = enrichment_df

# Postprocess (adapt from temporal_clustering.py)
processed_results = {}
term_size_min = config['analysis_params']['pathway_enrichment']['term_size_min']
term_size_max = config['analysis_params']['pathway_enrichment']['term_size_max']
top_n = config['analysis_params']['pathway_enrichment']['top_n_pathways']

for group, df in enrichment_results.items():
    filtered = df[(df['term_size'] > term_size_min) & (df['term_size'] < term_size_max)]
    sorted_df = filtered.sort_values('p_value')
    processed_results[group] = sorted_df.head(top_n)

# Combine and export
if processed_results:
    all_enrichment = pd.concat(processed_results.values(), ignore_index=True)
    output_path = os.path.join(project_root, 'data/_feast-human/enrichment_results.csv')
    all_enrichment.to_csv(output_path, index=False)
    print(f"Exported enrichment results to {output_path}")

# Optional: Pancreatic-specific
if 'is_pancreatic' in combined_df.columns and combined_df['is_pancreatic'].notna().any():
    panc_df = combined_df[combined_df['is_pancreatic'].notna()]
    # Repeat the above process for panc_df, export to pancreatic_enrichment_results.csv
    # Omitted for brevity; implement similarly if needed
