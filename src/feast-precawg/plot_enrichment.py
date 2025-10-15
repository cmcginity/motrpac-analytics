import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from matplotlib_venn import venn2, venn2_circles
from matplotlib import pyplot as plt
from datetime import datetime

# Project root
project_root = '/Users/curtismcginity/stanford/research/proj/MoTrPAC/dev'

# Load enrichment results
enrichment_path = os.path.join(project_root, 'data/_feast-human/enrich/enrichment_results.csv')
enrichment_df = pd.read_csv(enrichment_path)

# Load pancreatic overlap data
panc_overlap_path = os.path.join(project_root, 'data/_feast-human/da_filtered/pancreatic_overlap.parquet')
panc_overlap_df = pd.read_parquet(panc_overlap_path)

# Handle edge cases: Check for required columns
if 'gene_symbol' not in panc_overlap_df.columns or 'logFC' not in panc_overlap_df.columns:
    print("Warning: Missing 'gene_symbol' or 'logFC' columns in pancreatic_overlap.parquet. Skipping logFC plot.")
    panc_overlap_df = pd.DataFrame()  # Empty DF to skip plotting
else:
    # Ensure unique gene_symbols by taking mean logFC if duplicates
    panc_overlap_df = panc_overlap_df.groupby('gene_symbol')['logFC'].mean().reset_index()
    # Sort by logFC ascending
    panc_overlap_df = panc_overlap_df.sort_values('logFC')

# Output directory
RUN_DATE = datetime.now().strftime('%Y-%m-%d-%H%M')
output_dir = os.path.join(project_root, f'output_feast/plots/{RUN_DATE}_pdac_enrich')
os.makedirs(output_dir, exist_ok=True)

# Adapted from notebook: Bar plot for top pathways (per group if applicable)
def plot_top_enrichment_pathways(df, n_top=15, title="Top Enriched Pathways", save_path=None):
    # Sort by significance and ensure unique pathway names to avoid implicit aggregation
    df = df.sort_values('p_value').drop_duplicates(subset=['name'], keep='first').head(n_top)
    df['neg_log10_p'] = -np.log10(df['p_value'] + 1e-10)
    
    sns.set_style("whitegrid")
    sns.set_context("talk")
    fig, ax = plt.subplots(figsize=(10, 8))
    palette = sns.color_palette("YlOrRd_r", n_colors=n_top)
    # Disable error bars to prevent small horizontal lines at bar tips (seaborn may add CIs)
    try:
        sns.barplot(x='neg_log10_p', y='name', data=df, palette=palette, orient='h', ax=ax, errorbar=None)
    except TypeError:
        # Fallback for older seaborn versions
        sns.barplot(x='neg_log10_p', y='name', data=df, palette=palette, orient='h', ax=ax, ci=None)
    ax.set_xlabel(r'$-\log_{10}(P)$', fontsize=14)
    ax.set_ylabel('')
    ax.set_title(title, fontsize=16, weight='bold', pad=20)
    ax.yaxis.tick_right()
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

# New function for logFC bar chart
def plot_logfc_bar_chart(df, n_top=None, title="LogFC for Pancreatic Overlap Genes", save_path=None):
    if df.empty:
        print("No data to plot for logFC bar chart.")
        return
    
    # Optionally select top N by absolute logFC
    if n_top is not None:
        df = df.iloc[df['logFC'].abs().argsort()[-n_top:]]
        df = df.sort_values('logFC')  # Re-sort after selection
    
    # Compute colors
    neg_df = df[df['logFC'] < 0]
    pos_df = df[df['logFC'] > 0]
    
    # Normalize negatives: more negative -> darker (0 to 1, where 1 is most negative)
    if not neg_df.empty:
        neg_norm = (neg_df['logFC'].abs() - neg_df['logFC'].abs().min()) / (neg_df['logFC'].abs().max() - neg_df['logFC'].abs().min() + 1e-10)
        neg_norm = 1 - neg_norm  # Invert so more negative (larger abs) -> higher norm -> darker
        neg_colors = sns.color_palette('Blues', len(neg_df))
        neg_colors = [neg_colors[int(i * (len(neg_colors) - 1))] for i in neg_norm]
    else:
        neg_colors = []
    
    # Normalize positives: larger -> darker (0 to 1)
    if not pos_df.empty:
        pos_norm = (pos_df['logFC'] - pos_df['logFC'].min()) / (pos_df['logFC'].max() - pos_df['logFC'].min() + 1e-10)
        pos_colors = sns.color_palette('YlOrRd_r', len(pos_df))
        pos_colors = [pos_colors[int(i * (len(pos_colors) - 1))] for i in pos_norm]
    else:
        pos_colors = []
    
    # Combine colors in order of df
    colors = neg_colors + [(0.5, 0.5, 0.5)] * (len(df) - len(neg_df) - len(pos_df)) + pos_colors  # Gray for zeros if any
    
    sns.set_style("whitegrid")
    sns.set_context("talk")
    fig, ax = plt.subplots(figsize=(10, max(8, len(df)/2)))
    sns.barplot(x='logFC', y='gene_symbol', data=df, palette=colors, orient='h', ax=ax, errorbar=None)
    ax.set_xlabel('logFC', fontsize=14)
    ax.set_ylabel('')
    ax.set_title(title, fontsize=16, weight='bold', pad=20)
    ax.yaxis.tick_right()
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

# Generate bar plots per group
if 'group' in enrichment_df.columns:
    for group in enrichment_df['group'].unique():
        group_df = enrichment_df[enrichment_df['group'] == group]
        save_path = os.path.join(output_dir, f'top_pathways_{group}.png')
        plot_top_enrichment_pathways(group_df, title=f"Top Pathways: {group}", save_path=save_path)
else:
    save_path = os.path.join(output_dir, 'top_pathways_overall.png')
    plot_top_enrichment_pathways(enrichment_df, title="Top Pathways: Overall", save_path=save_path)

# Heatmap of -log p across groups and top pathways
if 'group' in enrichment_df.columns and len(enrichment_df['group'].unique()) > 1:
    # Pivot: rows pathways, columns groups, values -log p
    pivot_df = enrichment_df.pivot_table(index='name', columns='group', values='p_value', aggfunc='min')
    pivot_df = -np.log10(pivot_df + 1e-10)
    # Filter to top pathways overall
    top_pathways = pivot_df.max(axis=1).sort_values(ascending=False).head(20).index
    pivot_df = pivot_df.loc[top_pathways]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(pivot_df, cmap='YlOrRd', annot=True, fmt='.2f')
    plt.title('Pathway Enrichment Heatmap Across Timepoints')
    heatmap_path = os.path.join(output_dir, 'enrichment_heatmap.png')
    plt.savefig(heatmap_path)
    plt.close()

# Handle pancreatic if separate file exists
panc_path = os.path.join(project_root, 'data/_feast-human/pancreatic_enrichment_results.csv')
if os.path.exists(panc_path):
    panc_df = pd.read_csv(panc_path)
    panc_save_path = os.path.join(output_dir, 'top_pathways_pancreatic.png')
    plot_top_enrichment_pathways(panc_df, title="Top Pathways: Pancreatic", save_path=panc_save_path)

# Generate logFC bar chart for pancreatic overlap
logfc_save_path = os.path.join(output_dir, 'pancreatic_overlap_logfc.png')
plot_logfc_bar_chart(panc_overlap_df, title="LogFC for Pancreatic Overlap Genes", save_path=logfc_save_path)

# Load Venn counts and generate professional Venn diagram (optional if matplotlib_venn installed)
json_path = os.path.join(project_root, 'data/_feast-human/stats/venn_counts.json')
with open(json_path, 'r') as f:
    venn_counts = json.load(f)

total = venn_counts['total_significant_genes']
cancer = venn_counts['cancer_genes']
overlap = venn_counts['overlap_genes']

# Compute set sizes for venn2: (only A, only B, intersection)
only_significant = total - overlap
only_cancer = cancer - overlap

plt.figure(figsize=(8, 8))
v = venn2(subsets=(only_significant, only_cancer, overlap), set_labels=('Significant Genes', 'Cancer Genes'))
venn2_circles(subsets=(only_significant, only_cancer, overlap), linestyle='solid', linewidth=1)

# Style to mimic ggVennDiagram: clean, with colors
for text in v.set_labels:
    text.set_fontsize(14)
for text in v.subset_labels:
    if text:
        text.set_fontsize(12)
v.get_patch_by_id('10').set_color('skyblue')
v.get_patch_by_id('01').set_color('lightgreen')
v.get_patch_by_id('11').set_color('mediumseagreen')
for p in v.patches:
    if p:
        p.set_edgecolor('black')
        p.set_alpha(0.8)

plt.title('Overlap of Significant Genes and Cancer Markers', fontsize=16)
venn_path = os.path.join(output_dir, 'venn_diagram.png')
plt.savefig(venn_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Exported Venn diagram to {venn_path}")


print(f"Plots saved to {output_dir}")
