import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Project root
project_root = '/Users/curtismcginity/stanford/research/proj/MoTrPAC/dev'

# Load enrichment results
enrichment_path = os.path.join(project_root, 'data/_feast-human/enrichment_results.csv')
enrichment_df = pd.read_csv(enrichment_path)

# Output directory
output_dir = os.path.join(project_root, 'output/figures/cancer_enrichment')
os.makedirs(output_dir, exist_ok=True)

# Adapted from notebook: Bar plot for top pathways (per group if applicable)
def plot_top_enrichment_pathways(df, n_top=15, title="Top Enriched Pathways", save_path=None):
    df = df.sort_values('p_value').head(n_top)
    df['neg_log10_p'] = -np.log10(df['p_value'] + 1e-10)
    
    sns.set_style("whitegrid")
    sns.set_context("talk")
    fig, ax = plt.subplots(figsize=(10, 8))
    palette = sns.color_palette("YlOrRd_r", n_colors=n_top)
    sns.barplot(x='neg_log10_p', y='name', data=df, palette=palette, orient='h', ax=ax)
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

print(f"Plots saved to {output_dir}")
