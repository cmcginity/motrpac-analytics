import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Project root
project_root = '/Users/curtismcginity/stanford/research/proj/MoTrPAC/dev'

# Load enrichment results
enrichment_path = os.path.join(project_root, 'data/_feast-human/enrich/gse_hallmark_enrichment_results.csv')
enrichment_df = pd.read_csv(enrichment_path)

# Output directory
RUN_DATE = datetime.now().strftime('%Y-%m-%d-%H%M')
output_dir = os.path.join(project_root, f'output_feast/plots/{RUN_DATE}_hall_enrich')
os.makedirs(output_dir, exist_ok=True)

# Horizontal bar plot for significance
def plot_hallmark_bar(df, n_top=50, title="Hallmark Enrichment Bar Plot", save_path=None):
    df = df.sort_values('Adjusted P-value').head(n_top)
    df['neg_log10_p'] = -np.log10(df['Adjusted P-value'] + 1e-10)
    
    sns.set_style("whitegrid")
    sns.set_context("talk")
    fig, ax = plt.subplots(figsize=(10, max(8, len(df)/2)))
    palette = sns.color_palette("YlOrRd_r", n_colors=len(df))
    sns.barplot(x='neg_log10_p', y='Term', data=df, palette=palette, orient='h', ax=ax, errorbar=None)
    ax.set_xlabel(r'$-\log_{10}(Adj. P)$', fontsize=14)
    ax.set_ylabel('')
    ax.set_title(title, fontsize=16, weight='bold', pad=20)
    ax.yaxis.tick_right()
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

# Dot plot
def plot_hallmark_dot(df, title="Hallmark Enrichment Dot Plot", save_path=None):
    df = df.sort_values('Adjusted P-value')
    df['neg_log10_p'] = -np.log10(df['Adjusted P-value'] + 1e-10)
    
    sns.set_style("whitegrid")
    sns.set_context("talk")
    fig, ax = plt.subplots(figsize=(12, max(8, len(df)/1.5)))
    scatter = sns.scatterplot(
        data=df, x='neg_log10_p', y='Term', size='gene_count', hue='avg_logFC',
        palette='RdBu', sizes=(20, 200), ax=ax
    )
    # Normalize hue for better color scaling
    norm = plt.Normalize(df['avg_logFC'].min(), df['avg_logFC'].max())
    sm = plt.cm.ScalarMappable(cmap="RdBu", norm=norm)
    sm.set_array([])
    # Remove old legend and add separate legends
    ax.legend_.remove()
    # Add hue colorbar
    cbar = ax.figure.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('Avg logFC')
    # Add size legend manually
    handles, labels = scatter.get_legend_handles_labels()
    size_legend = ax.legend(handles=handles[0:4], labels=labels[0:4], title="Gene Count", bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    ax.add_artist(size_legend)
    ax.set_xlabel(r'$-\log_{10}(Adj. P)$', fontsize=14)
    ax.set_ylabel('')
    ax.set_title(title, fontsize=16, weight='bold', pad=20)
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

# Generate plots
bar_save_path = os.path.join(output_dir, 'hallmark_barplot.png')
plot_hallmark_bar(enrichment_df, title="Top Hallmark Gene Sets by Significance", save_path=bar_save_path)

dot_save_path = os.path.join(output_dir, 'hallmark_dotplot.png')
plot_hallmark_dot(enrichment_df, title="Hallmark Enrichment Dot Plot", save_path=dot_save_path)

print(f"Plots saved to {output_dir}")
