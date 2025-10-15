import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import matplotlib.colors

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
        palette='RdBu_r', sizes=(20, 200), ax=ax
    )
    # Normalize hue for better color scaling
    norm = plt.Normalize(df['avg_logFC'].min(), df['avg_logFC'].max())
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
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

# New logFC-colored bar plot
def plot_hallmark_logfc_bar(df, n_top=50, sort_by_logfc_groups=False, title="Hallmark Enrichment Bar Plot (Colored by Avg logFC)", save_path=None):
    df = df.sort_values('Adjusted P-value').head(n_top)
    df['neg_log10_p'] = -np.log10(df['Adjusted P-value'] + 1e-10)
    
    if sort_by_logfc_groups:
        # Split into positive and negative logFC
        pos_df = df[df['avg_logFC'] >= 0].sort_values('neg_log10_p', ascending=False)
        neg_df = df[df['avg_logFC'] < 0].sort_values('neg_log10_p', ascending=True)
        # Concat positives first, then negatives
        df = pd.concat([pos_df, neg_df])
    else:
        # Default: Sort by significance descending
        df = df.sort_values('neg_log10_p', ascending=False)
    
    # Compute colors based on avg_logFC
    # norm = plt.Normalize(df['avg_logFC'].min(), df['avg_logFC'].max())
    min_val = df['avg_logFC'].min()
    max_val = df['avg_logFC'].max()
    max_abs = max(abs(min_val), abs(max_val))
    norm = matplotlib.colors.TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)
    cmap = plt.cm.get_cmap('RdBu_r')
    colors = [cmap(norm(value)) for value in df['avg_logFC']]
    
    sns.set_style("whitegrid")
    sns.set_context("talk")
    fig, ax = plt.subplots(figsize=(14, max(8, len(df)/2)))  # Increased width for space
    sns.barplot(x='neg_log10_p', y='Term', data=df, palette=colors, orient='h', ax=ax, errorbar=None)
    ax.set_xlabel(r'$-\log_{10}(Adj. P)$', fontsize=14)
    ax.set_ylabel('')
    ax.set_title(title, fontsize=16, weight='bold', pad=20, loc='center')
    ax.yaxis.tick_right()
    
    # Add colorbar for avg_logFC on the right
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
    sm.set_array([])
    cbar = ax.figure.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.05, location='left')  # Increased pad
    cbar.set_label('Avg logFC')
    
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

# Generate plots
bar_save_path = os.path.join(output_dir, 'hallmark_barplot.png')
plot_hallmark_bar(enrichment_df, title="Top Hallmark Gene Sets by Significance", save_path=bar_save_path)

dot_save_path = os.path.join(output_dir, 'hallmark_dotplot.png')
plot_hallmark_dot(enrichment_df, title="Hallmark Enrichment Dot Plot", save_path=dot_save_path)

logfc_bar_save_path = os.path.join(output_dir, 'hallmark_logfc_barplot.png')
plot_hallmark_logfc_bar(enrichment_df, sort_by_logfc_groups=True, title="Top Hallmark Gene Sets by Significance (Colored by Avg logFC)", save_path=logfc_bar_save_path)

print(f"Plots saved to {output_dir}")
