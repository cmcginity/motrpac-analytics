import os
import pickle
import pandas as pd
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import skfuzzy as fuzz
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from gprofiler import GProfiler
from wordcloud import WordCloud
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


print(f"Polars version: {pl.__version__}")

# Configuration
CONFIG = {
    "base_dir_da": "/Users/curtismcginity/stanford/research/proj/MoTrPAC/dev/data/rat-6m",
    "deseq_file_name": "rat-acute-06_t68-liver_epigen-atac-seq_da_timewise-deseq2-phase-frip-t2f_v2.0.txt",
    "pathway_enrichment_file_name": "rat-acute-06_epigenomics_metadata_rat-acute-06_t68-liver_epigen-atac-seq_metadata_features_v1.0.txt",
    "tissue": "liver",
    "significance_threshold": 0.05,
    "cluster_range": list(range(3, 16)),
    "num_random_seeds": 1,
    "use_precomputed_seeds": False,
    "explicit_seeds": [19805, 15184, 98839, 23642, 57386, 6259, 55376, 9620, 88540, 57914, 55293, 96857, 7553, 82196, 20470, 96857, 72936, 72936, 26744, 99338, 60150, 55293, 7553, 90714, 60218, 59272],
    "explicit_cluster_choice": {
        "male": {
            "liver": 9
        },
        "female": {
            "liver": 4
        }
    },
    "base_dir_output": "/Users/curtismcginity/stanford/research/proj/MoTrPAC/dev/data/output",
    "fuzziness": 1.2,
    "palette": "viridis",
    "term_size_min": 0,
    "term_size_max": 400,
    "top_n_pathways": 75,
    "columns": {
        "p_value_adj": "adj_p_value",
        "sex": "sex",
        "feature_ID": "feature_ID",
        "timepoint": "timepoint",
        "logFC": "logFC"
    }
}

def get_output_dir(tissue):
    """Creates and returns the output directory for a given tissue."""
    output_dir = os.path.join(CONFIG["base_dir_output"], tissue)
    print(f"Ensuring output directory exists: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def load_and_preprocess_data(file_name, significance_threshold, sex):
    """
    Loads, parses, and preprocesses the input data using Polars for performance.
    """
    file_path = os.path.join(CONFIG["base_dir_da"], file_name)
    print(f"Loading data from: {file_path}")

    p_val_col = CONFIG["columns"]["p_value_adj"]
    sex_col = CONFIG["columns"]["sex"]
    feature_col = CONFIG["columns"]["feature_ID"]
    
    try:
        # Use Polars for fast, multi-threaded CSV reading.
        df_pl = pl.read_csv(file_path, separator="\t")

        # 1. Standardize data types FIRST. This is crucial for filtering.
        df_pl = df_pl.with_columns(
            pl.when(pl.col(p_val_col) == "NA")
            .then(pl.lit(1.0, dtype=pl.Float64))
            .otherwise(pl.col(p_val_col).cast(pl.Float64, strict=False))
            .alias(p_val_col)
        ).drop_nulls(subset=[p_val_col])

        # 3. Filter based on significance and sex.
        df_filtered_pl = (
            df_pl.lazy()
            .filter(pl.col(sex_col) == sex) # Filter by sex.
            .filter(pl.col(p_val_col).min().over(feature_col) < significance_threshold) # Keep all timepoints for features that are significant at any point
            .collect()
        )
        
        # Convert to pandas at the end for compatibility with the rest of the script.
        df_sex = df_filtered_pl.to_pandas()
        
        print(f"{df_sex.head(20)}")
        print(f"  - Found {df_sex[feature_col].nunique()} significant features for {sex}.")
        return df_sex

    except Exception as e:
        print(f"An error occurred during data loading with Polars: {e}")
        # Return an empty DataFrame to allow the pipeline to continue gracefully
        return pd.DataFrame()

def xie_beni_index(data: np.ndarray, centers: np.ndarray, 
                  membership_matrix: np.ndarray, m: float) -> float:
    """Calculate Xie-Beni index: ratio of within-cluster compactness to between-cluster separation"""
    n = data.shape[0]
    c = centers.shape[0]
    
    # Add a small epsilon to prevent division by zero in log
    epsilon = np.finfo(float).eps
    u_clipped = np.clip(membership_matrix, epsilon, 1.0)
    u_power_m = u_clipped ** m
    
    squared_distances = cdist(data, centers, 'sqeuclidean').T # (c, n_samples)
    
    compactness = np.sum(u_power_m * squared_distances)
    
    if c > 1:
        # Pairwise squared distances between centroids
        center_dists_sq = cdist(centers, centers, 'sqeuclidean')
        # Set diagonal to infinity so we find the minimum of off-diagonal elements
        np.fill_diagonal(center_dists_sq, np.inf)
        min_dist_sq = np.min(center_dists_sq)
        
        # Add epsilon for numerical stability
        denominator = (n * min_dist_sq) + epsilon
        return compactness / denominator
    else:
        return np.inf

def run_cmeans_for_k(processed_df, k, precomputed_seed=None):
    """
    Runs c-means for a single k over multiple seeds and returns the best result.
    If a precomputed_seed is provided, it will use that seed directly.
    """
    data_values = processed_df.values
    data_t = data_values.T
    m = CONFIG.get("fuzziness", 1.2)

    # If a precomputed seed is provided, use it. Otherwise, search.
    if precomputed_seed is not None:
        seeds_to_test = [precomputed_seed]
        print(f"  - Using precomputed seed {precomputed_seed} for k={k}")
    else:
        # Generate a diverse and comprehensive list of seeds to test
        num_random = CONFIG.get("num_random_seeds", 10)
        explicit = CONFIG.get("explicit_seeds", [])
        random_seeds = np.random.randint(0, 100000, size=num_random)
        
        # Combine explicit seeds with random seeds and ensure uniqueness
        seeds_to_test = sorted(list(set(explicit + list(random_seeds))))
        print(f"  - Testing {len(seeds_to_test)} seeds for k={k}: {seeds_to_test}")

    best_seed_result = {
        "model": None, "xb_index": np.inf, "seed": -1, "labels": None, 
        "centroids": None, "fpc": -1
    }

    for seed in seeds_to_test:
        # Set the seed for numpy's random number generator for reproducibility, mirroring notebook
        np.random.seed(seed)

        # Perform fuzzy c-means clustering using the notebook's format
        cntr, u, _, _, _, _, fpc = fuzz.cluster.cmeans(
            data=data_t,
            c=k,
            m=m,
            error=0.0005,
            maxiter=1000,
            init=None
        )
        
        xb_index = xie_beni_index(data_values, cntr, u, m)
        # For very verbose debugging, uncomment the line below
        # print(f"  k={k}, seed={seed}: XB={xb_index:.4f}, FPC={fpc:.4f}")
        
        if xb_index < best_seed_result["xb_index"]:
            best_seed_result = {
                "model": {'u': u, 'centroids': cntr, 'fpc': fpc},
                "xb_index": xb_index,
                "seed": seed,
                "labels": np.argmax(u, axis=0),
                "centroids": cntr,
                "fpc": fpc
            }
            
    print(f"--> Best for k={k}: seed={best_seed_result['seed']} with XB={best_seed_result['xb_index']:.4f}, FPC={best_seed_result['fpc']:.4f}")
    return best_seed_result

def plot_clustering_metrics(metrics_df, output_dir, tissue, sex):
    """Plots the Xie-Beni index and FPC vs. the number of clusters."""
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Xie-Beni Index
    ax1.plot(metrics_df['n_clusters'], metrics_df['xie_beni_index'], 'o-')
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Xie-Beni Index')
    ax1.set_title(f'Xie-Beni Index (m={CONFIG["fuzziness"]})')
    ax1.set_yscale('log')
    ax1.grid(True, which="both", ls="-", alpha=0.5)
    ax1.tick_params(axis='x')

    # Plot 2: Fuzzy Partition Coefficient
    ax2.plot(metrics_df['n_clusters'], metrics_df['fpc'], 'o-')
    ax2.set_xlabel('Number of Clusters')
    ax2.set_ylabel('Fuzzy Partition Coefficient')
    ax2.set_title('Fuzzy Partition Coefficient')
    ax2.grid(True, which="both", ls="-", alpha=0.5)
    ax2.tick_params(axis='x')

    fig.suptitle(f"Clustering Evaluation Metrics for {sex.capitalize()} {tissue.capitalize()}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plot_path = os.path.join(output_dir, f"{tissue}_{sex}_clustering_metrics.png")
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"  - Saved clustering metrics plot to {plot_path}")

def find_optimal_clusters(data_filtered, output_dir, tissue, sex):
    """
    Finds the optimal number of clusters by evaluating metrics across a range of k values.
    """
    processed_df = preprocess_data_for_clustering(data_filtered)
    
    # Check if we should use precomputed seeds
    use_precomputed = CONFIG.get("use_precomputed_seeds", False)
    precomputed_seeds = {}
    if use_precomputed:
        seed_file_path = os.path.join(output_dir, f"{tissue}_{sex}_best_seeds.pkl")
        try:
            with open(seed_file_path, "rb") as f:
                precomputed_seeds = pickle.load(f)
            print(f"--> Loaded precomputed seeds from {seed_file_path}")
        except FileNotFoundError:
            print(f"--> WARNING: 'use_precomputed_seeds' is True but file not found: {seed_file_path}. Seed search will be performed instead.")
            use_precomputed = False  # Fallback to searching

    metrics = []
    all_results = {}
    
    print(f"Starting clustering search for {sex}...")
    for k in CONFIG["cluster_range"]:
        seed_for_k = precomputed_seeds.get(k) if use_precomputed else None
        best_result_for_k = run_cmeans_for_k(processed_df, k, precomputed_seed=seed_for_k)
        metrics.append({
            'n_clusters': k,
            'xie_beni_index': best_result_for_k['xb_index'],
            'fpc': best_result_for_k['fpc']
        })
        all_results[k] = best_result_for_k

    metrics_df = pd.DataFrame(metrics)
    
    # Plot clustering metrics
    plot_clustering_metrics(metrics_df, output_dir, tissue, sex)

    # Determine optimal k, allowing for a manual override from the config
    source = "Xie-Beni Index"
    optimal_k = int(metrics_df.loc[metrics_df['xie_beni_index'].idxmin()]['n_clusters'])

    explicit_override = CONFIG.get("explicit_cluster_choice", {}).get(sex, {}).get(tissue)
    if explicit_override:
        if explicit_override in all_results:
            optimal_k = explicit_override
            source = "explicit choice"
        else:
            print(f"--> WARNING: Manual k={explicit_override} for {tissue} {sex} not in cluster range {CONFIG['cluster_range']}. Using automatic value.")

    print(f"--> Optimal number of clusters for {sex} based on {source}: {optimal_k}")

    # Save artifacts
    optimal_result = all_results[optimal_k]
    file_path = os.path.join(output_dir, f"{tissue}_{sex}_optimal_clustering.pkl")
    with open(file_path, "wb") as f:
        pickle.dump(optimal_result, f)
    print(f"  - Saved optimal clustering results to {file_path}")

    best_seeds = {k: res["seed"] for k, res in all_results.items()}
    file_path = os.path.join(output_dir, f"{tissue}_{sex}_best_seeds.pkl")
    with open(file_path, "wb") as f:
        pickle.dump(best_seeds, f)
    print(f"  - Saved best seeds for each k to {file_path}")
        
    return all_results, optimal_k, processed_df

def preprocess_data_for_clustering(df):
    """
    Transforms dataframe into format for clustering: each feature_ID becomes a row with timepoint values.
    """
    feature_col = CONFIG["columns"]["feature_ID"]
    log2fc_col = CONFIG["columns"]["logFC"]
    timepoint_col = CONFIG["columns"]["timepoint"]
    
    # Sort by timepoint to ensure correct column order after pivot
    timepoints = sorted(df[timepoint_col].unique())
    
    print(f"  - Pivoting data for clustering...")
    pivot_df = df.pivot_table(
        index=feature_col, 
        columns=timepoint_col, 
        values=log2fc_col,
    )
    
    # Reorder columns to be monotonically increasing timepoints
    pivot_df = pivot_df[timepoints]
    
    print(f"  - Pivoted data shape (before NaN removal): {pivot_df.shape}")
    
    # Drop features with any missing timepoints, as they cannot be clustered.
    n_before = len(pivot_df)
    pivot_df.dropna(inplace=True)
    n_after = len(pivot_df)
    if n_before > n_after:
        print(f"  - Dropped {n_before - n_after} features with missing timepoint data.")

    print(f"  - Final data shape for clustering: {pivot_df.shape}")
    
    # NOTE: Scaling functionality could be added here if needed in the future.
    # For now, we return the unscaled, pivoted data.
    
    return pivot_df

def plot_all_feature_trajectories(data_filtered, output_dir, tissue, sex):
    """
    Plots all feature trajectories using the long-format data, colored by max logFC, with a colorbar.
    """
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create a copy to avoid modifying the original dataframe
    plot_data = data_filtered.copy()
    # Replace inf with NaN for plotting compatibility
    plot_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Calculate the metric for the color scale (max logFC per feature)
    metric_col = CONFIG["columns"]["logFC"]
    feature_col = CONFIG["columns"]["feature_ID"]
    time_col = CONFIG["columns"]["timepoint"]
    
    color_scale_metric = (
        plot_data
        .groupby(feature_col)[metric_col]
        .mean()
        .sort_values(ascending=False)
    )

    # Set up the colormap and normalization
    palette_name = "coolwarm" # A professional blue-red palette
    cmap = plt.get_cmap(palette_name)
    norm = Normalize(vmin=color_scale_metric.min(), vmax=color_scale_metric.max())

    # Plot each feature's trajectory
    # Looping is necessary for this type of plot with a continuous color mapping.
    # Performance is generally acceptable for a few thousand features.
    for feature_id in color_scale_metric.index:
        feature_data = plot_data[plot_data[feature_col] == feature_id].sort_values(time_col)
        color_val = color_scale_metric[feature_id]
        
        ax.plot(
            feature_data[time_col], 
            feature_data[metric_col], 
            color=cmap(norm(color_val)), 
            alpha=0.2
        )

    # Set labels, title, and limits
    ax.set_title(f"All Feature Trajectories for {sex.capitalize()} {tissue.capitalize()}")
    ax.set_xlabel("Timepoint")
    ax.set_ylabel("Log Fold Change")
    ax.tick_params(axis='x', labelrotation=45)
    
    # Create and add the colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Average Log Fold Change')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{tissue}_{sex}_all_trajectories.png")
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"  - Saved all trajectories plot with colorbar to {plot_path}")

def plot_clusters_with_centroids(data_with_clusters, processed_df, centroids, output_dir, tissue, sex, optimal_k):
    """Plots all feature trajectories colored by cluster with centroids on a single plot."""
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)
    plt.figure(figsize=(12, 8))

    # Replace inf with NaN for plotting compatibility to avoid warnings
    plot_data = data_with_clusters.copy()
    plot_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    palette = sns.color_palette(CONFIG["palette"], optimal_k)
    
    sns.lineplot(
        data=plot_data,
        x=CONFIG["columns"]["timepoint"],
        y=CONFIG["columns"]["logFC"],
        hue="cluster",
        units=CONFIG["columns"]["feature_ID"],
        estimator=None,
        alpha=0.15,
        palette=palette
    )
    
    for i in range(optimal_k):
        plt.plot(processed_df.columns, centroids[i], color=palette[i], linewidth=4, label=f"Cluster {i+1} Centroid")

    plt.title(f"Feature Trajectories by Cluster for {sex.capitalize()} {tissue.capitalize()}")
    plt.xlabel("Timepoint")
    plt.ylabel("Log Fold Change")
    # plt.xticks(rotation=45)
    
    # Improve legend
    handles, labels = plt.gca().get_legend_handles_labels()
    # Keep only centroid labels
    centroid_handles = [h for h, l in zip(handles, labels) if "Centroid" in l]
    centroid_labels = [l for l in labels if "Centroid" in l]
    plt.legend(centroid_handles, centroid_labels)

    plot_path = os.path.join(output_dir, f"{tissue}_{sex}_clusters_with_centroids.png")
    plt.savefig(plot_path)
    plt.close()

def plot_only_centroids(centroids, timepoints, output_dir, tissue, sex, optimal_k):
    """Plots only the cluster centroids with professional styling."""
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)
    plt.figure(figsize=(10, 6))
    palette = sns.color_palette(CONFIG["palette"], optimal_k)

    for i in range(optimal_k):
        plt.plot(timepoints, centroids[i], color=palette[i], linewidth=3, label=f'Cluster {i+1}')
    
    plt.title(f'Cluster Centroids for {sex.capitalize()} {tissue.capitalize()} (n={optimal_k})')
    plt.xlabel('Timepoint')
    plt.ylabel('Log Fold Change')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{tissue}_{sex}_centroids_only.png")
    plt.savefig(plot_path)
    plt.close()

def plot_individual_clusters(data_with_clusters, processed_df, centroids, output_dir, tissue, sex, optimal_k):
    """Creates an individual, consistently-scaled plot for each cluster."""
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)
    palette = sns.color_palette(CONFIG["palette"], optimal_k)

    # Calculate consistent y-axis limits across all plots, ignoring potential infs
    plot_data = data_with_clusters.copy()
    plot_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    y_min, y_max = plot_data[CONFIG["columns"]["logFC"]].min(), plot_data[CONFIG["columns"]["logFC"]].max()

    for i in range(optimal_k):
        plt.figure(figsize=(10, 6))
        cluster_data = plot_data[plot_data["cluster"] == i]
        n_features = cluster_data[CONFIG["columns"]["feature_ID"]].nunique()

        sns.lineplot(
            data=cluster_data,
            x=CONFIG["columns"]["timepoint"],
            y=CONFIG["columns"]["logFC"],
            units=CONFIG["columns"]["feature_ID"],
            estimator=None,
            color=palette[i],
            alpha=0.15,
            linewidth=0.8
        )
        
        plt.plot(processed_df.columns, centroids[i], color="black", linewidth=3, label=f'Cluster {i+1} Centroid')
        
        plt.ylim(y_min, y_max)
        plt.title(f'{tissue.capitalize()} {sex.capitalize()} - Cluster {i+1} ({n_features} features)')
        plt.xlabel('Timepoint')
        plt.ylabel('Log Fold Change')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"{tissue}_{sex}_cluster_{i+1}.png")
        plt.savefig(plot_path)
        plt.close()

def plot_cluster_centroids_array(data_with_clusters, processed_df, centroids, output_dir, tissue, sex, optimal_k):
    """Plots an array of clusters, each with its member trajectories and centroid."""
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)
    palette = sns.color_palette(CONFIG["palette"], optimal_k)

    n_cols = 3
    n_rows = (optimal_k + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharex=True, sharey=True)
    axes = axes.flatten()
    
    plot_data = data_with_clusters.copy()
    plot_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    y_min, y_max = plot_data[CONFIG["columns"]["logFC"]].min(), plot_data[CONFIG["columns"]["logFC"]].max()
    plt.ylim(y_min, y_max)

    for i in range(optimal_k):
        ax = axes[i]
        cluster_data = plot_data[plot_data["cluster"] == i]
        n_features = cluster_data[CONFIG["columns"]["feature_ID"]].nunique()

        sns.lineplot(
            data=cluster_data,
            x=CONFIG["columns"]["timepoint"],
            y=CONFIG["columns"]["logFC"],
            units=CONFIG["columns"]["feature_ID"],
            estimator=None,
            color=palette[i],
            alpha=0.1,
            linewidth=0.8,
            ax=ax
        )
        ax.plot(processed_df.columns, centroids[i], color='black', linewidth=2.5)
        ax.set_title(f"Cluster {i+1} ({n_features} features)")
        ax.set_ylabel("Log Fold Change" if i % n_cols == 0 else "")
        
        # Add x-axis labels and rotate ticks only for the bottom-most plots in each column
        if i + n_cols >= optimal_k:
            ax.set_xlabel("Timepoint")
            ax.tick_params(axis='x', labelrotation=45)
        else:
            ax.set_xlabel("")
    
    for i in range(optimal_k, len(axes)):
        axes[i].set_visible(False)
        
    fig.suptitle(f"Cluster Array for {sex.capitalize()} {tissue.capitalize()}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = os.path.join(output_dir, f"{tissue}_{sex}_cluster_array.png")
    plt.savefig(plot_path)
    plt.close()

def run_pathway_enrichment(feature_ids, labels, optimal_k, output_dir, tissue, sex):
    """
    Performs pathway enrichment analysis for each cluster.
    """
    # Prepare clustered features, ensuring the index is reset to be a column for merging.
    clustered_features_df = pd.DataFrame({
        CONFIG["columns"]["feature_ID"]: feature_ids,
        'cluster_assignment': labels
    })
    unique_clustered_features_df = clustered_features_df.drop_duplicates().reset_index(drop=True)
    print(f"\nProcessing {len(unique_clustered_features_df)} unique features for pathway enrichment.")

    # Load gene lookup data
    gene_lookup_file_path = os.path.join(CONFIG["base_dir_da"], CONFIG["pathway_enrichment_file_name"])
    gene_cols_to_use = ['feature_id', 'geneId', 'gene_name', 'short_annotation']
    try:
        print(f"\nLoading gene lookup data from: {gene_lookup_file_path}")
        gene_lookup_df = pd.read_csv(gene_lookup_file_path, sep='\t', usecols=lambda col: col in gene_cols_to_use, engine='python')
        print("\nGene lookup data loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Gene lookup file not found at {gene_lookup_file_path}")
        return {}
    except Exception as e:
        print(f"Error loading gene lookup file: {e}")
        return {}
        
    # Merge clustered features with gene data. The left DF has a 'feature_ID' column,
    # while the right DF (gene_lookup_df) has feature_id
    merged_df = pd.merge(
        unique_clustered_features_df,
        gene_lookup_df,
        left_on=CONFIG["columns"]["feature_ID"],
        right_on='feature_id',
        how='left'
    )
    print("\nMerged clustered features with gene data.")
    if 'geneId' not in merged_df.columns:
        merged_df['geneId'] = pd.NA

    # Perform pathway enrichment for each cluster
    gp = GProfiler(return_dataframe=True, user_agent='motrpac_temporal_analysis')
    enrichment_results_by_cluster = {}
    all_enrichment_dfs = []
    
    for cluster_id in sorted(merged_df['cluster_assignment'].unique()):
        print(f"\nPerforming pathway enrichment for Cluster {cluster_id}...")
        genes_in_cluster = merged_df[merged_df['cluster_assignment'] == cluster_id]['geneId'].dropna().astype(str).tolist()

        if not genes_in_cluster:
            print(f"No genes found for Cluster {cluster_id}. Skipping enrichment.")
            enrichment_results_by_cluster[cluster_id] = pd.DataFrame()
            continue
        
        print(f"Number of genes in Cluster {cluster_id}: {len(genes_in_cluster)}")
        try:
            enrichment_df = gp.profile(
                organism='rnorvegicus',
                query=genes_in_cluster,
                sources=['GO:BP', 'KEGG', 'REAC'],
                user_threshold=0.05,
                significance_threshold_method='gSCS'
            )
            
            if not enrichment_df.empty:
                print(f"Found {len(enrichment_df)} significant pathways for Cluster {cluster_id}.")
                enrichment_df['cluster'] = cluster_id
                enrichment_results_by_cluster[cluster_id] = enrichment_df
                all_enrichment_dfs.append(enrichment_df)
            else:
                print(f"No significant pathways found for Cluster {cluster_id}.")
                enrichment_results_by_cluster[cluster_id] = pd.DataFrame()

        except Exception as e:
            print(f"Error during enrichment analysis for Cluster {cluster_id}: {e}")
            enrichment_results_by_cluster[cluster_id] = pd.DataFrame()
    
    # Save results
    if all_enrichment_dfs:
        concatenated_enrichment = pd.concat(all_enrichment_dfs, ignore_index=True)
        csv_path = os.path.join(output_dir, f"{tissue}_{sex}_all_clusters_enrichment.csv")
        concatenated_enrichment.to_csv(csv_path, index=False)
        print(f"\nSaved combined enrichment results to {csv_path}")

    pickle_path = os.path.join(output_dir, f"{tissue}_{sex}_all_clusters_enrichment.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(enrichment_results_by_cluster, f)
    print(f"Saved enrichment dictionary to {pickle_path}")
        
    return enrichment_results_by_cluster

def postprocess_enrichment_results(
                enrichment_results_by_cluster,
                output_dir,
                tissue,
                sex,
                *,
                term_size_min=CONFIG.get("term_size_min", 0),
                term_size_max=CONFIG.get("term_size_max", 400),
                top_n_pathways=CONFIG.get("top_n_pathways", 75)
    ):
    """
    Filters and processes raw enrichment results based on term size and p-value.
    """
    # term_size_min = CONFIG.get("term_size_min", 0)
    # term_size_max = CONFIG.get("term_size_max", 400)
    # top_n_pathways = CONFIG.get("top_n_pathways", 75)
    
    processed_results = {}
    print("\n--- Starting Post-Processing of Enrichment Results ---")

    for cluster_id, enrichment_df in enrichment_results_by_cluster.items():
        print(f"\nProcessing Cluster {cluster_id}...")
        if enrichment_df.empty:
            print(f"  - No enrichment results to process.")
            processed_results[cluster_id] = pd.DataFrame()
            continue

        # 1. Filter by term size
        if 'term_size' not in enrichment_df.columns:
            print(f"  - Warning: 'term_size' column not found. Skipping term size filtering.")
            filtered_df = enrichment_df.copy()
        else:
            initial_count = len(enrichment_df)
            # Apply term size filter based on notebook logic
            filtered_df = enrichment_df[
                (enrichment_df['term_size'] > term_size_min) & 
                (enrichment_df['term_size'] < term_size_max)
            ].copy()
            print(f"  - Filtered by term_size ({term_size_min} < size < {term_size_max}). Kept {len(filtered_df)} of {initial_count} pathways.")

        if filtered_df.empty:
            if initial_count > 0:
                 print(f"  - Warning: Term size filtering removed all pathways.")
            processed_results[cluster_id] = pd.DataFrame()
            continue
            
        # 2. Sort by p-value and take top N
        if 'p_value' not in filtered_df.columns:
            print(f" Cluster {cluster_id}: - Warning: 'p_value' column not found. Cannot sort or select top N.")
            processed_results[cluster_id] = filtered_df
            continue

        sorted_df = filtered_df.sort_values(by='p_value', ascending=True)
        top_n_df = sorted_df.head(top_n_pathways)
        print(f"  - Selected top {len(top_n_df)} pathways by p-value (max was {top_n_pathways}).")
        
        processed_results[cluster_id] = top_n_df

    print("\n--- Post-Processing Complete ---")

    # Save the processed results dictionary as a pickle artifact
    pickle_path = os.path.join(output_dir, f"{tissue}_{sex}_all_clusters_processed_enrichment.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(processed_results, f)
    print(f"Saved processed enrichment dictionary to {pickle_path}")

    return processed_results

def plot_trajectories_with_wordclouds(data_with_clusters, processed_df, centroids, enrichment_results, output_dir, tissue, sex, optimal_k):
    """Plots cluster trajectories with word clouds of enriched pathways in a grid."""
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)
    palette = sns.color_palette(CONFIG["palette"], optimal_k)
    
    n_rows = optimal_k
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4.5 * n_rows), gridspec_kw={'width_ratios': [2, 1]})

    plot_data = data_with_clusters.copy()
    plot_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    y_min, y_max = plot_data[CONFIG["columns"]["logFC"]].min(), plot_data[CONFIG["columns"]["logFC"]].max()

    for i in range(optimal_k):
        ax_traj = axes[i, 0] if n_rows > 1 else axes[0]
        ax_wc = axes[i, 1] if n_rows > 1 else axes[1]
        
        # Trajectory plot
        cluster_data = plot_data[plot_data["cluster"] == i]
        n_features = cluster_data[CONFIG["columns"]["feature_ID"]].nunique()

        sns.lineplot(
            data=cluster_data,
            x=CONFIG["columns"]["timepoint"],
            y=CONFIG["columns"]["logFC"],
            units=CONFIG["columns"]["feature_ID"],
            estimator=None,
            color=palette[i],
            alpha=0.1,
            linewidth=0.8,
            ax=ax_traj
        )
        ax_traj.plot(processed_df.columns, centroids[i], color='black', linewidth=2.5)
        ax_traj.set_title(f"Cluster {i+1} ({n_features} features)")
        ax_traj.set_ylim(y_min, y_max)
        ax_traj.set_ylabel("Log Fold Change")
        
        # Add x-axis labels and rotate ticks only for the bottom-most plot
        if i == n_rows - 1:
            ax_traj.set_xlabel("Timepoint")
            ax_traj.tick_params(axis='x', labelrotation=45)
        else:
            ax_traj.set_xlabel("")
        
        # Word cloud
        ax_wc.axis("off")
        if i in enrichment_results and not enrichment_results[i].empty:
            terms = enrichment_results[i]['name'].dropna().tolist()
            text = " ".join(terms)
            if text:
                wordcloud = WordCloud(
                    background_color="white",
                    width=400,
                    height=300,
                    colormap=CONFIG["palette"],
                    max_words=30,
                    contour_width=1,
                    contour_color='steelblue',
                    max_font_size=100,
                    # random_state=42
                ).generate(text)
                ax_wc.imshow(wordcloud, interpolation='bilinear')
                ax_wc.set_title(f"Top Enriched Pathways ({len(terms)})")

    fig.suptitle(f"Cluster Analysis for {sex.capitalize()} {tissue.capitalize()}: Trajectories and Pathway Enrichment", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plot_path = os.path.join(output_dir, f"{tissue}_{sex}_trajectories_with_wordclouds.png")
    plt.savefig(plot_path)
    plt.close()

def main():
    """
    Main function to run the temporal clustering analysis pipeline.
    """
    output_dir = get_output_dir(CONFIG["tissue"])
    
    for sex in ["male", "female"]:
        data_filtered = load_and_preprocess_data(
            CONFIG["deseq_file_name"],
            CONFIG["significance_threshold"],
            sex
        )
        print(f"Processing {sex} {CONFIG['tissue']} data...")
        print(f"Data shape: {data_filtered.shape}")

        if data_filtered.empty:
            print(f"No significant features for {sex}. Skipping.")
            continue
            
        all_clustering_results, optimal_k, processed_df = find_optimal_clusters(
            data_filtered,
            output_dir,
            CONFIG["tissue"],
            sex
        )

        optimal_result = all_clustering_results[optimal_k]
        labels = optimal_result["labels"]
        centroids = optimal_result["centroids"]

        # Create a mapping from feature ID to cluster and merge it into the long-format data
        feature_to_cluster_map = pd.Series(labels, index=processed_df.index, name='cluster')
        data_with_clusters = data_filtered.merge(feature_to_cluster_map, left_on=CONFIG["columns"]["feature_ID"], right_index=True)

        plot_all_feature_trajectories(data_filtered, output_dir, CONFIG["tissue"], sex)
        plot_clusters_with_centroids(data_with_clusters, processed_df, centroids, output_dir, CONFIG["tissue"], sex, optimal_k)
        plot_only_centroids(centroids, processed_df.columns, output_dir, CONFIG["tissue"], sex, optimal_k)
        plot_individual_clusters(data_with_clusters, processed_df, centroids, output_dir, CONFIG["tissue"], sex, optimal_k)
        plot_cluster_centroids_array(data_with_clusters, processed_df, centroids, output_dir, CONFIG["tissue"], sex, optimal_k)

        enrichment_results = run_pathway_enrichment(processed_df.index, labels, optimal_k, output_dir, CONFIG["tissue"], sex)
        
        if enrichment_results:
            
            term_size_min = CONFIG.get("term_size_min", 0)
            term_size_max = CONFIG.get("term_size_max", 400)
            top_n_pathways = CONFIG.get("top_n_pathways", 75)

            processed_enrichment_results = postprocess_enrichment_results(
                enrichment_results,
                output_dir,
                CONFIG["tissue"],
                sex,
                term_size_min=term_size_min,
                term_size_max=term_size_max,
                top_n_pathways=top_n_pathways
            )

            plot_trajectories_with_wordclouds(
                data_with_clusters,
                processed_df,
                centroids,
                processed_enrichment_results,
                output_dir,
                CONFIG["tissue"],
                sex,
                optimal_k
            )

if __name__ == "__main__":
    main()