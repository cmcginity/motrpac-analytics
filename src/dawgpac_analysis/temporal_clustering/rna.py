import io
import os
import pickle
import gc
import psutil
import pandas as pd
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import skfuzzy as fuzz
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from collections import defaultdict, Counter
from gprofiler import GProfiler
from wordcloud import WordCloud
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from datetime import date, datetime

from ..google_utils import GoogleCloudHelper

RUN_DATE = datetime.now().strftime('%Y-%m-%d-%H%M')

print(f"Polars version: {pl.__version__}")

def get_output_paths(local_base_dir, artifact_type, tissue, sex, slug, ext):
    """Generates standardized local path and a clean cloud filename."""
    drive_filename = f"{RUN_DATE}_{tissue}_{sex}_{slug}.{ext}"
    local_path = os.path.join(
        local_base_dir,
        artifact_type,
        RUN_DATE,
        drive_filename,
    )
    return local_path, drive_filename

def load_and_preprocess_data(file_buffer, config):
    """
    Loads, parses, and preprocesses the input data from a buffer using Polars.
    Does NOT perform significance filtering.
    """
    print("Loading and parsing data from buffer.")

    p_val_col = config["columns"]["p_value_adj"]
    logFC_col = config["columns"]["logFC"]
    sex_col = config["columns"]["sex"]
    feature_col = config["columns"]["feature_id"]
    significance_threshold = config['analysis_params']['significance_threshold']
    
    try:
        # Use Polars for fast, multi-threaded CSV reading from in-memory buffer.
        file_buffer.seek(0)
        df_pl = pl.read_csv(file_buffer, separator="\t", null_values="NA")

        # 1. Standardize data types FIRST. This is crucial for filtering.
        df_pl = df_pl.with_columns(
            pl.col(p_val_col).cast(pl.Float64, strict=False),
            pl.col(logFC_col).cast(pl.Float64, strict=False)
        ).drop_nulls(subset=[p_val_col, logFC_col])

        # 2. Derive timepoint from the 'contrast' column.
        df_pl = df_pl.with_columns(
            pl.col("contrast") # acute_00.0h_IPE-control_00.0h_IPE.male
            .str.split("-")
            .list.get(0)
            .str.split("acute_")
            .list.get(1)
            .alias("timepoint")
        ) 
        
        print(f"  - Parsed initial dataframe with shape: {df_pl.shape}")
        return df_pl

    except Exception as e:
        print(f"An error occurred during data loading with Polars: {e}")
        # Return an empty DataFrame to allow the pipeline to continue gracefully
        return pl.DataFrame()

def xie_beni_index(
    data: np.ndarray,
    centers: np.ndarray,
    membership_matrix: np.ndarray,
    m: float,
) -> float:
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

def run_cmeans_for_k(processed_df, k, config, precomputed_seed=None):
    """
    Runs c-means for a single k over multiple seeds and returns the best result.
    If a precomputed_seed is provided, it will use that seed directly.
    """
    data_values = processed_df.values
    data_t = data_values.T
    m = config["analysis_params"]["clustering"].get("fuzziness", 1.2)

    # If a precomputed seed is provided, use it. Otherwise, search.
    if precomputed_seed is not None:
        seeds_to_test = [precomputed_seed]
        print(f"  - Using precomputed seed {precomputed_seed} for k={k}")
    else:
        # Generate a diverse and comprehensive list of seeds to test
        num_random = config["analysis_params"]["clustering"].get("num_random_seeds", 10)
        explicit = config["analysis_params"]["clustering"].get("explicit_seeds", [])
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

def plot_clustering_metrics(metrics_df, output_dir, tissue, sex, config):
    """Plots the Xie-Beni index and FPC vs. the number of clusters."""
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Xie-Beni Index
    ax1.plot(metrics_df['n_clusters'], metrics_df['xie_beni_index'], 'o-')
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Xie-Beni Index')
    ax1.set_title(f'Xie-Beni Index (m={config["analysis_params"]["clustering"]["fuzziness"]})')
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

def find_optimal_clusters(data_filtered, output_dir, tissue, sex, config, g_helper=None, drive_artifact_folder_id=None):
    """
    Finds the optimal number of clusters by evaluating metrics across a range of k values.
    """
    processed_df = preprocess_data_for_clustering(data_filtered, config)
    
    # Check if we should use precomputed seeds
    use_precomputed = config["analysis_params"]["clustering"].get("use_precomputed_seeds", False)
    precomputed_seeds = {}
    if use_precomputed:
        seed_file_path = os.path.join(output_dir, f"{tissue}_{sex}_best_seeds.pkl")
        try:
            # TODO: Add logic to download from GDrive if not found locally
            with open(seed_file_path, "rb") as f:
                precomputed_seeds = pickle.load(f)
            print(f"--> Loaded precomputed seeds from {seed_file_path}")
        except FileNotFoundError:
            print(f"--> WARNING: 'use_precomputed_seeds' is True but file not found: {seed_file_path}. Seed search will be performed instead.")
            use_precomputed = False  # Fallback to searching

    metrics = []
    all_results = {}
    
    cluster_range = list(range(
        config['analysis_params']['clustering']['k_start'],
        config['analysis_params']['clustering']['k_end'] + 1
    ))

    print(f"Starting clustering search for {sex}...")
    for k in cluster_range:
        seed_for_k = precomputed_seeds.get(k) if use_precomputed else None
        best_result_for_k = run_cmeans_for_k(
            processed_df,
            k,
            config,
            precomputed_seed=seed_for_k,
        )
        metrics.append({
            'n_clusters': k,
            'xie_beni_index': best_result_for_k['xb_index'],
            'fpc': best_result_for_k['fpc']
        })
        all_results[k] = best_result_for_k

    metrics_df = pd.DataFrame(metrics)
    
    # Plot clustering metrics
    # This still saves locally only, needs refactoring if cloud upload is desired for this plot
    plot_clustering_metrics(metrics_df, output_dir, tissue, sex, config)

    # Determine optimal k, allowing for a manual override from the config
    source = "Xie-Beni Index"
    optimal_k = int(metrics_df.loc[metrics_df['xie_beni_index'].idxmin()]['n_clusters'])

    explicit_override = config["analysis_params"]["clustering"].get("explicit_cluster_choice", {}).get(sex, {}).get(tissue, {})
    if explicit_override:
        if explicit_override in all_results:
            optimal_k = explicit_override
            source = "explicit choice"
        else:
            print(f"--> WARNING: Manual k={explicit_override} for {tissue} {sex} not in cluster range {cluster_range}. Using automatic value.")

    print(f"--> Optimal number of clusters for {sex} based on {source}: {optimal_k}")

    # --- Save artifacts ---
    # 1. Optimal Clustering Result
    optimal_result = all_results[optimal_k]
    optimal_result_bytes = pickle.dumps(optimal_result)
    if output_dir:
        optimal_local_path = os.path.join(output_dir, f"{tissue}_{sex}_optimal_clustering.pkl")
        with open(optimal_local_path, "wb") as f:
            f.write(optimal_result_bytes)
        print(f"  - Saved optimal clustering results to {optimal_local_path}")

    if g_helper and drive_artifact_folder_id:
        optimal_drive_filename = f"{RUN_DATE}_{tissue}_{sex}_optimal_clustering.pkl"
        g_helper.upload_buffer_to_drive(
            io.BytesIO(optimal_result_bytes),
            drive_artifact_folder_id,
            optimal_drive_filename,
            'application/octet-stream'
        )

    # 2. Best Seeds
    best_seeds = {k: res["seed"] for k, res in all_results.items()}
    best_seeds_bytes = pickle.dumps(best_seeds)
    if output_dir:
        seeds_local_path = os.path.join(output_dir, f"{tissue}_{sex}_best_seeds.pkl")
        with open(seeds_local_path, "wb") as f:
            f.write(best_seeds_bytes)
        print(f"  - Saved best seeds for each k to {seeds_local_path}")

    if g_helper and drive_artifact_folder_id:
        seeds_drive_filename = f"{RUN_DATE}_{tissue}_{sex}_best_seeds.pkl"
        g_helper.upload_buffer_to_drive(
            io.BytesIO(best_seeds_bytes),
            drive_artifact_folder_id,
            seeds_drive_filename,
            'application/octet-stream'
        )
        
    return all_results, optimal_k, processed_df

def preprocess_data_for_clustering(df, config):
    """
    Transforms dataframe into format for clustering: each feature_ID becomes a row with timepoint values.
    """
    feature_col = config["columns"]["feature_id"]
    log2fc_col = config["columns"]["logFC"]
    timepoint_col = config["columns"]["timepoint"]
    
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

def plot_all_feature_trajectories(data_filtered, config, tissue, sex, local_path=None, g_helper=None, drive_folder_id=None, drive_filename=None):
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
    metric_col = config["columns"]["logFC"]
    feature_col = config["columns"]["feature_id"]
    time_col = config["columns"]["timepoint"]
    
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

    if local_path:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        plt.savefig(
            local_path,
            dpi=config['plotting']['dpi'],
            bbox_inches='tight',
        )
        print(f"  - Saved plot locally to {local_path}")

    plot_file_obj = None
    if g_helper and drive_folder_id and drive_filename:
        buffer = io.BytesIO()
        plt.savefig(
            buffer,
            format='png',
            dpi=config['plotting']['dpi'],
            bbox_inches='tight',
        )
        buffer.seek(0)
        plot_file_obj = g_helper.upload_buffer_to_drive(
            buffer,
            drive_folder_id,
            drive_filename,
            'image/png',
        )
    
    plt.close(fig)
    return plot_file_obj

def plot_clusters_with_centroids(data_with_clusters, processed_df, centroids, optimal_k, config, tissue, sex, local_path=None, g_helper=None, drive_folder_id=None, drive_filename=None):
    """Plots all feature trajectories colored by cluster with centroids on a single plot."""
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)
    plt.figure(figsize=(12, 8))

    # Replace inf with NaN for plotting compatibility to avoid warnings
    plot_data = data_with_clusters.copy()
    plot_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    palette = sns.color_palette(config.get("palette", "viridis"), optimal_k)
    
    sns.lineplot(
        data=plot_data,
        x=config["columns"]["timepoint"],
        y=config["columns"]["logFC"],
        hue="cluster",
        units=config["columns"]["feature_id"],
        estimator=None,
        alpha=0.15,
        palette=palette
    )
    
    for i in range(optimal_k):
        plt.plot(
            processed_df.columns,
            centroids[i],
            color=palette[i],
            linewidth=4,
            label=f"Cluster {i+1} Centroid",
        )

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

    plt.tight_layout()

    if local_path:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        plt.savefig(
            local_path,
            dpi=config['plotting']['dpi'],
            bbox_inches='tight',
        )
        print(f"  - Saved plot locally to {local_path}")

    plot_file_obj = None
    if g_helper and drive_folder_id and drive_filename:
        buffer = io.BytesIO()
        plt.savefig(
            buffer,
            format='png',
            dpi=config['plotting']['dpi'],
            bbox_inches='tight',
        )
        buffer.seek(0)
        plot_file_obj = g_helper.upload_buffer_to_drive(
            buffer,
            drive_folder_id,
            drive_filename,
            'image/png',
        )
    
    plt.close()
    return plot_file_obj

def plot_only_centroids(centroids, timepoints, optimal_k, config, tissue, sex, local_path=None, g_helper=None, drive_folder_id=None, drive_filename=None):
    """Plots only the cluster centroids with professional styling."""
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)
    plt.figure(figsize=(10, 6))
    palette = sns.color_palette(config.get("palette", "viridis"), optimal_k)

    for i in range(optimal_k):
        plt.plot(
            timepoints,
            centroids[i],
            color=palette[i],
            linewidth=3,
            label=f'Cluster {i+1}',
        )
    
    plt.title(f'Cluster Centroids for {sex.capitalize()} {tissue.capitalize()} (n={optimal_k})')
    plt.xlabel('Timepoint')
    plt.ylabel('Log Fold Change')
    plt.legend()
    plt.xticks(rotation=45)
    
    plt.tight_layout()

    if local_path:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        plt.savefig(
            local_path,
            dpi=config['plotting']['dpi'],
            bbox_inches='tight',
        )
        print(f"  - Saved plot locally to {local_path}")

    plot_file_obj = None
    if g_helper and drive_folder_id and drive_filename:
        buffer = io.BytesIO()
        plt.savefig(
            buffer,
            format='png',
            dpi=config['plotting']['dpi'],
            bbox_inches='tight',
        )
        buffer.seek(0)
        plot_file_obj = g_helper.upload_buffer_to_drive(
            buffer,
            drive_folder_id,
            drive_filename,
            'image/png',
        )
    
    plt.close()
    return plot_file_obj

def plot_individual_clusters(data_with_clusters, processed_df, centroids, optimal_k, config, tissue, sex, local_path_prefix=None, g_helper=None, drive_folder_id=None, drive_filename_prefix=None):
    """Creates an individual, consistently-scaled plot for each cluster."""
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)
    palette = sns.color_palette(config.get("palette", "viridis"), optimal_k)

    # Calculate consistent y-axis limits across all plots, ignoring potential infs
    plot_data = data_with_clusters.copy()
    plot_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    y_min, y_max = plot_data[config["columns"]["logFC"]].min(), plot_data[config["columns"]["logFC"]].max()

    for i in range(optimal_k):
        plt.figure(figsize=(10, 6))
        cluster_data = plot_data[plot_data["cluster"] == i]
        n_features = cluster_data[config["columns"]["feature_id"]].nunique()

        sns.lineplot(
            data=cluster_data,
            x=config["columns"]["timepoint"],
            y=config["columns"]["logFC"],
            units=config["columns"]["feature_id"],
            estimator=None,
            color=palette[i],
            alpha=0.15,
            linewidth=0.8
        )
        
        plt.plot(
            processed_df.columns,
            centroids[i],
            color="black",
            linewidth=3,
            label=f'Cluster {i+1} Centroid',
        )
        
        plt.ylim(y_min, y_max)
        plt.title(f'{tissue.capitalize()} {sex.capitalize()} - Cluster {i+1} ({n_features} features)')
        plt.xlabel('Timepoint')
        plt.ylabel('Log Fold Change')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        if local_path_prefix:
            local_path = f"{local_path_prefix}{i+1}.png"
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            plt.savefig(
                local_path,
                dpi=config['plotting']['dpi'],
                bbox_inches='tight',
            )
            print(f"  - Saved plot locally to {local_path}")

        if g_helper and drive_folder_id and drive_filename_prefix:
            drive_filename = f"{drive_filename_prefix}{i+1}.png"
            buffer = io.BytesIO()
            plt.savefig(
                buffer,
                format='png',
                dpi=config['plotting']['dpi'],
                bbox_inches='tight',
            )
            buffer.seek(0)
            g_helper.upload_buffer_to_drive(
                buffer,
                drive_folder_id,
                drive_filename,
                'image/png',
            )
        
        plt.close()

def plot_cluster_centroids_array(data_with_clusters, processed_df, centroids, optimal_k, config, tissue, sex, local_path=None, g_helper=None, drive_folder_id=None, drive_filename=None):
    """Plots an array of clusters, each with its member trajectories and centroid."""
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)
    palette = sns.color_palette(config.get("palette", "viridis"), optimal_k)

    n_cols = 3
    n_rows = (optimal_k + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        sharex=True,
        sharey=True,
    )
    axes = axes.flatten()
    
    plot_data = data_with_clusters.copy()
    plot_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    y_min, y_max = plot_data[config["columns"]["logFC"]].min(), plot_data[config["columns"]["logFC"]].max()
    plt.ylim(y_min, y_max)

    for i in range(optimal_k):
        ax = axes[i]
        cluster_data = plot_data[plot_data["cluster"] == i]
        n_features = cluster_data[config["columns"]["feature_id"]].nunique()

        sns.lineplot(
            data=cluster_data,
            x=config["columns"]["timepoint"],
            y=config["columns"]["logFC"],
            units=config["columns"]["feature_id"],
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

    if local_path:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        plt.savefig(
            local_path,
            dpi=config['plotting']['dpi'],
            bbox_inches='tight',
        )
        print(f"  - Saved plot locally to {local_path}")

    plot_file_obj = None
    if g_helper and drive_folder_id and drive_filename:
        buffer = io.BytesIO()
        plt.savefig(
            buffer,
            format='png',
            dpi=config['plotting']['dpi'],
            bbox_inches='tight',
        )
        buffer.seek(0)
        plot_file_obj = g_helper.upload_buffer_to_drive(
            buffer,
            drive_folder_id,
            drive_filename,
            'image/png',
        )
    
    plt.close()
    return plot_file_obj

def run_pathway_enrichment(feature_ids, labels, optimal_k, gene_lookup_buffer, config, tissue, sex, local_artifact_dir=None, g_helper=None, drive_artifact_folder_id=None, export_local_csv=False):
    """
    Performs pathway enrichment analysis for each cluster.
    """
    # Prepare clustered features, ensuring the index is reset to be a column for merging.
    clustered_features_df = pd.DataFrame({
        config["columns"]["feature_id"]: feature_ids,
        'cluster_assignment': labels
    })
    unique_clustered_features_df = clustered_features_df.drop_duplicates().reset_index(drop=True)
    print(f"\nProcessing {len(unique_clustered_features_df)} unique features for pathway enrichment.")

    # Load gene lookup data
    gene_cols_to_use = ['feature_id', 'gene_symbol']
    try:
        print(f"\nLoading gene lookup data from buffer.")
        gene_lookup_buffer.seek(0)
        gene_lookup_df = pd.read_csv(
            gene_lookup_buffer,
            sep='\t',
            usecols=lambda col: col in gene_cols_to_use,
            engine='python',
        )
        print("\nGene lookup data loaded successfully.")

        # Filter out rows where gene_symbol is just a repeat of feature_id
        initial_count = len(gene_lookup_df)
        gene_lookup_df = gene_lookup_df[gene_lookup_df['feature_id'] != gene_lookup_df['gene_symbol']]
        print(f"  - Filtered out {initial_count - len(gene_lookup_df)} rows where gene_symbol matched feature_id.")

    except FileNotFoundError:
        print(f"Error: Gene lookup file not found.")
        return {}
    except Exception as e:
        print(f"Error loading gene lookup file: {e}")
        return {}
        
    # Merge clustered features with gene data. The left DF has a 'feature_ID' column,
    # while the right DF (gene_lookup_df) has feature_id
    merged_df = pd.merge(
        unique_clustered_features_df,
        gene_lookup_df,
        left_on=config["columns"]["feature_id"],
        right_on='feature_id',
        how='left'
    )
    print("\nMerged clustered features with gene data.")
    if 'gene_symbol' not in merged_df.columns:
        merged_df['gene_symbol'] = pd.NA

    # Save merged df with gene info
    if local_artifact_dir and export_local_csv:
        csv_local_path = os.path.join(
            local_artifact_dir,
            f"{tissue}_{sex}_all_clusters_genes.csv"
        )
        merged_df.to_csv(csv_local_path, index=False)
        print(f"\nSaved cluster genes to {csv_local_path}")
        
    if g_helper and drive_artifact_folder_id:
        drive_filename = f"{RUN_DATE}_{tissue}_{sex}_all_clusters_genes.csv"
        buffer = io.StringIO()
        merged_df.to_csv(buffer, index=False)
        buffer.seek(0)
        g_helper.upload_buffer_to_drive(
            io.BytesIO(buffer.read().encode('utf-8')),
            drive_artifact_folder_id,
            drive_filename,
            'text/csv'
        )

    # Perform pathway enrichment for each cluster
    gp = GProfiler(return_dataframe=True, user_agent='motrpac_temporal_analysis')
    enrichment_results_by_cluster = {}
    all_enrichment_dfs = []
    
    for cluster_id in sorted(merged_df['cluster_assignment'].unique()):
        print(f"\nPerforming pathway enrichment for Cluster {cluster_id}...")
        genes_in_cluster = merged_df[merged_df['cluster_assignment'] == cluster_id]['gene_symbol'].dropna().astype(str).tolist()

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
        if local_artifact_dir and export_local_csv:
            csv_local_path = os.path.join(
                local_artifact_dir,
                f"{tissue}_{sex}_all_clusters_enrichment.csv"
            )
            concatenated_enrichment.to_csv(csv_local_path, index=False)
            print(f"\nSaved combined enrichment results to {csv_local_path}")

        if g_helper and drive_artifact_folder_id:
            drive_filename = f"{RUN_DATE}_{tissue}_{sex}_all_clusters_enrichment.csv"
            buffer = io.StringIO()
            concatenated_enrichment.to_csv(buffer, index=False)
            buffer.seek(0)
            g_helper.upload_buffer_to_drive(
                io.BytesIO(buffer.read().encode('utf-8')),
                drive_artifact_folder_id,
                drive_filename,
                'text/csv'
            )

    enrichment_results_bytes = pickle.dumps(enrichment_results_by_cluster)
    if local_artifact_dir:
        pickle_local_path = os.path.join(
            local_artifact_dir,
            f"{tissue}_{sex}_all_clusters_enrichment.pkl",
        )
        with open(pickle_local_path, "wb") as f:
            f.write(enrichment_results_bytes)
        print(f"Saved enrichment dictionary to {pickle_local_path}")

    if g_helper and drive_artifact_folder_id:
        drive_filename = f"{RUN_DATE}_{tissue}_{sex}_all_clusters_enrichment.pkl"
        g_helper.upload_buffer_to_drive(
            io.BytesIO(enrichment_results_bytes),
            drive_artifact_folder_id,
            drive_filename,
            'application/octet-stream',
        )
        
    return enrichment_results_by_cluster, merged_df

def postprocess_enrichment_results(
                enrichment_results_by_cluster,
                tissue,
                sex,
                local_artifact_dir=None, 
                g_helper=None, 
                drive_artifact_folder_id=None,
                *,
                term_size_min=0,
                term_size_max=400,
                top_n_pathways=75
    ):
    """
    Filters and processes raw enrichment results based on term size and p-value.
    """
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
    processed_results_bytes = pickle.dumps(processed_results)
    if local_artifact_dir:
        pickle_local_path = os.path.join(local_artifact_dir, f"{tissue}_{sex}_all_clusters_processed_enrichment.pkl")
        with open(pickle_local_path, "wb") as f:
            f.write(processed_results_bytes)
        print(f"Saved processed enrichment dictionary to {pickle_local_path}")

    if g_helper and drive_artifact_folder_id:
        drive_filename = f"{RUN_DATE}_{tissue}_{sex}_all_clusters_processed_enrichment.pkl"
        g_helper.upload_buffer_to_drive(
            io.BytesIO(processed_results_bytes),
            drive_artifact_folder_id,
            drive_filename,
            'application/octet-stream'
        )

    return processed_results

def plot_trajectories_with_wordclouds(data_with_clusters, processed_df, centroids, enrichment_results, merged_df, optimal_k, config, tissue, sex, local_path=None, g_helper=None, drive_folder_id=None, drive_filename=None):
    """Plots cluster trajectories with word clouds of enriched pathways in a grid."""
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)
    palette = sns.color_palette(config.get("palette", "viridis"), optimal_k)
    
    n_rows = optimal_k
    fig, axes = plt.subplots(
        n_rows,
        3,
        figsize=(18, 4.5 * n_rows),
        gridspec_kw={'width_ratios': [2, 1, 1]}
    )

    plot_data = data_with_clusters.copy()
    plot_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    y_min, y_max = plot_data[config["columns"]["logFC"]].min(), plot_data[config["columns"]["logFC"]].max()

    for i in range(optimal_k):
        ax_traj = axes[i, 0] if n_rows > 1 else axes[0]
        ax_gene_wc = axes[i, 1] if n_rows > 1 else axes[1]
        ax_pathway_wc = axes[i, 2] if n_rows > 1 else axes[2]
        
        # Trajectory plot
        cluster_data = plot_data[plot_data["cluster"] == i]
        n_features = cluster_data[config["columns"]["feature_id"]].nunique()

        sns.lineplot(
            data=cluster_data,
            x=config["columns"]["timepoint"],
            y=config["columns"]["logFC"],
            units=config["columns"]["feature_id"],
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
        
        # Gene Word cloud
        ax_gene_wc.axis("off")
        genes_for_cluster = merged_df[merged_df['cluster_assignment'] == i]
        gene_names = genes_for_cluster['gene_symbol'].replace('NA', np.nan).dropna().str.strip().tolist()
        if gene_names:
            gene_freq = Counter(gene_names)
            
            # Determine frequency range for color mapping
            if gene_freq:
                max_freq = max(gene_freq.values())
                min_freq = min(gene_freq.values())
            else:
                max_freq, min_freq = 1, 1

            # Custom color function to map frequency to a color from the 'Blues' colormap
            cmap = plt.get_cmap(config.get("palette", "Blues"))
            def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
                if max_freq == min_freq:
                    norm_freq = 0.5  # Mid-range color for uniform frequencies
                else:
                    norm_freq = (gene_freq.get(word, 0) - min_freq) / (max_freq - min_freq)
                
                # Scale to use a more visible part of the colormap (avoiding pure white)
                scaled_norm_freq = 0.2 + 0.8 * norm_freq
                
                color = cmap(scaled_norm_freq)
                # WordCloud expects RGB tuples of integers (0-255)
                return tuple(int(c * 255) for c in color[:3])

            wordcloud = WordCloud(
                background_color="white",
                width=400,
                height=300,
                color_func=color_func,
                max_words=30,
                contour_width=1,
                contour_color='steelblue',
                max_font_size=100,
            ).generate_from_frequencies(gene_freq)
            ax_gene_wc.imshow(wordcloud, interpolation='bilinear')
            ax_gene_wc.set_title(f"Top Genes ({len(gene_freq)})")
            
        # Pathway Word cloud
        ax_pathway_wc.axis("off")
        if i in enrichment_results and not enrichment_results[i].empty:
            terms = enrichment_results[i]['name'].dropna().tolist()
            # remove low signal words from the list
            low_signal_words = [
            #     "acid",
            #     "anatomical",
            #     "biological",
            #     "biosynthetic",
            #     "catabolic",
            #     "cell",
            #     "cellular",
            #     "communication",
            #     "compound",
            #     "development",
            #     "developmental",
            #     "differentiation",
            #     "expression",
            #     "localization",
            #     "macromolecule",
            #     "metabolic",
            #     "metabolism",
            #     "molecule",
            #     "negative",
            #     "organismal",
            #     "pathway",
            #     "pathways",
            #     "positive",
            #     "process",
            #     "processes",
            #     "quality",
            #     "regulating",
            #     "regulation",
            #     "regulatory",
            #     "response",
            #     "signal",
            #     "signaling",
            #     "small",
            #     "stimulus",
            #     "structure",
            #     "system",
            #     "transduction"
            ]
            text = " ".join(terms)
            # remove any words that appear in the list of low signal words
            text = " ".join([term for term in text.split(" ") if term not in low_signal_words])
            if text:
                wordcloud = WordCloud(
                    background_color="white",
                    width=400,
                    height=300,
                    colormap=config.get("palette", "viridis"),
                    max_words=30,
                    contour_width=1,
                    contour_color='steelblue',
                    max_font_size=100,
                    # random_state=42
                ).generate(text)
                ax_pathway_wc.imshow(wordcloud, interpolation='bilinear')
                ax_pathway_wc.set_title(f"Top Enriched Pathways ({len(terms)})")

    fig.suptitle(f"Cluster Analysis for {sex.capitalize()} {tissue.capitalize()}: Trajectories and Pathway Enrichment", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    if local_path:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        plt.savefig(
            local_path,
            dpi=config['plotting']['dpi'],
            bbox_inches='tight'
        )
        print(f"  - Saved plot locally to {local_path}")

    plot_file_obj = None
    if g_helper and drive_folder_id and drive_filename:
        buffer = io.BytesIO()
        plt.savefig(
            buffer,
            format='png',
            dpi=config['plotting']['dpi'],
            bbox_inches='tight'
        )
        buffer.seek(0)
        plot_file_obj = g_helper.upload_buffer_to_drive(
            buffer,
            drive_folder_id,
            drive_filename,
            'image/png'
        )
    
    plt.close()
    return plot_file_obj

def main(config):
    """
    Main function to run the RNA-seq temporal clustering pipeline.
    """
    print("Starting RNA-seq Temporal Clustering Pipeline...")
    
    # Read the new project ID from the config dictionary
    gcs_project_id = config.get('personal_gcp_project_id')
    if not gcs_project_id:
        raise ValueError("The 'personal_gcp_project_id' must be set in your configuration (e.g., defaults.yaml)")

    # Pass the project ID when initializing the helper class
    g_helper = GoogleCloudHelper(gcs_quota_project_id=gcs_project_id)

    # Create top-level output folder in Google Drive
    drive_run_folder_id = g_helper.create_drive_folder(
        f"{RUN_DATE}_transcript-rna-seq_temporal_clustering",
        config['drive']['root_output_folder_id']
    )
    print(f"Created main output folder in Google Drive with ID: {drive_run_folder_id}")

    plot_links_for_slides = {}
    local_base_dir = config.get("local_base_output_dir", "output")
    export_local_figures_and_csv = config.get("export_local_output", False)
    print(f"Local output directory: {local_base_dir}")
    print(f"Export figures and CSVs locally: {export_local_figures_and_csv}")

    print(f"Polars version: {pl.__version__}")

    for tissue, files in config['tissue_files'].items():
        print(f"\n--- Processing tissue: {tissue.upper()} ---")

        # Download data from GCS into memory buffers
        print("Downloading data from Google Cloud Storage...")
        try:
            da_gcs_path = os.path.join(config['gcs_paths']['da_data'], files['da_file'])
            da_buffer = g_helper.download_gcs_file_as_stringio(config['gcs_bucket_name'], da_gcs_path)
            
            feature_map_gcs_path = os.path.join(config['gcs_paths']['feature_map_data'], files['feature_map_file'])
            feature_map_buffer = g_helper.download_gcs_file_as_stringio(config['gcs_bucket_name'], feature_map_gcs_path)
            print("Data downloaded successfully.")
        except Exception as e:
            print(f"Error downloading data from GCS: {e}")
            continue

        # Load and preprocess data ONCE per tissue. No significance filtering yet.
        full_df_pl = load_and_preprocess_data(da_buffer, config)
        
        # Clean up buffer immediately
        del da_buffer
        gc.collect()

        if full_df_pl.is_empty():
            print(f"No data loaded for tissue {tissue}. Skipping.")
            continue
        
        # Define columns and threshold for use in the loop
        sex_col = config["columns"]["sex"]
        p_val_col = config["columns"]["p_value_adj"]
        feature_col = config["columns"]["feature_id"]
        significance_threshold = config['analysis_params']['significance_threshold']
        
        datasets_to_process = [
            (full_df_pl.lazy().filter(pl.col(sex_col) == "male"), "male"),
            (full_df_pl.lazy().filter(pl.col(sex_col) == "female"), "female")
        ]

        for sex_lazy_df, sex in datasets_to_process:
            print(f"\n--- Processing sex: {sex.upper()} ---")
            
            # Apply the significance filter to the sex-specific lazy frame
            sex_specific_filtered_lazy_df = sex_lazy_df.filter(
                pl.col(p_val_col).min().over(feature_col) < significance_threshold
            )

            # Execute the lazy query and get a concrete Polars DataFrame
            sex_specific_df_pl = sex_specific_filtered_lazy_df.collect()

            if sex_specific_df_pl.is_empty():
                print(f"No significant features for {sex} in {tissue}. Skipping.")
                continue
            
            # Convert to pandas for the rest of the pipeline
            data_filtered = sex_specific_df_pl.to_pandas()
            del sex_specific_df_pl
            gc.collect()
            
            print(f"  - Found {data_filtered[feature_col].nunique()} significant features for {sex}.")
            
            drive_fig_folder_id = g_helper.create_drive_folder(f"{tissue}_{sex}_figures", drive_run_folder_id)
            drive_artifact_folder_id = g_helper.create_drive_folder(f"{tissue}_{sex}_artifacts", drive_run_folder_id)

            output_dir = None
            if local_base_dir:
                output_dir = os.path.join(local_base_dir, RUN_DATE, tissue, sex)
                os.makedirs(output_dir, exist_ok=True)
                print(f"Output will be saved to: {output_dir}")
            
            all_clustering_results, optimal_k, processed_df = find_optimal_clusters(
                data_filtered,
                output_dir,
                tissue,
                sex,
                config,
                g_helper=g_helper,
                drive_artifact_folder_id=drive_artifact_folder_id,
            )

            optimal_result = all_clustering_results[optimal_k]
            labels = optimal_result["labels"]
            centroids = optimal_result["centroids"]

            feature_to_cluster_map = pd.Series(labels, index=processed_df.index, name='cluster')
            data_with_clusters = data_filtered.merge(feature_to_cluster_map, left_on=config["columns"]["feature_id"], right_index=True)
            
            # --- Plotting calls ---
            
            # Plot 1: All trajectories
            local_plot_path, drive_plot_filename = get_output_paths(
                local_base_dir,
                'figures',
                tissue,
                sex,
                'all_trajectories',
                'png'
            )
            plot_file_obj = plot_all_feature_trajectories(
                data_filtered,
                config,
                tissue,
                sex,
                local_path=local_plot_path if export_local_figures_and_csv else None,
                g_helper=g_helper,
                drive_folder_id=drive_fig_folder_id,
                drive_filename=drive_plot_filename
            )
            if plot_file_obj:
                plot_links_for_slides[f'{tissue}_{sex}_all_trajectories'] = plot_file_obj

            # Plot 2: Clusters with Centroids
            local_plot_path, drive_plot_filename = get_output_paths(
                local_base_dir,
                'figures',
                tissue,
                sex,
                'clusters_with_centroids',
                'png'
            )
            plot_file_obj = plot_clusters_with_centroids(
                data_with_clusters,
                processed_df,
                centroids,
                optimal_k,
                config,
                tissue,
                sex,
                local_path=local_plot_path if export_local_figures_and_csv else None,
                g_helper=g_helper,
                drive_folder_id=drive_fig_folder_id,
                drive_filename=drive_plot_filename
            )
            if plot_file_obj:
                plot_links_for_slides[f'{tissue}_{sex}_clusters_with_centroids'] = plot_file_obj

            # Plot 3: Centroids Only
            local_plot_path, drive_plot_filename = get_output_paths(
                local_base_dir,
                'figures',
                tissue,
                sex,
                'centroids_only',
                'png',
            )
            plot_file_obj = plot_only_centroids(
                centroids,
                processed_df.columns,
                optimal_k,
                config,
                tissue,
                sex,
                local_path=local_plot_path if export_local_figures_and_csv else None,
                g_helper=g_helper,
                drive_folder_id=drive_fig_folder_id,
                drive_filename=drive_plot_filename
            )
            if plot_file_obj:
                plot_links_for_slides[f'{tissue}_{sex}_centroids_only'] = plot_file_obj

            # Plot 4: Individual Clusters
            local_path_prefix_str, drive_filename_prefix_str = get_output_paths(
                local_base_dir,
                'figures',
                tissue,
                sex,
                'cluster_',
                ''
            )
            local_path_prefix_str = local_path_prefix_str.strip('.')
            drive_filename_prefix_str = drive_filename_prefix_str.strip('.')
            plot_individual_clusters(
                data_with_clusters,
                processed_df,
                centroids,
                optimal_k,
                config,
                tissue,
                sex,
                local_path_prefix=local_path_prefix_str if export_local_figures_and_csv else None,
                g_helper=g_helper,
                drive_folder_id=drive_fig_folder_id,
                drive_filename_prefix=drive_filename_prefix_str
            )

            # Plot 5: Cluster Array
            local_plot_path, drive_plot_filename = get_output_paths(
                local_base_dir,
                'figures',
                tissue,
                sex,
                'cluster_array',
                'png'
            )
            plot_file_obj = plot_cluster_centroids_array(
                data_with_clusters,
                processed_df,
                centroids,
                optimal_k,
                config,
                tissue,
                sex,
                local_path=local_plot_path if export_local_figures_and_csv else None,
                g_helper=g_helper,
                drive_folder_id=drive_fig_folder_id,
                drive_filename=drive_plot_filename
            )
            if plot_file_obj:
                plot_links_for_slides[f'{tissue}_{sex}_cluster_array'] = plot_file_obj
            
            # --- Enrichment and final plot ---
            enrichment_results, merged_genes_df = run_pathway_enrichment(
                processed_df.index,
                labels,
                optimal_k,
                feature_map_buffer,
                config,
                tissue,
                sex,
                local_artifact_dir=output_dir, 
                g_helper=g_helper, 
                drive_artifact_folder_id=drive_artifact_folder_id,
                export_local_csv=export_local_figures_and_csv
            )
            
            if enrichment_results:
                processed_enrichment_results = postprocess_enrichment_results(
                    enrichment_results,
                    tissue,
                    sex,
                    local_artifact_dir=output_dir,
                    g_helper=g_helper,
                    drive_artifact_folder_id=drive_artifact_folder_id,
                    term_size_min=config['analysis_params']['pathway_enrichment']['term_size_min'],
                    term_size_max=config['analysis_params']['pathway_enrichment']['term_size_max'],
                    top_n_pathways=config['analysis_params']['pathway_enrichment']['top_n_pathways']
                )

                if processed_enrichment_results:
                    local_plot_path, drive_plot_filename = get_output_paths(
                        local_base_dir,
                        'figures',
                        tissue,
                        sex,
                        'trajectories_with_wordclouds',
                        'png'
                    )
                    plot_file_obj = plot_trajectories_with_wordclouds(
                        data_with_clusters,
                        processed_df,
                        centroids,
                        processed_enrichment_results,
                        merged_genes_df,
                        optimal_k,
                        config,
                        tissue,
                        sex,
                        local_path=local_plot_path if export_local_figures_and_csv else None,
                        g_helper=g_helper,
                        drive_folder_id=drive_fig_folder_id,
                        drive_filename=drive_plot_filename
                    )
                    if plot_file_obj:
                        plot_links_for_slides[f'{tissue}_{sex}_trajectories_with_wordclouds'] = plot_file_obj

            print(f"Memory usage before cleanup: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
            print(f"\n--- Cleaning up memory for {tissue} {sex} ---")
            plt.close('all')
            # List all large objects created in the loop to be deleted
            vars_to_delete = [
                'data_filtered',
                # 'sex_specific_df_pl',
                'all_clustering_results',
                'processed_df', 
                'optimal_result',
                'data_with_clusters',
                'enrichment_results', 
                'merged_genes_df',
                'processed_enrichment_results'
            ]
            for var_name in vars_to_delete:
                if var_name in locals():
                    exec(f'del {var_name}')
            
            # Manually trigger the garbage collector
            gc.collect()
            print("--- Memory cleanup complete ---")
            print(f"Memory usage after cleanup: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")

        # After processing both sexes for a tissue, clear the full df
        del full_df_pl
        gc.collect()

    # --- Final Reporting ---
    print("\n--- Updating Google Slides Presentation ---")
    if config.get('slides', {}).get('presentation_id') and plot_links_for_slides:
        for placeholder, file_obj in plot_links_for_slides.items():
            if not file_obj or 'id' not in file_obj:
                print(f"  - Skipping placeholder '{placeholder}' due to missing file object or ID.")
                continue

            print(f"  - Replacing placeholder '{{{{{placeholder}}}}}'...")
            try:
                # The webContentLink needs to be modified for direct image replacement
                # A common format is `https://drive.google.com/uc?id=<FILE_ID>`
                file_id = file_obj.get('id')
                clean_url = f"https://drive.google.com/uc?id={file_id}"

                # --- BEGIN DEBUG LOGGING ---
                print(f"    - Presentation ID: {config['slides']['presentation_id']}")
                print(f"    - Placeholder Text: {placeholder}")
                print(f"    - Image URL: {clean_url}")
                # --- END DEBUG LOGGING ---

                g_helper.replace_image_in_slides(
                    config['slides']['presentation_id'],
                    clean_url,
                    placeholder
                )
            except Exception as e:
                print(f"    --> Error updating slide for {placeholder}: {e}")
        print("--- Google Slides Update Complete ---")