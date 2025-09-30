
import polars as pl
import os
import io
import yaml
import gc
from dotenv import load_dotenv
from google_utils import GoogleCloudHelper

def load_and_preprocess_data_atac(file_buffer, config):
    """
    Loads, parses, and preprocesses the input data from a buffer using Polars.
    Does NOT perform significance filtering.
    """
    print("Loading and parsing data from buffer.")

    p_val_col = config["columns"]["p_value_adj"]
    sex_col = config["columns"]["sex"]
    feature_col = config["columns"]["feature_id"]
    significance_threshold = config['analysis_params']['significance_threshold']
    
    try:
        # Use Polars for fast, multi-threaded CSV reading from in-memory buffer.
        file_buffer.seek(0)
        df_pl = pl.read_csv(file_buffer, separator="\t")

        # 1. Standardize data types FIRST. This is crucial for filtering.
        df_pl = df_pl.with_columns(
            pl.when(pl.col(p_val_col) == "NA")
            .then(pl.lit(1.0, dtype=pl.Float64))
            .otherwise(pl.col(p_val_col).cast(pl.Float64, strict=False))
            .alias(p_val_col)
        ).drop_nulls(subset=[p_val_col])

        # 2. Derive timepoint from the 'contrast' column.
        df_pl = df_pl.with_columns(
            pl.col("contrast")
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

def load_and_preprocess_data_prot_ph(file_buffer, config):
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
            pl.col("contrast")
            .str.split("-")
            .list.get(0)
            .str.split("acute_")
            .list.get(1)
            .str.split("_pass")
            .list.get(0)
            .alias("timepoint")
        ) 
        
        print(f"  - Parsed initial dataframe with shape: {df_pl.shape}")
        return df_pl

    except Exception as e:
        print(f"An error occurred during data loading with Polars: {e}")
        # Return an empty DataFrame to allow the pipeline to continue gracefully
        return pl.DataFrame()

def load_and_preprocess_data_rna(file_buffer, config):
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
            pl.col("contrast")
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

def main():
    print("Starting Filtered Data Extraction...")
    
    load_dotenv()
    
    export_dir = 'data/_filtered'
    export_dir_all = 'data/_filtered/_all'
    # config_path = 'config/temporal_clustering.yaml' # atac
    # config_path = 'config/proteomics.yaml' # prot_ph
    config_path = 'config/rna.yaml' # rna
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config['gcs_bucket_name'] = os.getenv('GCS_BUCKET_NAME')
    print(f"GCS bucket name: {config['gcs_bucket_name']}")
    
    gcs_project_id = config.get('gcs_project_id')
    gcs_bucket_name = config['gcs_bucket_name']
    da_data_path = config['gcs_paths']['da_data']
    tissue_files = config['tissue_files']
    significance_threshold = config['analysis_params']['significance_threshold']
    sex_col = config['columns']['sex']
    p_val_col = config['columns']['p_value_adj']
    feature_col = config['columns']['feature_id']

    # Determine ome type from config_path
    use_atac = False
    use_prot_ph = False
    use_rna = False
    if 'temporal_clustering' in config_path:
        use_atac = True
    elif 'proteomics' in config_path:
        use_prot_ph = True
    elif 'rna' in config_path:
        use_rna = True
    else:
        raise ValueError("Unknown config")
    
    os.makedirs(export_dir, exist_ok=True)
    
    g_helper = GoogleCloudHelper(gcs_quota_project_id=gcs_project_id)
    
    for tissue, files in tissue_files.items():

        # Download data from GCS into memory buffers
        print("Downloading data from Google Cloud Storage...")
        try:
            da_file = files['da_file']
            da_gcs_path = os.path.join(da_data_path, da_file)
            
            print(f"Processing {tissue}: {da_gcs_path}")
            da_buffer = g_helper.download_gcs_file_as_stringio(gcs_bucket_name, da_gcs_path)
            print("Data downloaded successfully.")
        except Exception as e:
            print(f"Error downloading data from GCS: {e}")
            continue
        
        # Call appropriate preprocessing
        if use_atac:
            full_df_pl = load_and_preprocess_data_atac(da_buffer, config)
        elif use_prot_ph:
            full_df_pl = load_and_preprocess_data_prot_ph(da_buffer, config)
        elif use_rna:
            full_df_pl = load_and_preprocess_data_rna(da_buffer, config)
        
        del da_buffer
        gc.collect()
        
        if full_df_pl.is_empty():
            print(f"No data loaded for {tissue}. Skipping.")
            continue

        # filter by significance
        filtered_lazy_df = full_df_pl.lazy().filter(
                (pl.col(p_val_col).min().over(feature_col) < significance_threshold)
            )
        filtered_df_pl = filtered_lazy_df.collect()

        # export to csv
        os.makedirs(export_dir_all, exist_ok=True)
        filename = f"{tissue}_{da_file.replace('.txt', '.csv')}"
        full_path = os.path.join(export_dir_all, filename)
        filtered_df_pl.write_csv(full_path)
        print(f"Exported to {full_path}")

        del full_df_pl
        gc.collect()
        
        for sex in ['male', 'female']:
            sex_specific_filtered_lazy_df = filtered_df_pl.lazy().filter(
                (pl.col(sex_col) == sex)
            )
            sex_specific_df_pl = sex_specific_filtered_lazy_df.collect()
            
            filename = f"{sex}_{tissue}_{da_file.replace('.txt', '.csv')}"
            export_dir_sex = f"{export_dir}/_{sex}"
            os.makedirs(export_dir_sex, exist_ok=True)
            full_path = os.path.join(export_dir_sex, filename)
            sex_specific_df_pl.write_csv(full_path)
            print(f"Exported to {full_path}")
            
            del sex_specific_df_pl
            gc.collect()
        
        del filtered_df_pl
        gc.collect()
    

    print("Extraction finished successfully.")

if __name__ == "__main__":
    main()
