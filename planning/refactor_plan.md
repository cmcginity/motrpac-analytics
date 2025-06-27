# MoTrPAC Temporal Clustering: Final, Comprehensive Refactoring Plan

### Synopsis of Final Requirements

This plan details the steps to refactor the original `temporal_clustering.py` script into a production-ready pipeline within the new project structure. The final script will achieve the following:

* **Cloud Integration:** Read input data from Google Cloud Storage (GCS) and write all outputs (figures, reports, data artifacts) to a structured folder system in Google Drive.
* **User-Based Authentication:** Operate using personal Google account credentials via an OAuth 2.0 flow.
* **Centralized & Scalable Configuration:** Manage parameters through a `config/` directory, with shared settings in `defaults.yaml` and pipeline-specific settings in `temporal_clustering.yaml`.
* **Modular Codebase:** Separate cloud interaction logic into `src/dawgpac_analysis/google_utils.py` and analysis logic into `src/dawgpac_analysis/temporal_clustering/atac.py`.
* **Standardized Naming:** Employ a helper function to generate consistent, dated filenames and local directory structures.
* **Enhanced Pipeline Logic:** Save all intermediate clustering results to enable future analysis, while preserving the ability to use pre-computed seeds.
* **High-Quality, Styled Plots:** Preserve all existing plot styling and export figures at a high resolution (300 DPI).
* **Automated Reporting:** Loop through all configured tissues and automatically insert the final plots into a specified Google Slides presentation.

### Prerequisite: Version Control

This plan assumes you have already committed your new project structure to a dedicated git branch (e.g., `feat/pipeline-refactor`). All the following work should be done on this branch.

---

### Step 1: Populate Core Utility and Configuration Files

*This step ensures the static parts of the project are correctly in place before we begin refactoring the logic.*

1.  **Populate `src/dawgpac_analysis/google_utils.py`:**
    * This file centralizes all Google API interactions. Paste the following complete code into this file.

    ```python
    # src/dawgpac_analysis/google_utils.py
    import io
    import os
    import pickle
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload
    from google.cloud import storage

    SCOPES = [
        '[https://www.googleapis.com/auth/drive](https://www.googleapis.com/auth/drive)',
        '[https://www.googleapis.com/auth/presentations](https://www.googleapis.com/auth/presentations)',
        '[https://www.googleapis.com/auth/devstorage.read_only](https://www.googleapis.com/auth/devstorage.read_only)'
    ]
    
    def get_user_credentials():
        creds = None
        if os.path.exists('token.json'):
            creds = Credentials.from_authorized_user_file('token.json', SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
                creds = flow.run_local_server(port=0)
            with open('token.json', 'w') as token:
                token.write(creds.to_json())
        return creds

    class GoogleCloudHelper:
        def __init__(self):
            self.credentials = get_user_credentials()
            self.drive_service = build('drive', 'v3', credentials=self.credentials)
            self.slides_service = build('slides', 'v1', credentials=self.credentials)
            self.storage_client = storage.Client(credentials=self.credentials)

        def download_gcs_file_as_stringio(self, bucket_name, source_blob_name):
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(source_blob_name)
            content = blob.download_as_string().decode('utf-8')
            return io.StringIO(content)

        def create_drive_folder(self, folder_name, parent_folder_id=None):
            parents = [parent_folder_id] if parent_folder_id else []
            file_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder',
                'parents': parents
            }
            folder = self.drive_service.files().create(body=file_metadata, fields='id').execute()
            return folder.get('id')

        def upload_buffer_to_drive(self, buffer, folder_id, file_name, mimetype):
            buffer.seek(0)
            file_metadata = {'name': file_name, 'parents': [folder_id]}
            media = MediaIoBaseUpload(buffer, mimetype=mimetype, resumable=True)
            file = self.drive_service.files().create(
                body=file_metadata, media_body=media, fields='id, webContentLink'
            ).execute()
            print(f"Uploaded '{file_name}' to Drive. ID: {file.get('id')}")
            return file

        def replace_image_in_slides(self, presentation_id, image_url, placeholder_text):
            requests = [{'replaceAllShapesWithImage': {
                'imageUrl': image_url, 'replaceMethod': 'CENTER_INSIDE',
                'containsText': {'text': f'{{{{{placeholder_text}}}}}', 'matchCase': False}
            }}]
            self.slides_service.presentations().batchUpdate(
                presentationId=presentation_id, body={'requests': requests}
            ).execute()
    ```

2.  **Populate `config/defaults.yaml`:**
    * This file holds settings shared across all potential pipelines.

    ```yaml
    # config/defaults.yaml
    drive:
      root_output_folder_id: "your_google_drive_folder_id_for_all_outputs"
    slides:
      presentation_id: "your_google_slides_presentation_id_here"
    
    local_base_output_dir: "output"

    plotting:
      dpi: 300 # Dots per inch for high-quality figures
    ```

3.  **Populate `config/temporal_clustering.yaml`:**
    * This file holds parameters specific to the ATAC-seq temporal clustering analysis.

    ```yaml
    # config/temporal_clustering.yaml
    gcs_paths:
      da_data: "path/to/your/da_data/"
      feature_map_data: "path/to/your/feature_map_data/"

    tissue_files:
      liver:
        da_file: "rat-acute-06_t68-liver_epigen-atac-seq_da_timewise-deseq2-phase-frip-t2f_v2.0.txt"
        feature_map_file: "rat-acute-06_epigenomics_metadata_rat-acute-06_t68-liver_epigen-atac-seq_metadata_features_v1.0.txt"
      # ... add other tissues here ...

    analysis_params:
      significance_threshold: 0.05
      clustering:
        k_start: 3
        k_end: 15
        fuzziness: 1.2
        use_precomputed_seeds: false
        num_random_seeds: 1
      pathway_enrichment:
        term_size_min: 0
        term_size_max: 400
        top_n_pathways: 75
        
    columns:
      p_value_adj: "adj_p_value"
      sex: "sex"
      feature_ID: "feature_ID"
      timepoint: "timepoint"
      logFC: "logFC"
    ```

### Step 2: Refactor the Analysis Script (`src/dawgpac_analysis/temporal_clustering/atac.py`)

*This is the main task. Work through `atac.py`, refactoring each piece as you go.*

1.  **Initial Setup (Imports, Helpers, `main` function shell):**
    * Update the import strucutre to reflect the new paths.

    ```python
    # src/dawgpac_analysis/temporal_clustering/atac.py
    import io
    import os
    import pickle
    import pandas as pd
    import polars as pl
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import skfuzzy as fuzz
    from scipy.spatial.distance import cdist
    from gprofiler import GProfiler
    from wordcloud import WordCloud
    from datetime import date
    
    # Note the relative import path for the helper from within the package
    from ..google_utils import GoogleCloudHelper 

    RUN_DATE = date.today().isoformat()
    
    def get_output_paths(local_base_dir, artifact_type, tissue, sex, slug, ext):
        """Generates standardized local path and a clean cloud filename."""
        drive_filename = f"{RUN_DATE}_{tissue}_{sex}_{slug}.{ext}"
        local_path = os.path.join(local_base_dir, artifact_type, RUN_DATE, drive_filename)
        return local_path, drive_filename

    # --- ALL REFACTORED FUNCTIONS FROM YOUR ORIGINAL SCRIPT WILL GO HERE ---
    
    def main(config):
        """Main function to run the ATAC-seq temporal clustering pipeline."""
        # --- THE ORCHESTRATION LOGIC FROM YOUR ORIGINAL main() WILL GO HERE ---
        pass
    ```

2.  **Refactor Data Input & Processing Functions:**
    * Copy `load_and_preprocess_data`, `preprocess_data_for_clustering`, `xie_beni_index`, and `run_cmeans_for_k` into `atac.py`.
    * Modify `load_and_preprocess_data`'s signature to `def load_and_preprocess_data(file_buffer, config, sex):`
    * In that function, replace `pl.read_csv(file_path, ...)` with `pl.read_csv(file_buffer, ...)` and access column names from `config['columns']`.
    * Ensure other functions take the `config` object as a parameter if they need to access analysis settings (e.g., `run_cmeans_for_k` for `fuzziness`).

3.  **Refactor All Data Output & Plotting Functions:**
    * Copy every `plot_*` function into `atac.py`.
    * **Preserve Styling:** All `sns.set_style`, `sns.set_context`, color palettes, and other styling calls must be kept exactly as they are.
    * **Change Signatures:** Update signatures to accept pre-formatted paths and the config object.
        * *Example Old:* `plot_clusters_with_centroids(data_with_clusters, ..., output_dir, tissue, sex, optimal_k)`
        * *Example New:* `def plot_clusters_with_centroids(..., config, local_path=None, g_helper=None, drive_folder_id=None, drive_filename=None):`
    * **Implement Hybrid Save Logic** in place of the old `plt.savefig(plot_path)` call:
        ```python
        # Example save logic to replace the old savefig call
        plt.tight_layout() # Ensure layout is good before saving

        if local_path:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            plt.savefig(local_path, dpi=config['plotting']['dpi'], bbox_inches='tight')

        plot_file_obj = None
        if g_helper and drive_folder_id and drive_filename:
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=config['plotting']['dpi'], bbox_inches='tight')
            plot_file_obj = g_helper.upload_buffer_to_drive(buffer, drive_folder_id, drive_filename, 'image/png')
        
        plt.close()
        return plot_file_obj # Return for the Slides step
        ```

4.  **Refactor `find_optimal_clusters` (The most complex part):**
    * Copy the original function into `atac.py`.
    * **Change Signature:** to `def find_optimal_clusters(processed_df, config, tissue, sex, local_artifact_dir=None, g_helper=None, drive_artifact_folder_id=None):`
    * **Update Seed Loading Logic:** Modify the `use_precomputed_seeds` block to load the `best_seeds.pkl` file from `local_artifact_dir`.
    * **Save Intermediate Results:** In the `for k in CLUSTER_RANGE:` loop, after `run_cmeans_for_k`, add the hybrid save logic for the `best_result_for_k` object.
    * **Save Final Seeds:** At the end of the function, use the same hybrid save logic for the `best_seeds` dictionary.

5.  **Refactor Pathway Enrichment (`run_pathway_enrichment` and `postprocess_enrichment_results`):**
    * Copy these functions into `atac.py`.
    * Update them to use the hybrid save model for their `.csv` and `.pkl` outputs, accepting `local_path`, `drive_folder_id`, etc.

6.  **Refactor the `main` Orchestration Logic:**
    * Copy the logic from your original `main` function into the `main(config)` function in `atac.py`.
    * **Initialization:**
        ```python
        g_helper = GoogleCloudHelper()
        drive_run_folder_id = g_helper.create_drive_folder(RUN_DATE, config['drive']['root_output_folder_id'])
        drive_fig_folder_id = g_helper.create_drive_folder("figures", drive_run_folder_id)
        drive_artifact_folder_id = g_helper.create_drive_folder("clustering_artifacts", drive_run_folder_id)
        plot_links_for_slides = {}
        CLUSTER_RANGE = list(range(config['analysis_params']['clustering']['k_start'], config['analysis_params']['clustering']['k_end'] + 1))
        ```
    * **Main Loops:** Implement the `for tissue...` and `for sex...` loops. Inside, orchestrate the analysis:
        1.  Download data from GCS into memory buffers.
        2.  Call `load_and_preprocess_data`.
        3.  Call `find_optimal_clusters`.
        4.  For every plot or artifact to be saved, call `get_output_paths` to generate names, then call the appropriate refactored plotting/saving function.
        5.  Store the returned `webContentLink` from plot uploads in the `plot_links_for_slides` dictionary.
    * **Final Reporting:** At the end of `main(config)`, add the loop to update Google Slides.

### Step 3: Implement the Pipeline Entry Point (`run_pipeline.py`)

1.  **Populate `run_pipeline.py` at the root of your `dev/` directory:**
    ```python
    # run_pipeline.py
    import yaml
    import argparse
    import os
    from dotenv import load_dotenv
    from src.dawgpac_analysis.temporal_clustering import atac

    def load_config(pipeline_name):
        """Loads default and pipeline-specific configs and merges them."""
        with open("config/defaults.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        config_path = f"config/{pipeline_name}.yaml"
        with open(config_path, 'r') as f:
            pipeline_config = yaml.safe_load(f)
        
        config.update(pipeline_config)
        return config

    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Run a MoTrPAC analysis pipeline.")
        parser.add_argument(
            "pipeline_config", 
            type=str,
            help="The name of the pipeline config file to run (e.g., 'temporal_clustering')."
        )
        args = parser.parse_args()

        print(f"Loading configuration for '{args.pipeline_config}'...")
        config = load_config(args.pipeline_config)
        
        load_dotenv()
        config['gcs_bucket_name'] = os.getenv('GCS_BUCKET_NAME')

        if args.pipeline_config == 'temporal_clustering':
            print("Starting ATAC Temporal Clustering Pipeline...")
            atac.main(config) 
            print("Pipeline finished successfully.")
        else:
            print(f"Error: Unknown pipeline '{args.pipeline_config}'")
    ```

### Step 4: Execution

1.  **Activate Conda Environment:**
    `conda activate motrpac_analysis`
2.  **Run the Pipeline from the `dev/` directory:**
    ```bash
    python run_pipeline.py temporal_clustering
    ```