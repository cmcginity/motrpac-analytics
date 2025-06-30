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
