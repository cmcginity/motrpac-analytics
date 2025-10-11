import os
import subprocess
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Project root
project_root = '/Users/curtismcginity/stanford/research/proj/MoTrPAC/dev'
scripts_dir = os.path.join(project_root, 'src/feast-precawg')

# List of scripts in order (Steps 2-6)
scripts = [
    # 'process_feature_to_gene.py',
    # 'filter_da_datasets.py',
    # 'combine_filtered_da.py',
    # 'merge_cancer.py',
    # 'perform_enrichment.py',
    # 'plot_enrichment.py',
    'hall_perform_enrichment.py',
    'hall_plot_enrichment.py'
]

def run_script(script_name):
    script_path = os.path.join(scripts_dir, script_name)
    try:
        result = subprocess.run(['python', script_path], check=True, capture_output=True, text=True)
        logging.info(f"Successfully ran {script_name}: {result.stdout}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running {script_name}: {e.stderr}")
        raise

if __name__ == "__main__":
    logging.info("Starting pipeline...")
    for script in scripts:
        run_script(script)
    logging.info("Pipeline completed successfully.")

# Documentation:
# This script runs the FEAST pre-CAWG analysis pipeline.
# Ensure environment is activated and GCS credentials are set.
# For testing: Run this script; check outputs in data/_feast-human/ and output/figures/.
# Verify: Mappings have correct merges, filtered DA only include cancer features, enrichment has expected pathways, plots are generated.
