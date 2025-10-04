# MoTrPAC Analytics Repository


This repository contains tools and pipelines for analyzing omics data from the MoTrPAC (Molecular Transducers of Physical Activity Consortium) studies, focusing on temporal clustering and differential analysis of multiple organ tissues. Its focus is on processing and analyzing ATAC-seq, RNA-seq, proteomics, and metabolomics data from acute and chronic exercise protocols.
## Setup

1. **Environment**: Use the provided `environment.yaml` to set your environment; eg:
   ```
   conda env create -f environment.yaml
   conda activate motrpac-dev
   ```

2. **Dependencies**: Core libraries include Pandas, NumPy, SciPy, and custom modules in `src/dawgpac_analysis`. For Google Cloud integration, ensure credentials are in `secrets/token.json`.

3. **Data**: Raw data is read in from a GCS bucket. Filtered and processed datasets are generated and stored in `data/_filtered/`.

## Usage

- **Run Pipeline**: Execute the main script with a config file, e.g.:
  ```
  python run_pipeline.py --config config/proteomics.yaml
  ```
  This handles data loading, filtering, clustering, and output generation.

- **Notebooks**: Exploratory analyses and testing are in `notebooks/`, such as `atac_6m_liver.ipynb` for ATAC-seq clustering optimization.

- **Output**: Results, figures, and pickled models go to `output/`, timestamped for versioning. Remote data is sent to Google Drive.

## Project Structure

- `config/`: YAML configs for different omics types and analyses.
- `data/`: Input data, filtered subsets, and stashed intermediates.
- `notebooks/`: Jupyter notebooks for development and visualization.
- `output/`: Generated results, figures, and logs.
- `planning/`: Notes on refactoring and future enhancements.
- `src/dawgpac_analysis/`: Core Python modules for temporal clustering, data combination, and utilities.
- `run_pipeline.py`: Entry point for running analyses.

We're continually refining the clustering algorithms and integrating more datasets, aiming to uncover deeper insights into exercise-induced molecular changes—stay tuned for expansions that could reshape our understanding of physiology.

Contributions welcome; check the planning docs for ongoing refactors.
