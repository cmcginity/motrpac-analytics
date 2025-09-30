
import polars as pl
import os
import glob

def combine_and_export(file_list, output_path, format='csv', mini=False, common_cols=None):
    if not file_list:
        return
    
    df_list = []
    for f in file_list:
        if format == 'csv':
            df = pl.read_csv(f)
        elif format == 'parquet':
            df = pl.read_parquet(f)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Cast all columns to String for consistency
        df = df.with_columns([pl.col(c).cast(pl.String) for c in df.columns])
        
        if mini and common_cols:
            # Select only existing common columns (ignore missing)
            existing_cols = [c for c in common_cols if c in df.columns]
            df = df.select(existing_cols)
        
        df_list.append(df)
    
    if mini:
        combined_df = pl.concat(df_list, how='vertical')
    else:
        combined_df = pl.concat(df_list, how='diagonal')
    
    if format == 'csv':
        combined_df.write_csv(output_path)
    elif format == 'parquet':
        combined_df.write_parquet(output_path)
    
    print(f"Exported combined file to {output_path}")

def main(format='csv'):
    base_dir = 'data/_filtered'
    sex_groups = ['_all', '_male', '_female']
    ome_map = {'epigen-atac-seq': 'atac', 'prot-ph': 'prot-ph', 'transcript-rna-seq': 'rna'}
    common_cols = ['tissue', 'assay', 'sex', 'feature_id', 'logFC', 'logFC_se', 'p_value', 'adj_p_value', 'timepoint']
    
    for sex in sex_groups:
        group_dir = os.path.join(base_dir, sex)
        files = glob.glob(os.path.join(group_dir, '*.*'))
        
        if not files:
            print(f"No files found in {group_dir}. Skipping.")
            continue
        
        ome_files = {ome: [] for ome in ome_map.values()}
        all_files = []
        
        for file in files:
            filename = os.path.basename(file)
            matched = False
            for key, ome in ome_map.items():
                if key in filename:
                    ome_files[ome].append(file)
                    all_files.append(file)
                    matched = True
                    break
            if not matched:
                print(f"Warning: No OME match for {filename}. Skipping.")
        
        # Combine per OME
        for ome, file_list in ome_files.items():
            if file_list:
                output_path = os.path.join(group_dir, f"{sex.lstrip('_')}_{ome}_significant_features.{format}")
                combine_and_export(file_list, output_path, format, mini=False)
        
        # Combine all with two versions
        if all_files:
            # Full version
            full_output_path = os.path.join(group_dir, f"{sex.lstrip('_')}_significant_features.{format}")
            combine_and_export(all_files, full_output_path, format, mini=False)
            
            # Mini version
            mini_output_path = os.path.join(group_dir, f"{sex.lstrip('_')}_significant_features_mini.{format}")
            combine_and_export(all_files, mini_output_path, format, mini=True, common_cols=common_cols)

if __name__ == "__main__":
    main()
