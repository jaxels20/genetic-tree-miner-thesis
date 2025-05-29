import pandas as pd
import os

INPUT_DIR = "./data/table_2/evaluation_results/"
OUTPUT_FILE = "./figures/table_2"

def consolidate_results():
    # list all csv files in a directory
    csv_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.csv')]
    
    df = pd.DataFrame()
    for file in csv_files:
        if "GTM" in file:
            gtm_df = get_genetic_miner_median_run(file)
            df = pd.concat([df, gtm_df], ignore_index=True)
        else:
            loaded_df = pd.read_csv(INPUT_DIR + file)
            df = pd.concat([df, loaded_df], ignore_index=True)
    
    # df manipulation
    column_order = ['Dataset', 'Discovery Method', 'F1 Score', 'Log Fitness', 'Precision', 'Generalization', 'Simplicity', 'Objective Fitness', 'Time (s)']
    df = df[column_order]
    df.sort_values(by=['Dataset', 'Discovery Method'], inplace=True)
    df['Time (s)'] = df['Time (s)'].replace('-', 0)
    
    agg_df = df.copy()
    agg_df = agg_df.groupby('Discovery Method').agg({
        'F1 Score': 'mean',
        'Log Fitness': 'mean',
        'Precision': 'mean',
        'Objective Fitness': 'mean',
        'Generalization': 'mean',
        'Simplicity': 'mean',
        'Time (s)': 'mean'
    }).reset_index()
    agg_df['Dataset'] = 'Aggregated'
    
    collected_df = pd.concat([df, agg_df], ignore_index=True)
    collected_df['Time (s)'] = collected_df['Time (s)'].round(2).astype(str)
    collected_df.loc[(collected_df['Discovery Method'] == 'SM') & (collected_df['Time (s)'] == '0.0'), 'Time (s)'] = '-'
    collected_df['Objective Fitness'] = collected_df['Objective Fitness'] / 100
    
    return collected_df
 
def get_genetic_miner_median_run(file_name):
    df = pd.read_csv(INPUT_DIR + file_name)
    median_idx = (
        df
        .groupby('Dataset')['Objective Fitness']
        .apply(lambda x: (x - x.median()).abs().idxmin())    # idxmin returns the index of the first occurrence of the minimum value
    )
    df = df.loc[median_idx].reset_index(drop=True)
    
    return df
    
if __name__ == "__main__":
    df = consolidate_results()
    df.to_csv(OUTPUT_FILE + ".csv", index=False, float_format="%.2f")
    df.to_latex(OUTPUT_FILE + ".tex", index=False, float_format="%.2f")
