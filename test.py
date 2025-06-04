from src.EventLog import EventLog
from src.Objective import Objective
from src.Discovery import Discovery
from src.utils import load_hyperparameters_from_csv, calculate_percentage_of_log
import time as t
from src.Evaluator import SingleEvaluator
from src.Filtering import Filtering
import json
import os
import pandas as pd

def main():
    df = pd.read_csv("./figures/table_2.csv")
    
    # cast time column to float
    df['Time (s)'].replace('-', '0', inplace=True)
    df['Time (s)'] = df['Time (s)'].astype(float)
    
    df = df[df['Dataset'] != 'Aggregated']
    
    # aggregate acroos datasets
    df = df.groupby(['Discovery Method']).mean(numeric_only=True).reset_index()
    df['Time (s)'].replace(0, '-', inplace=True)
    
    df.to_latex('./figures/table_2_aggregated.tex', index=False, float_format="%.2f")
    
if __name__ == "__main__":
    main()