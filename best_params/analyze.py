import pandas as pd
import os

datasets = os.listdir("./best_params")
datasets = [dataset for dataset in datasets if dataset.endswith(".csv")]



for dataset in datasets:
    df = pd.read_csv(f"./best_params/{dataset}")
    

