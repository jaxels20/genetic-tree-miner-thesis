import pandas as pd
import os
import plotly.express as px
from pandas.plotting import parallel_coordinates
import matplotlib.pyplot as plt


datasets = os.listdir("./best_params")
datasets = [dataset for dataset in datasets if dataset.endswith(".csv")]

dfs = []


for dataset in datasets:
    df = pd.read_csv(f"./best_params/{dataset}")
    
    dfs.append(df)



merged_df = pd.concat(dfs, ignore_index=True)

# Plot
fig = px.parallel_coordinates(
    merged_df,
    dimensions=[col for col in merged_df.columns if col != 'dataset' and col != 'objective'],
    color='objective',
    labels={col: col.replace('_', ' ') for col in merged_df.columns},
)

fig.show()


    
    

