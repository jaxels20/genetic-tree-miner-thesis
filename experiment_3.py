import plotly.express as px
from pandas.plotting import parallel_coordinates
import matplotlib.pyplot as plt
import os

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

# Save the figure in ./experiment_3
fig.write_image("./experiment_3/parallel_coordinates_plot.png", width=800, height=600)
