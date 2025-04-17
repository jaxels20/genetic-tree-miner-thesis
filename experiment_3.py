import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
import numpy as np


hyperparameters = pd.read_csv("./best_parameters_all_datasets.csv", index_col=0)
df = hyperparameters.copy()

df.loc[df['mutator'] == 'NonTournament', 'tournament_size'] = -1_000_000_000
df.loc[df['generator'] == 'Sequential', 'log_filtering'] = 1_000_000_000
df.loc[df['generator'] == 'Buttom', 'log_filtering'] = -1_000_000_000

# Remove rows where 'objective' is NaN, since that's used for coloring
df = df[df['objective'].notna()]

df.rename(columns={
    'random_creation_rate': 'Random Creation Rate',
    'mutation_rate': 'Mutation Rate',
    'crossover_rate': 'Crossover Rate',
    'elite_rate': 'Elite Rate',
    'population_size': 'Population Size',
    'percentage_of_log': 'Percentage of Log',
    'log_filtering': 'Log Filtering',
    'tournament_size': 'Tournament Size',
    'mutator': 'Mutator',
    'generator': 'Generator',
    'objective': 'Objective',
    'dataset': 'Dataset',
}, inplace=True)
df['Mutator'].replace({
    'Tournament': 'Tourn.',
    'NonTournament': 'NonTourn.',
}, inplace=True)


skip_cols = ['Dataset', 'Objective']
dimension_cols = [col for col in df.columns if col not in skip_cols]

dimensions = []
label_encoders = {}

for col in dimension_cols:
    col_data = df[col]

    if col_data.dtype == 'object' or col_data.dtype.name == 'category':
        # Label encode string columns
        le = LabelEncoder()
        valid_mask = col_data.notna()
        df.loc[valid_mask, col] = le.fit_transform(col_data[valid_mask])
        df[col] = df[col].astype(float)  # Parcoords needs float type
        label_encoders[col] = le

        # Build tickvals and ticktext for category axis
        tickvals = list(range(len(le.classes_)))
        ticktext = list(le.classes_)

        values = df[col].tolist()
        values = [x if not np.isnan(x) else None for x in values]  # Replace NaN with None
        dimensions.append(dict(
            label=col,
            values=values,
            tickvals=tickvals,
            ticktext=ticktext
        ))
    else:
        # Numeric column
        values = df[col].tolist()
        values = [x if not np.isnan(x) else None for x in values]  # Replace NaN with None
        dimensions.append(dict(
            label=col.replace('_', ' '),
            values=values
        ))

desired_col_order = ['Tournament Size', 'Mutator', 'Random Creation Rate', 
                    'Mutation Rate', 'Crossover Rate', 'Elite Rate',
                    'Population Size', 'Percentage of Log', 'Generator',
                    'Log Filtering']

# Reorder dimensions
dimensions_dict = {d['label']: d for d in dimensions}

dimensions = [dimensions_dict[col] for col in desired_col_order]

custom_ylim = {
    'Tournament Size' : [0.3, 0.5], 
    'Random Creation Rate' : [0,1],
    'Mutation Rate' : [0,1], 
    'Crossover Rate' : [0,1], 
    'Elite Rate' : [0,1],
    'Population Size' : [20,60], 
    'Percentage of Log' : [0.01, 0.3], 
    'Log Filtering' : [0.001, 0.1]
}

y_tick_vals = {
    'Tournament Size' : [x / 100 for x in range(30, 52, 2)], 
    'Random Creation Rate' : [x / 100 for x in range(0, 102, 10)],
    'Mutation Rate' : [x / 100 for x in range(0, 102, 10)], 
    'Crossover Rate' : [x / 100 for x in range(0, 102, 10)], 
    'Elite Rate' : [x / 100 for x in range(0, 102, 10)],
    'Population Size' : [x  for x in range(20, 65, 5)],
    'Percentage of Log' : [x / 100 for x in range(1, 30, 3)] + [0.3], 
    'Log Filtering' : [x / 1000 for x in range(1, 102, 20)]
}


# Add 'range' to each dimension dict
for dim in dimensions:
    col_label = dim['label']
    if col_label in custom_ylim:
        dim['range'] = custom_ylim[col_label]
        dim['tickvals'] = y_tick_vals[col_label]

# Create the parcoords plot
fig = go.Figure(
    data=go.Parcoords(
        line=dict(
            color='black',
            showscale=False
        ),
        dimensions=dimensions,
        labelside='bottom',
        labelangle=15
    )
)


    # Layout adjustments
fig.update_layout(
    font=dict(family='Arial', size=14),
    # legend=dict(
    #     orientation='h',
    #     yanchor='bottom',
    #     xanchor='center',
    #     y=1.05,        # slightly above the top of the plot
    #     x=0.5
    # ),
    margin=dict(l=60, r=80, t=50, b=90),
    template='simple_white',
    height=500,
    width=900
)

fig.write_image("./experiment_3/plot.pdf")
