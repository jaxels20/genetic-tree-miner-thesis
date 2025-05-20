import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
import numpy as np
from src.utils import load_hyperparameters_from_csv

INPUT_FILE = "./data/figure_5a/figure_5a.csv"
OUTPUT_DIR = "./figures"

def plot_data(df):

    df.rename(columns={
        'random_creation_rate': 'Random Creation Rate',
        'elite_rate': 'Elite Rate',
        'population_size': 'Population Size',
        'percentage_of_log': 'Percentage of Log',
        'log_filtering': 'Log Filtering',
        'tournament_size': 'Tournament Size',
        'tournament_rate': 'Tournament Rate',
        'tournament_mutation_rate': 'Mutation Rate',
        'generator': 'Generator',
        'objective': 'Objective',
        'dataset': 'Dataset',
    }, inplace=True)

    skip_cols = ['Dataset', 'Objective', "Generator"]
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

    desired_col_order = ['Tournament Size', 'Mutation Rate', 'Tournament Rate', 'Random Creation Rate', 
                        'Elite Rate',
                        'Population Size',
                        'Log Filtering']

    # Reorder dimensions
    dimensions_dict = {d['label']: d for d in dimensions}
    dimensions = [dimensions_dict[col] for col in desired_col_order]

    custom_ylim = {
        'Tournament Size' : [0.1, 0.3],
        'Tournament Rate' : [0,1],
        'Mutation Rate' : [0,1], 
        'Random Creation Rate' : [0,1],
        'Elite Rate' : [0,1],
        'Population Size' : [20,60], 
        'Log Filtering' : [0.001, 0.1]
    }

    y_tick_vals = {
        'Tournament Size' : [x / 100 for x in range(10, 32, 4)],
        'Tournament Rate' : [x / 100 for x in range(0, 102, 20)],
        'Mutation Rate' : [x / 100 for x in range(0, 102, 20)], 
        'Random Creation Rate' : [x / 100 for x in range(0, 102, 20)],
        'Elite Rate' : [x / 100 for x in range(0, 102, 20)],
        'Population Size' : [x  for x in range(20, 65, 10)],
        'Log Filtering' : [x / 1000 for x in range(0, 102, 20)]
    }


    # Add 'range' to each dimension dict
    for dim in dimensions:
        col_label = dim['label']
        if col_label in custom_ylim:
            dim['range'] = custom_ylim[col_label]
            dim['tickvals'] = y_tick_vals[col_label]

        dim['label'] = col_label.upper().replace(" ", "_")
        if col_label == "Mutation Rate":
            dim['label'] = "MUTATION_PROBABILITY"
        if col_label == "Log Filtering":
            dim['label'] = "INITAL_SAMPLING_RATE"
        
    # Create the parcoords plot
    fig = go.Figure()
    fig = fig.add_trace(
        go.Parcoords(
            line=dict(
                color='black',
            ),
            dimensions=dimensions,
            labelside='bottom',
            labelangle=15
        )
    )

    # Layout adjustments
    fig.update_layout(
        font=dict(family='Courier New', size=14),
        margin=dict(l=60, r=80, t=50, b=90),
        template='simple_white',
        height=400,
        width=850
    )
    # Save the figure
    fig.write_image(os.path.join(OUTPUT_DIR, "figure_5a.png"))

if __name__ == "__main__":
    df = pd.read_csv(INPUT_FILE)
    # normalize elite rate, random creation rate and tournament rate by making it sum up to 1 for each row
    total = df['elite_rate'] + df['random_creation_rate'] + df['tournament_rate']
    df['elite_rate'] = df['elite_rate'] / total
    df['random_creation_rate'] = df['random_creation_rate'] / total
    df['tournament_rate'] = df['tournament_rate'] / total
    plot_data(df)