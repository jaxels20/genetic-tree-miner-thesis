import os
import sys
import numpy as np
import plotly.graph_objects as go
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

INPUT_DATA = "./data/figure_7/data.csv"
OUTPUT_DIR = "./figures"

def create_plot_archived(df):
    # Define the column used for coloring
    color_col = 'objective_fitness'

    # Drop it from the list of dimensions
    dimension_cols = [col for col in df.columns if col not in [color_col, 'dataset']]

    # Define custom axis limits and ticks
    custom_ylim =  {
        'random_creation_rate': [0, 1],
        'elite_rate': [0, 1],
        'population_size': [20, 70],
        'tournament_size': [0.1, 0.3],
        'log_filtering': [0.001, 0.200],
        'tournament_rate': [0, 1],
        'tournament_mutation_rate': [0, 1],
    }

    y_tick_vals = {
        'random_creation_rate': [x / 100 for x in range(0, 102, 10)],
        'elite_rate': [x / 100 for x in range(0, 102, 10)],
        'population_size': list(range(20, 75, 5)),
        'tournament_size': [x / 100 for x in range(10, 31, 2)],
        'log_filtering': [x / 1000 for x in range(1, 202, 20)],
        'tournament_rate': [x / 100 for x in range(0, 102, 10)],
        'tournament_mutation_rate': [x / 100 for x in range(0, 102, 10)],
    }

    # return unique values of specific column in df
    unique_datasets = df["dataset"].unique()
    
    for dataset in unique_datasets:
        # filter the dataframe for the current dataset
        df_dataset = df[df["dataset"] == dataset]
        
        # Build dimensions
        dimensions = []
        for col in dimension_cols:
            values = df_dataset[col].tolist()
            dim = dict(
                label=col.replace('_', ' ').title(),
                values=[x if not np.isnan(x) else None for x in values]
            )
            if col in custom_ylim:
                dim['range'] = custom_ylim[col]
                dim['tickvals'] = y_tick_vals[col]
            dimensions.append(dim)

        # Create parallel coordinates plot
        fig = go.Figure(
            data=go.Parcoords(
                line=dict(
                    color=df_dataset[color_col],
                    colorscale='Viridis',
                    cmin=0.6,
                    cmax=1.0,
                    colorbar=dict(
                        title=dict(
                            text='Obj. Fitness', 
                            font=dict(size=16)
                        ),
                        tickfont=dict(size=16)
                    )
                ),
                dimensions=dimensions,
                labelside='bottom',
                labelangle=15
            )
        )

        # Layout settings
        fig.update_layout(
            font=dict(family='Times New Roman', size=18),
            margin=dict(l=80, r=50, t=10, b=50),
            template='simple_white'
        )

        # make a directory of output_dir if it does not exist
        if not os.path.exists(f"{OUTPUT_DIR}/figure_7"):
            os.makedirs(f"{OUTPUT_DIR}/figure_7")
        
        # Save to file
        fig.write_image(f"{OUTPUT_DIR}/figure_7/{dataset}.pdf")

def create_plot(df, baseline):
    fig = go.Figure()
    color = "lightgrey"

    # Compute IQR per Dataset
    iqr_df = (
        df.groupby('dataset')['objective_fitness']
        .agg(lambda x: x.quantile(1) - x.quantile(0))
        .reset_index(name='IQR')
    )

    # Sort datasets by IQR
    sorted_datasets = iqr_df.sort_values('IQR')['dataset'].tolist()

    fig.add_trace(go.Scatter(
        x=df['dataset'],
        y=df['objective_fitness'],
        mode='markers',
        name='Sampled',
        marker=dict(color=color, size=6, line=dict(width=1, color='black'))
    ))
    
    # baseline
    fig.add_trace(go.Scatter(
        x=baseline['Dataset'],
        y=baseline['Objective Fitness'] * 100,
        mode='markers',
        name='Default',
        marker=dict(color='red', size=6, line=dict(width=1, color='black')),
    ))

    # Update x-axis order
    fig.update_layout(
        xaxis=dict(categoryorder='array', categoryarray=sorted_datasets)
    )

    # Layout adjustments
    fig.update_layout(
        boxmode='group',  # group boxes of same x-axis value
        font=dict(family='Times New Roman', size=20),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            xanchor='center',
            y=0.95,        # slightly above the top of the plot
            x=0.5
        ),
        xaxis_title='Dataset',
        yaxis_title='Objective Fitness',
        margin=dict(l=0, r=0, t=0, b=150),
        template='simple_white',
    )
    
    # Rotate x-axis labels
    fig.update_xaxes(tickangle=45, tickmode='array', tickvals=sorted_datasets)
    fig.update_yaxes(range=[80, 100])
    
    # save the plot
    fig.write_image(f"{OUTPUT_DIR}/figure_7.pdf")   

def load_baseline_df(file_path):
    df = pd.read_csv(file_path)
    df = df[df['Discovery Method'] == 'GTM-300']
    df = df[['Discovery Method', 'Dataset', 'Objective Fitness']]
    df = df[df['Dataset'] != 'Aggregated']
    return df

if __name__ == "__main__":
    data = pd.read_csv(INPUT_DATA)
    data['objective_fitness'] = data['objective_fitness'] * 100
    baseline = load_baseline_df('./figures/table_2.csv')

    create_plot(data, baseline)



